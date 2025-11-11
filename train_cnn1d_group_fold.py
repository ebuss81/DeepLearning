"""
Train a simple 1D CNN on time-series windows with GroupKFold cross-validation.

Example:
  python train_cnn1d.py \
    --dataset my_dummy \
    --dev_path /path/to/dev.pt \
    --test_path /path/to/test.pt \
    --epochs 200 \
    --lr 0.01 \
    --batch_size 32
"""

import os
import math
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm.auto import tqdm
from torch.utils.data import TensorDataset
from sklearn.model_selection import GroupKFold

from models.cnn1d import build_cnn1d
from engine.loop import train_one_epoch, evaluate
from engine.utils import (
    set_seed,
    save_history,
    save_checkpoint,
    create_dataloaders,
    make_param_groups,
)
from engine.callbacks import EarlyStopping


# ---------------------------
# Argparse
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch 1D-CNN Training (GroupKFold)')

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')

    # Scheduler / training
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs per fold')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')

    # Dataset
    parser.add_argument('--dataset', type=str, default='my_dummy', choices=['my_dummy'])
    parser.add_argument('--dev_path', type=str, default='/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_dev_3446_samples.pt')
    parser.add_argument('--test_path', type=str, default='/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_test_574_samples.pt')

    # GroupKFold
    parser.add_argument('--n_splits', type=int, default=5, help='Number of GroupKFold splits')

    # Dataloader
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)

    # Model
    parser.add_argument('--n_layers', type=int, default=16, help='Number of conv blocks')
    parser.add_argument('--d_model', type=int, default=256, help='Channels in conv blocks')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # General
    parser.add_argument('--resume', '-r', action='store_true', help='[Not supported with GroupKFold] Resume from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--history_dir', type=str, default='my_stuff')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()

    if args.resume:
        raise NotImplementedError("Resume is not supported in GroupKFold mode (multiple folds).")

    # Repro & device
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = (device == 'cuda')

    # ---------------------------
    # Data: load dev + test (expects groups in dev)
    # ---------------------------
    if args.dataset != 'my_dummy':
        raise NotImplementedError("Only --dataset my_dummy is implemented.")

    dev_data = torch.load(args.dev_path, map_location="cpu")
    test_data = torch.load(args.test_path, map_location="cpu")

    X_dev = dev_data["X"].float()      # [N_dev, L] or [N_dev, L, 1]
    y_dev = dev_data["y"].long()      # [N_dev]
    print(X_dev)
    groups_df = pd.read_csv("/DeepLearning/Data_raw/Raw_TS_Classification_groups_3446_samples.csv")
    groups_dev = groups_df["plant_id"]   # [N_dev] group IDs

    X_test = test_data["X"].float()
    y_test = test_data["y"].long()

    # Ensure channel dimensionBest trial:
    #   value (val_acc): 77.54%
    #   params:
    #     lr: 0.003415726157870655
    #     weight_decay: 0.04719056775538778
    #     d_model: 32
    #     n_layers: 6
    #     dropout: 0.1710287837589986
    #     label_smoothing: 0.04919908141530631
    #     batch_size: 32
    if X_dev.ndim == 2:
        X_dev = X_dev.unsqueeze(-1)   # [N_dev, L, 1]
    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(-1)

    d_input = X_dev.shape[-1]
    classes = torch.unique(torch.cat([y_dev, y_test]))
    d_output = len(classes)

    # Class weights (inverse frequency; normalized) based on dev labels
    from collections import Counter
    counts = Counter(y_dev.tolist())
    weights = torch.tensor(
        [1.0 / max(1, counts[i]) for i in range(d_output)],
        dtype=torch.float,
    )
    class_weights = (weights / weights.mean()).to(device)

    # Test set (same for all folds)
    test_set = TensorDataset(X_test, y_test)

    # ---------------------------
    # GroupKFold setup
    # ---------------------------
    gkf = GroupKFold(n_splits=args.n_splits)
    fold_indices = list(gkf.split(X_dev, y_dev, groups_dev))

    # Safety: limit n_layers so pooling never shrinks length to 0
    L0 = X_dev.shape[1]  # sequence length
    max_pools = int(math.log2(L0))  # max times we can halve before hitting 1
    max_layers_from_length = max_pools * 2  # because we pool every 2nd block

    if args.n_layers > max_layers_from_length:
        print(f"[Warning] Requested n_layers={args.n_layers} is too deep for sequence length {L0}. "
              f"Clamping to {max_layers_from_length}.")
        n_layers = max_layers_from_length
    else:
        n_layers = args.n_layers

    # ---------------------------
    # Cross-validation over folds
    # ---------------------------
    all_history = []        # per-epoch, per-fold logs
    fold_results = []       # per-fold summary

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n========== Fold {fold_id+1}/{len(fold_indices)} ==========")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")

        # Build datasets for this fold
        train_set = TensorDataset(X_dev[train_idx], y_dev[train_idx])
        val_set   = TensorDataset(X_dev[val_idx],   y_dev[val_idx])

        # Dataloaders
        trainloader, valloader, testloader = create_dataloaders(
            train_set, val_set, test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        # ---------------------------
        # Model / loss / optim / sched for this fold
        # ---------------------------
        print('==> Building model..')
        model = build_cnn1d(
            d_input=d_input,
            d_output=d_output,
            d_model=args.d_model,
            n_layers=n_layers,
            dropout=args.dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

        param_groups = make_param_groups(model, weight_decay=args.weight_decay)
        optimizer = optim.AdamW(param_groups, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        early_stopper = EarlyStopping(patience=args.patience, mode="max")

        best_val_acc = 0.0
        fold_history = []  # only this fold
        ckpt_name = f"ckpt_fold{fold_id}.pth"

        # ---------------------------
        # Training loop for this fold
        # ---------------------------
        for epoch in range(args.epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"[Fold {fold_id+1}] Epoch {epoch} | lr={current_lr:.3e} | "
                f"best val acc={best_val_acc:.2f}% for #{early_stopper.num_bad_epochs} epochs"
            )

            train_metrics = train_one_epoch(
                model, trainloader, optimizer, criterion, device
            )
            val_metrics = evaluate(
                model, valloader, criterion, device, split_name="val"
            )
            test_metrics = evaluate(
                model, testloader, criterion, device, split_name="test"
            )

            scheduler.step()

            record = {
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_acc": float(train_metrics["acc"]),
                "val_loss": float(val_metrics["loss"]),
                "val_acc": float(val_metrics["acc"]),
                "test_loss": float(test_metrics["loss"]),
                "test_acc": float(test_metrics["acc"]),
                "lr": float(current_lr),
            }
            fold_history.append(record)
            all_history.append(record)

            val_acc = record["val_acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save checkpoint on val improvement (per-fold)
                save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "acc": best_val_acc,
                        "epoch": epoch,
                        "fold": fold_id,
                    },
                    out_dir=args.checkpoint_dir,
                    name=ckpt_name,
                )

            # Early stopping on val_acc
            if early_stopper.step(val_acc, epoch):
                tqdm.write(
                    f"[Fold {fold_id+1}] Early stopping at epoch {epoch}. "
                    f"Best val acc: {early_stopper.best:.2f}%"
                )
                break

        # ---------------------------
        # Evaluate best checkpoint of this fold on test
        # ---------------------------
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        if os.path.isfile(ckpt_path):
            best_state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(best_state["model"])
        else:
            print(f"[Warning] No checkpoint found for fold {fold_id} at {ckpt_path}, using last model.")

        # Rebuild testloader (same as before, just to be explicit)
        _, _, testloader = create_dataloaders(
            train_set, val_set, test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        best_test_metrics = evaluate(
            model, testloader, criterion, device, split_name="test_best"
        )

        fold_summary = {
            "fold": fold_id,
            "best_val_acc": float(best_val_acc),
            "test_acc_at_best_val": float(best_test_metrics["acc"]),
        }
        fold_results.append(fold_summary)
        print(
            f"[Fold {fold_id+1}] best val acc = {best_val_acc:.2f}% | "
            f"test acc at best = {best_test_metrics['acc']:.2f}%"
        )

    # ---------------------------
    # Aggregate results across folds
    # ---------------------------
    import numpy as np

    val_accs = np.array([fr["best_val_acc"] for fr in fold_results])
    test_accs = np.array([fr["test_acc_at_best_val"] for fr in fold_results])

    print("\n===== Cross-validation summary =====")
    for fr in fold_results:
        print(
            f"Fold {fr['fold']+1}: best val acc = {fr['best_val_acc']:.2f}% | "
            f"test acc at best = {fr['test_acc_at_best_val']:.2f}%"
        )

    print(
        f"\nMean val acc:  {val_accs.mean():.2f}% ± {val_accs.std():.2f}% "
        f"over {len(fold_results)} folds"
    )
    print(
        f"Mean test acc: {test_accs.mean():.2f}% ± {test_accs.std():.2f}% "
        f"over {len(fold_results)} folds"
    )

    # ---------------------------
    # Save logs (all folds)
    # ---------------------------
    save_history(all_history, args.history_dir)
    print(
        f"\nSaved per-epoch, per-fold history to "
        f"{args.history_dir}/history.json / .csv / .pt"
    )


if __name__ == "__main__":
    main()