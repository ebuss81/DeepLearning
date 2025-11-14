"""
Train a simple 1D CNN on time-series windows.

Example:
  python train_cnn1d.py \
    --dataset my_dummy \
    --dev_path /path/to/dev.pt \
    --test_path /path/to/test.pt \
    --epochs 200 \
    --lr 0.01 \
    --batch_size 32
"""
import numpy as np
from sklearn.metrics import confusion_matrix

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm.auto import tqdm

from data.loaders import load_my_dummy
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
    parser = argparse.ArgumentParser(description='PyTorch 1D-CNN Training (simple)')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.06, help='Weight decay')

    # Scheduler / training
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--metric', type=str, default='f1_macro', choices=["f1_macro", "acc"], help='Metric for early stopping') # todo implemtn till end

    # Dataset
    parser.add_argument('--dataset', type=str, default='my_dummy', choices=['my_dummy'])
    parser.add_argument('--dev_path', type=str, default='Data_raw/2classes/Raw_TS_Classification_dev_2870_samples.pt')
    parser.add_argument('--test_path', type=str, default='Data_raw/2classes/Raw_TS_Classification_test_574_samples.pt')

    # Dataloader
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)

    # Model
    parser.add_argument('--n_layers', type=int, default=4, help='Number of conv blocks')
    parser.add_argument('--d_model', type=int, default=6, help='Channels in conv blocks')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--kernel_size', type=int, default=51, help='Conv kernel size')

    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--history_dir', type=str, default='my_stuff')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()

    # Repro & device
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = (device == 'cuda')

    # ---------------------------
    # Data
    # ---------------------------
    if args.dataset != 'my_dummy':
        raise NotImplementedError("Only --dataset my_dummy is implemented.")

    (train_set, val_set, test_set,
     d_input, d_output, class_weights) = load_my_dummy(
        dev_path=args.dev_path,
        test_path=args.test_path,
        seed=args.seed,
    )

    trainloader, valloader, testloader = create_dataloaders(
        train_set, val_set, test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # ---------------------------
    # Model
    # ---------------------------
    print('==> Building model..')
    model = build_cnn1d(
        d_input=d_input,
        d_output=d_output,
        start_dim=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        kernel_size = args.kernel_size,
        n_mlp = 10,
    ).to(device)

    best_metric = 0.0
    start_epoch = 0

    # ---------------------------
    # Resume
    # ---------------------------
    if args.resume:
        ckpt_path = os.path.join(args.checkpoint_dir, 'ckpt.pth')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        best_metric = checkpoint.get('f1_score_macro', 0.0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"==> Resumed from epoch {start_epoch} | best val metric {best_metric:.2f}%")

    # ---------------------------
    # Loss / Optimizer / Scheduler
    # ---------------------------
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    # Exclude biases & norm layers from weight decay
    param_groups = make_param_groups(model, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---------------------------
    # Training loop
    # ---------------------------
    early_stopper = EarlyStopping(patience=args.patience, mode="max")
    history = []
    best_val_metric = best_metric

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(
            f"Epoch {epoch} | lr={current_lr:.3e} | best val metric={best_val_metric:.2f}% (# {early_stopper.num_bad_epochs} epochs) | train acc {history[-1]['train_acc'] if epoch > 0 else 0:.2f}% | test metric {history[-1]['test_f1_macro'] if epoch > 0 else 0:.2f}%")

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
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_f1_macro": float(val_metrics.get("f1_macro", 0.0)),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "test_f1_macro": float(test_metrics.get("f1_macro", 0.0)),
            "lr": float(current_lr),
        }
        history.append(record)

        val_metric = record["val_f1_macro"]
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            # Save checkpoint on val improvement
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "val_metric": best_val_metric,
                    "epoch": epoch,
                },
                out_dir=args.checkpoint_dir,
                name="ckpt.pth",
            )

        # Early stopping on val_acc
        if early_stopper.step(val_metric, epoch):
            tqdm.write(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best val acc: {early_stopper.best:.2f}%"
            )
            break

    # ---------------------------
    # Save logs
    # ---------------------------
    save_history(history, args.history_dir)
    print(
        f"Saved history to {args.history_dir}/history.json, "
        f"{args.history_dir}/history.csv and {args.history_dir}/history.pt"
    )

    # ---------------------------
    # Confusion matrix on test set (best checkpoint)
    # ---------------------------
    ckpt_path = os.path.join(args.checkpoint_dir, 'ckpt.pth')
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded best checkpoint (val metric = {checkpoint.get('val_metric', 0.0):.2f}%) for confusion matrix.")
    else:
        print("No checkpoint found, using last-epoch model for confusion matrix.")

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in testloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    cm = confusion_matrix(all_targets, all_preds)

    print("Confusion matrix (test set):")
    print(cm)

    os.makedirs(args.history_dir, exist_ok=True)
    cm_path = os.path.join(args.history_dir, "confusion_matrix_test.csv")
    np.savetxt(cm_path, cm, fmt="%d", delimiter=",")
    print(f"Saved confusion matrix to {cm_path}")



if __name__ == "__main__":
    main()
