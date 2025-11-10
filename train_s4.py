import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm.auto import tqdm

from DeepLearning.models.my_s4 import build_s4
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
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')

    # Scheduler / training
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')

    # Dataset
    parser.add_argument('--dataset', type=str, default='my_dummy', choices=['my_dummy'])
    parser.add_argument('--dev_path', type=str, default='/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_dev_3446_samples.pt')
    parser.add_argument('--test_path', type=str, default='/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_test_574_samples.pt')

    # Dataloader
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)

    # Model
    parser.add_argument('--n_layers', type=int, default=8, help='Number of conv blocks')
    parser.add_argument('--d_model', type=int, default=256, help='Channels in conv blocks')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    #parser.add_argument('--kernel_size', type=int, default=15, help='Conv kernel size')

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
    model = build_s4(
        d_input=d_input,
        d_output=d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        lr = args.lr
    ).to(device)

    best_acc = 0.0
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
        best_acc = checkpoint.get('acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"==> Resumed from epoch {start_epoch} | best val acc {best_acc:.2f}%")

    def setup_optimizer(model, lr, weight_decay, epochs):
        """
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.

        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """

        # All parameters in the model
        all_parameters = list(model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )
        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return optimizer, scheduler
    # ---------------------------
    # Loss / Optimizer / Scheduler
    # ---------------------------
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    # Exclude biases & norm layers from weight decay
    param_groups = make_param_groups(model, weight_decay=args.weight_decay) # note sure if needed for S4
    #optimizer = optim.AdamW(param_groups, lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=args.epochs
    #)
    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
    )

    # ---------------------------
    # Training loop
    # ---------------------------
    early_stopper = EarlyStopping(patience=args.patience, mode="max")
    history = []
    best_val_acc = best_acc

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(f"Epoch {epoch} | lr={current_lr:.3e} | best val acc={best_val_acc:.2f}% (# {early_stopper.num_bad_epochs} epochs) | train acc {history[-1]['train_acc'] if epoch>0 else 0:.2f}% | test acc {history[-1]['test_acc'] if epoch>0 else 0:.2f}%")

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
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "lr": float(current_lr),
        }
        history.append(record)

        val_acc = record["val_acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint on val improvement
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "acc": best_val_acc,
                    "epoch": epoch,
                },
                out_dir=args.checkpoint_dir,
                name="ckpt.pth",
            )

        # Early stopping on val_acc
        if early_stopper.step(val_acc, epoch):
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


if __name__ == "__main__":
    main()