"""
Train a simple 1D CNN on your time-series windows.
Structure mirrors the S4 example but is much simpler.

Run (example):
  python train_cnn1d.py --dataset my_dummy --epochs 200 --lr 0.01 --batch_size 32
"""

import os
import json
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader

from DeepLearning.models.CNN_model import CNN1D

# ---------------------------
# Args
# ---------------------------
import argparse
parser = argparse.ArgumentParser(description='PyTorch 1D-CNN Training (simple)')
# Optimizer
parser.add_argument('--lr', default=0.01, choices=[0.01, 0.1], type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay')
# Scheduler
parser.add_argument('--epochs', default=500, type=int, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='my_dummy', choices=['mnist','cifar10','my_dummy'], type=str)
# Dataloader
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--batch_size', default=64, choices=[16, 32, 64], type=int)
# Model
parser.add_argument('--n_layers', default=16, choices=[2, 4, 6], type=int, help='Number of conv blocks')
parser.add_argument('--d_model', default=256, choices=[32, 64, 128], type=int, help='Channels in conv blocks')
parser.add_argument('--dropout', default=0.2, choices=[0.0, 0.1], type=float)
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0
start_epoch = 0

# ---------------------------
# Data
# ---------------------------
print(f'==> Preparing {args.dataset} data..')
if args.dataset == 'my_dummy':
    # Load dev (train+val pool)
    dev_path  = "/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_dev_1718_samples.pt"
    test_path = "/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_test_286_samples.pt"
    data_dev  = torch.load(dev_path)
    X_dev, y_dev = data_dev["X"].float(), data_dev["y"].long()
    print(f"Dev set: {np.unique(y_dev)} samples")
    # Load test
    data_test = torch.load(test_path)
    X_test, y_test = data_test["X"].float(), data_test["y"].long()

    # Make labels contiguous across dev+test
    all_y = torch.cat([y_dev, y_test])
    classes = sorted(torch.unique(all_y).tolist())
    mapping = {c:i for i,c in enumerate(classes)}
    y_dev  = torch.tensor([mapping[int(v)] for v in y_dev.tolist()], dtype=torch.long)
    y_test = torch.tensor([mapping[int(v)] for v in y_test.tolist()], dtype=torch.long)
    d_output = len(classes)
    # Add feature/channel dim: [N, L] -> [N, L, 1]
    X_dev  = X_dev.unsqueeze(-1)
    X_test = X_test.unsqueeze(-1)

    # Stratified split train/val from dev set
    idxs = np.arange(len(y_dev))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(idxs, y_dev.numpy()))
    train_set = TensorDataset(X_dev[train_idx], y_dev[train_idx])
    val_set   = TensorDataset(X_dev[val_idx],   y_dev[val_idx])
    test_set  = TensorDataset(X_test, y_test)

    # Class weights (inverse frequency; normalized)
    counts = Counter(y_dev.tolist())
    weights = torch.tensor([1.0 / max(1, counts[i]) for i in range(d_output)], dtype=torch.float)
    weights = weights / weights.mean()
    class_weights = weights.to(device)

    d_input = X_dev.shape[-1]  # = 1
else:
    raise NotImplementedError("This script is set up for --dataset my_dummy only.")

# Dataloaders
trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=(device=='cuda'))
valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device=='cuda'))
testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device=='cuda'))

# ---------------------------
# Utils
# ---------------------------
def save_history(history, path_json="my_stuff/history.json", path_csv="my_stuff/history.csv", path_pt="my_stuff/history.pt"):
    os.makedirs(os.path.dirname(path_json), exist_ok=True)

    with open(path_json, "w") as f:
        json.dump(history, f, indent=2)

    keys = ["epoch","train_loss","train_acc","val_loss","val_acc","test_loss","test_acc","lr"]
    with open(path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for h in history:
            row = {k: h.get(k, None) for k in keys}
            writer.writerow(row)

    torch.save(history, path_pt)

# ---------------------------
# Model (simple 1D CNN)
# ---------------------------
print('==> Building model..')
model = CNN1D(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
).to(device)

if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint.get('acc', 0.0)
    start_epoch = checkpoint.get('epoch', 0)

# ---------------------------
# Optimizer / Scheduler / Loss
# ---------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ---------------------------
# Training / Eval
# ---------------------------
def train():
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Train (%d/%d) | Loss: %.3f | Acc: %.2f%%' %
            (batch_idx+1, len(trainloader), running_loss / max(1, total), 100. * correct / max(1, total))
        )

    return {"loss": running_loss / max(1, total), "acc": 100. * correct / max(1, total)}

def evaluate(epoch, dataloader, split_name="val", checkpoint=False):
    global best_acc
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                f'{split_name.capitalize()} ({batch_idx+1}/{len(dataloader)}) | Loss: {running_loss/max(1,total):.3f} | Acc: {100.*correct/max(1,total):.2f}%'
            )

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100. * correct / max(1, total)

    if checkpoint and split_name == "val" and epoch_acc > best_acc:
        os.makedirs('checkpoint', exist_ok=True)
        state = {'model': model.state_dict(), 'acc': epoch_acc, 'epoch': epoch}
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = epoch_acc

    return {"loss": epoch_loss, "acc": epoch_acc}

# ---------------------------
# Loop
# ---------------------------
history = []
val_acc = 0.0
best_val_acc = 0.0
no_improve = 0
patience = 50  # early stop patience

pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    current_lr = optimizer.param_groups[0]["lr"]
    if epoch == 0:
        pbar.set_description(f'Epoch: {epoch} | lr: {current_lr:.3e}')
    else:
        pbar.set_description(f'Epoch: {epoch} | lr: {current_lr:.3e} | Val acc: {val_acc:.3f}')

    train_metrics = train()
    val_metrics = evaluate(epoch, valloader, split_name="val", checkpoint=True)
    test_metrics = evaluate(epoch, testloader, split_name="test", checkpoint=False)

    scheduler.step()

    record = {
        "epoch": epoch,
        "train_loss": float(train_metrics["loss"]),
        "train_acc":  float(train_metrics["acc"]),
        "val_loss":   float(val_metrics["loss"]),
        "val_acc":    float(val_metrics["acc"]),
        "test_loss":  float(test_metrics["loss"]),
        "test_acc":   float(test_metrics["acc"]),
        "lr":         float(current_lr),
    }
    history.append(record)
    val_acc = record["val_acc"]

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs.")
            break

save_history(history)
print("Saved history to my_stuff/history.json / .csv / .pt")
