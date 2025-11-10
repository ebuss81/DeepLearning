'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import json
import csv
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from torch.utils.data import TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.utils.data import TensorDataset

from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from tqdm.auto import tqdm



# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.001, choices= [0.01, 0.001,0.1], type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=200, type=int, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='my_dummy', choices=['mnist', 'cifar10','my_dummy'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, choices=[16, 32, 64],type = int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=8, choices=[16, 2, 4],type=int, help='Number of layers')
parser.add_argument('--d_model', default=256, choices=[32, 64, 128, 256], type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.2, choices=[0.1, 0.2] ,type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print(f'==> Preparing {args.dataset} data..')
if args.dataset == 'my_dummy':

    data_dev = torch.load("/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_dev_1718_samples.pt")
    X_dev, y_dev = data_dev["X"], data_dev["y"]
    X_dev = X_dev.float().unsqueeze(-1)  # (N, L, 1)m float because of torch operations
    y_dev = y_dev.long()  # (N,) class indices as int64, sine error function needs that


    dataset = TensorDataset(X_dev, y_dev)
    idxs = np.arange(len(y_dev))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(idxs, y_dev.numpy()))
    train_set = TensorDataset(X_dev[train_idx], y_dev[train_idx])
    val_set = TensorDataset(X_dev[val_idx], y_dev[val_idx])

    d_input = X_dev.shape[-1]  # 1 value per time step
    d_output = int(y_dev.max().item()) + 1  # 4 classes: sine, square, sawtooth, noise
    #print(len(train_set), len(val_set), len(test_set))

    test_data = torch.load("/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_test_286_samples.pt")
    X_test, y_test = test_data["X"], test_data["y"]
    X_test = X_test.float().unsqueeze(-1)
    y_test = y_test.long()
    test_set = TensorDataset(X_test, y_test)
    # for cross-entry-loss
    counts = Counter(y_dev.tolist())
    class_weights = torch.tensor([1.0 / counts[i] for i in range(d_output)], dtype=torch.float, device=device)



else: raise NotImplementedError

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)



def save_history(history, path_json="history.json", path_csv="history.csv", path_pt="history.pt"):
    # Save as JSON
    with open(path_json, "w") as f:
        json.dump(history, f, indent=2)

    # Save as CSV
    keys = ["epoch","train_loss","train_acc","val_loss","val_acc","test_loss","test_acc","lr"]
    with open(path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for h in history:
            row = {k: h.get(k, None) for k in keys}
            writer.writerow(row)

    # Save as PyTorch (easiest to reload in Python)
    torch.save(history, path_pt)


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

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

criterion = nn.CrossEntropyLoss(weight = class_weights, label_smoothing=0.1)
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
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

        running_loss += loss.item() * targets.size(0) # sum up batch loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Train (%d/%d) | Loss: %.3f | Acc: %.2f%%' %
            (batch_idx+1, len(trainloader), running_loss / max(1, total), 100. * correct / max(1, total))
        )

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100. * correct / max(1, total)
    return {"loss": epoch_loss, "acc": epoch_acc}

def eval(epoch, dataloader, split_name="val", checkpoint=False):
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

    # checkpoint on validation
    if checkpoint and split_name == "val":
        if epoch_acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': epoch_acc,
                'epoch': epoch,
            }
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = epoch_acc

    return {"loss": epoch_loss, "acc": epoch_acc}

history = []
val_acc = 0.0  # for progress bar text
iter_with_no_improvement = 0  # for early stopping
best_val_acc = 0.0 # for early stopping

pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    # current LR (first param group)
    current_lr = optimizer.param_groups[0]["lr"]
    if epoch == 0:
        pbar.set_description(f'Epoch: {epoch} | lr: {current_lr:.3e}')
    else:
        pbar.set_description(f'Epoch: {epoch} | lr: {current_lr:.3e} | Val acc: {val_acc:.3f}')

    # 1) train
    train_metrics = train()

    # 2) val (with checkpointing)
    val_metrics = eval(epoch, valloader, split_name="val", checkpoint=True)

    # 3) test (optional per epoch; or only after training)
    test_metrics = eval(epoch, testloader, split_name="test", checkpoint=False)

    # 4) scheduler step
    scheduler.step()

    # 5) record
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

    # early stopping (optional)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        iter_with_no_improvement = 0
    else:
        iter_with_no_improvement += 1
        if iter_with_no_improvement >=20:
            print("Early stopping due to no improvement in validation accuracy for 20 epochs.")
            break
# Save history once at the end
save_history(history, path_json="my_stuff/history.json", path_csv="my_stuff/history.csv", path_pt="my_stuff/history.pt")
print("Saved history to history.json / history.csv / history.pt")

