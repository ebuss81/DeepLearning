import os
import csv
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    train_set, val_set, test_set,
    batch_size: int,
    num_workers: int,
    device: str,
):
    pin_memory = (device == "cuda")
    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    valloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    testloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return trainloader, valloader, testloader


def save_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path_json = os.path.join(out_dir, "history.json")
    path_csv = os.path.join(out_dir, "history.csv")
    path_pt = os.path.join(out_dir, "history.pt")

    with open(path_json, "w") as f:
        json.dump(history, f, indent=2)

    keys = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc",
        "test_loss", "test_acc",
        "lr",
    ]
    with open(path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for h in history:
            row = {k: h.get(k, None) for k in keys}
            writer.writerow(row)

    torch.save(history, path_pt)


def save_checkpoint(state, out_dir, name="ckpt.pth"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, name))


def make_param_groups(model, weight_decay: float):
    """
    Create param groups where biases and norm parameters have no weight decay.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or "bn" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]