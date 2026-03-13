import os
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset
import pandas as pd


def normalize_from_train(X_train, X_val, X_test, eps=1e-8):
    """
    Normalize datasets using statistics from the training set only.
    Expected shape: [N, L, C]
    """
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True)
    std = torch.clamp(std, min=eps)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


def load_my_dummy(dev_path, test_path,group_path, seed=42, val_size=0.2):
    """
    Load 'my_dummy' dataset from dev/test .pt files and create train/val/test splits.

    Expected .pt structure:
        {
          "X": Tensor [N, L] or [N, L, 1],
          "y": Tensor [N]
        }
    """
    if not os.path.isfile(dev_path):
        raise FileNotFoundError(f"Dev file not found: {dev_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    dev = torch.load(dev_path, map_location="cpu")
    test = torch.load(test_path, map_location="cpu")

    X_dev, y_dev = dev["X"].float(), dev["y"].long()
    X_test, y_test = test["X"].float(), test["y"].long()
    print()

    # Make labels contiguous across dev+test
    all_y = torch.cat([y_dev, y_test])
    classes = sorted(torch.unique(all_y).tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y_dev = torch.tensor([mapping[int(v)] for v in y_dev.tolist()], dtype=torch.long)
    y_test = torch.tensor([mapping[int(v)] for v in y_test.tolist()], dtype=torch.long)
    print("hi", np.unique(y_test))
    d_output = len(classes)

    # Add feature/channel dim: [N, L] -> [N, L, 1]
    if X_dev.ndim == 2:
        X_dev = X_dev.unsqueeze(-1)
    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(-1)

    # Stratified split train/val from dev set
    idxs = np.arange(len(y_dev))
    groups = pd.read_csv(group_path) #note hard coded1
    groups = np.array(groups['plant_id'].values)
    # Combine class + group into a single stratification label
    strat_labels = np.array(list(zip(y_dev.numpy(), groups)))

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=seed
    )
    #train_idx, val_idx = next(sss.split(idxs, y_dev.numpy()))
    train_idx, val_idx = next(sss.split(idxs, strat_labels))

    X_train = X_dev[train_idx]
    y_train = y_dev[train_idx]

    X_val = X_dev[val_idx]
    y_val = y_dev[val_idx]

    print("BEFORE NORM")
    print("X_train nan:", torch.isnan(X_train).any())
    print("X_val nan:", torch.isnan(X_val).any())
    print("X_test nan:", torch.isnan(X_test).any())

    print("X_train inf:", torch.isinf(X_train).any())
    print("X_val inf:", torch.isinf(X_val).any())
    print("X_test inf:", torch.isinf(X_test).any())

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    X_train, X_val, X_test, mean, std = normalize_from_train(X_train, X_val, X_test)
    print("train nan:", np.isnan(X_train).any())
    print("val nan:", np.isnan(X_val).any())
    print("test nan:", np.isnan(X_test).any())

    print("train inf:", np.isinf(X_train).any())
    print("std min:", std.min())

    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)

    # Class weights (inverse frequency; normalized)
    counts = Counter(y_dev.tolist())
    weights = torch.tensor(
        [1.0 / max(1, counts[i]) for i in range(d_output)],
        dtype=torch.float,
    )
    weights = weights / weights.mean()

    d_input = X_dev.shape[-1]

    print(
        f"Loaded my_dummy:"
        f" classes={d_output}, "
        f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )

    return train_set, val_set, test_set, d_input, d_output, weights
