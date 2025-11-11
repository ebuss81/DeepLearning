import os
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset
import pandas as pd

def load_my_dummy(dev_path, test_path, seed=42, val_size=0.2):
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
    groups = pd.read_csv("Data_raw/2classes/Raw_TS_Classification_groups_3446_samples.csv") #note hard coded1
    groups = np.array(groups['plant_id'].values)
    # Combine class + group into a single stratification label
    strat_labels = np.array(list(zip(y_dev.numpy(), groups)))

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=seed
    )
    #train_idx, val_idx = next(sss.split(idxs, y_dev.numpy()))
    train_idx, val_idx = next(sss.split(idxs, strat_labels))

    train_set = TensorDataset(X_dev[train_idx], y_dev[train_idx])
    val_set = TensorDataset(X_dev[val_idx], y_dev[val_idx])
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
