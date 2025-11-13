

from models.cnn1d import build_cnn1d
from models.inception1d import build_inception1d
from models.my_s4 import build_s4
import torch
import torch.optim as optim


def CNN_details(trial,device, d_input, d_output):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True) # logarithmic scale
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True) # logarithmic scale?
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 20, step=2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    kernel_size = trial.suggest_categorical("kernel_size", [5, 9, 19, 29, 39, 49])
    n_mlp = trial.suggest_int("n_mlp", 1, 6)

    # ---- model / loss / optimizer / sched ----
    model = build_cnn1d(
        d_input=d_input,
        d_output=d_output,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        kernel_size=kernel_size,
        n_mlp=n_mlp,
    ).to(device)

    return model, lr, weight_decay, label_smoothing, batch_size

def Inception_details(trial,device, d_input, d_output):
    lr = trial.suggest_float("lr", 1e-4, 1e-1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1)
    # d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512])
    # n_layers = trial.suggest_int("n_layers", 2, 20, step=2)
    # dropout = trial.suggest_float("dropout", 0.0, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    out_channels = trial.suggest_categorical("out_channels", [8, 16, 32, 64])
    bottleneck_channels = trial.suggest_categorical("bottleneck_channels", [2, 4, 8, 16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [5, 9, 19, 29, 39, 49])

    # ---- model / loss / optimizer / sched ----
    model = build_inception1d(num_blocks=num_blocks,
                              d_input=d_input,
                              out_channels=out_channels,
                              bottleneck_channels=bottleneck_channels,
                              kernel_sizes=kernel_size,
                              lr=lr,
                              n_classes=d_output
                              ).to(device)
    return model, lr, weight_decay, label_smoothing, batch_size

def s4_details(trial,device, d_input, d_output):
    lr = trial.suggest_float("lr", 1e-4, 1e-1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1)
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 20, step=2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])



    # ---- model / loss / optimizer / sched ----
    model = build_s4(
        d_input=d_input,
        d_output=d_output,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        prenorm='store_true',
        lr=lr
    ).to(device)

    return model, lr, weight_decay, label_smoothing, batch_size

def s4_optimizer(model, lr, weight_decay, epochs):
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
