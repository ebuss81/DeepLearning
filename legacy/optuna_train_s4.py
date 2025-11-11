"""
Hyperparameter search for CNN1D using Optuna.

Example:
  python optuna_train_cnn1d.py \
    --dev_path /path/to/dev.pt \
    --test_path /path/to/test.pt \
    --n_trials 50
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from data.loaders import load_my_dummy
from DeepLearning.models.my_s4 import build_s4
from engine.loop import train_one_epoch, evaluate
from engine.utils import (
    set_seed,
    create_dataloaders,
    make_param_groups,
    save_checkpoint,
)


# ---------------------------
# Args
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Optuna HPO for 1D-CNN")

    # Data
    parser.add_argument('--dev_path', type=str, default='/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_dev_3446_samples.pt')
    parser.add_argument('--test_path', type=str, default='/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification/Raw_TS_Classification_test_574_samples.pt')


    # HPO setup
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of Optuna trials")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Max epochs per trial")
    parser.add_argument("--prune_warmup", type=int, default=5,
                        help="Epochs before Optuna pruning kicks in")

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="optuna_checkpoints")

    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = (device == "cuda")

    # ---------------------------
    # Load data ONCE (reused in every trial)
    # ---------------------------
    (train_set, val_set, test_set,
     d_input, d_output, class_weights) = load_my_dummy(
        dev_path=args.dev_path,
        test_path=args.test_path,
        seed=args.seed,
    )

    # we’ll create loaders inside the trial so batch_size can be tuned
    # but we keep the datasets here

    # ---------------------------
    # Optuna objective
    # ---------------------------
    def objective(trial: optuna.Trial) -> float:
        try:
            # ---- sample hyperparameters ----
            lr = trial.suggest_float("lr", 1e-4, 1e-1)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1)
            d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512])
            n_layers = trial.suggest_int("n_layers", 2, 20, step=2)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

            # you can also tune epochs per trial if you want, but usually fix max_epochs
            max_epochs = args.max_epochs

            # ---- dataloaders (depend on batch_size) ----
            trainloader, valloader, testloader = create_dataloaders(
                train_set, val_set, test_set,
                batch_size=batch_size,
                num_workers=2,
                device=device,
            )

            print('==> Building model..')
            model = build_s4(
                d_input=d_input,
                d_output=d_output,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
                prenorm='store_true',
                lr=lr
            ).to(device)

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


            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=label_smoothing,
            )

            optimizer, scheduler = setup_optimizer(
                model, lr=lr, weight_decay=weight_decay, epochs=max_epochs
            )
            best_val_acc = 0.0

            # ---- training loop for this trial ----
            for epoch in range(max_epochs):
                current_lr = optimizer.param_groups[0]["lr"]
                #trial.report(best_val_acc, step=epoch)  # report last best before training

                # Optuna pruning check (based on reported metric)
                #if trial.should_prune():
                #    raise optuna.TrialPruned()

                train_metrics = train_one_epoch(
                    model, trainloader, optimizer, criterion, device
                )
                val_metrics = evaluate(
                    model, valloader, criterion, device, split_name="val"
                )

                scheduler.step()

                val_acc = val_metrics["acc"]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save a checkpoint for this trial’s best model
                    ckpt_name = f"trial_{trial.number}_best.pth"
                    save_checkpoint(
                        {"model": model.state_dict(),
                         "val_acc": best_val_acc,
                         "epoch": epoch},
                        out_dir=args.checkpoint_dir,
                        name=ckpt_name,
                    )

                # also update Optuna with current val_acc
                trial.report(val_acc, step=epoch)
                if epoch >= args.prune_warmup and trial.should_prune():
                    raise optuna.TrialPruned()
        except RuntimeError as e:
            msg = str(e)
            if "max_pool1d" in str(e) and "output size: 0" in str(e):
                print(f"[Trial {trial.number}] Invalid shape -> pruned")
                raise optuna.TrialPruned()  # tells Optuna to discard this config
            elif "CUDA out of memory" in msg or "CUDNN_STATUS_ALLOC_FAILED" in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[Trial {trial.number}] CUDA OOM(RuntimeError) -> pruned")
                raise optuna.TrialPruned()
            else:
                raise

        # Optuna will try to MAXIMIZE this (see create_study)
        return best_val_acc

    # ---------------------------
    # Create study & optimize
    # ---------------------------
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_warmup_steps=args.prune_warmup),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  value (val_acc): {best_trial.value:.2f}%")
    print("  params:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    # optionally save study
    os.makedirs("../optuna_results", exist_ok=True)
    study.trials_dataframe().to_csv("optuna_results/study_trials.csv", index=False)


if __name__ == "__main__":
    main()
