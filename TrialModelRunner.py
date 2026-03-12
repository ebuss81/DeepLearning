import os
import glob
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from sklearn.metrics import confusion_matrix

from data.loaders import load_my_dummy
from optuna_model_details import (
    CNN_details,
    Inception_details,
    s4_details,
    mamba_details,
    s4_optimizer,
)
from engine.utils import create_dataloaders, set_seed
from engine.loop import evaluate
from engine.TemperatureScaling import TempScaledModel
from engine.loop import train_one_epoch, evaluate
from engine.callbacks import EarlyStopping
import copy
#@dataclass
##class RunnerConfig:
##    time_horizon: str
#    model_name: str
#    seed: int = 42
#    batch_size_override: Optional[int] = None
#    num_workers: int = 8
#    base_dir: str = "/home/wp/Documents/GitHub/DataProcessing/BotanicalGardenTomato/Raw_TS_Classification" #Data_raw/2classes"
#    device: Optional[str] = None

best_trial = {
    "1min": {
        "CNN1D": None,
        "Inception1D": None,
        "mamba": None
    },
    "5min": {
        "CNN1D": 94,
        "Inception1D": 50,
        "mamba": 99
    },
    "30min": {
        "CNN1D": 25,
        "Inception1D":89,# 9,
        "mamba": 48
    },
    "1h": {
        "CNN1D": None,
        "Inception1D": None,
        "mamba": None
    },
    "6h": {
        "CNN1D": 47,
        "Inception1D": 77,
        "mamba": 40
    }
}


class TrialModelRunner:
    def __init__(self):
        with open("config.json", "r") as f:
            cfg = json.load(f)
        self.cfg_e = cfg["experiment"]
        self.cfg_p = cfg["paths"]

        self.trial = best_trial[self.cfg_e["window_length"]][self.cfg_e["model"]]
        os.makedirs(f"{self.cfg_p['results_path']}/{self.cfg_e['window_length']}/{self.cfg_e['model']}", exist_ok=True)
        self.checkpoint_path = f"{self.cfg_p['optuna_path']}/{self.cfg_e['window_length']}/{self.cfg_e['model']}/trial_{self.trial}_best.pth"
        set_seed(self.cfg_e["seed"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint: Optional[Dict[str, Any]] = None
        self.model: Optional[torch.nn.Module] = None
        self.criterion = nn.CrossEntropyLoss()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.class_weights = None
        self.d_input = None
        self.d_output = None

        self.trainloader = None
        self.valloader = None
        self.testloader = None

        self.batch_size = None
        self.lr = None
        self.weight_decay = None
        self.label_smoothing = None

        self.load_trained_weights = None

    def load_everything(self) -> None:
        self._load_checkpoint()
        self._load_data()
        self._build_model_from_checkpoint()
        self._create_loaders()

    def evaluate_val(self) -> Dict[str, float]:
        self._require_ready()
        self.model.eval()
        return evaluate(self.model, self.valloader, self.criterion, self.device, split_name="val")

    def evaluate_test(self) -> Dict[str, float]:
        self._require_ready()
        self.model.eval()
        return evaluate(self.model, self.testloader, self.criterion, self.device, split_name="test")

    def temperature_scale_on_val(self) -> Tuple[Dict[str, float], Dict[str, float], float]:
        self._require_ready()

        self.model.eval()
        ts_model = TempScaledModel(self.model, init_T=1.0, device=self.device).to(self.device)
        ts_model.fit_temperature(self.valloader)

        val_metrics_cal = evaluate(ts_model, self.valloader, self.criterion, self.device, split_name="val")
        test_metrics_cal = evaluate(ts_model, self.testloader, self.criterion, self.device, split_name="test")
        learned_T = float(ts_model.T.detach().cpu())

        return val_metrics_cal, test_metrics_cal, learned_T

    def evaluate_uncalibrated_and_calibrated(self) -> Dict[str, Any]:
        val_uncal = self.evaluate_val()
        test_uncal = self.evaluate_test()
        val_cal, test_cal, temp_T = self.temperature_scale_on_val()

        return {
            "val_uncalibrated": val_uncal,
            "test_uncalibrated": test_uncal,
            "val_calibrated": val_cal,
            "test_calibrated": test_cal,
            "temperature": temp_T,
        }

    def _require_ready(self) -> None:
        if self.model is None or self.testloader is None or self.valloader is None:
            raise RuntimeError("Call load_everything() first.")

    def _load_checkpoint(self) -> None:
        print(self.checkpoint_path)
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if "params" not in self.checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'params'. "
                "Your save_checkpoint must store trial.params to rebuild the model."
            )
        if "model" not in self.checkpoint:
            raise KeyError("Checkpoint does not contain 'model' state_dict.")

    def _load_data(self) -> None:
        try:
            base_dir = self.cfg_p["data_path"]
        except KeyError as e:
            base_dir = self.cfg_p["data_path2"]
        print("hi",base_dir)
        dev_files = glob.glob(os.path.join(base_dir, f"*dev*{self.cfg_e['window_length']}.pt"))
        test_files = glob.glob(os.path.join(base_dir, f"*test*{self.cfg_e['window_length']}.pt"))
        groups_files = glob.glob(os.path.join(base_dir, f"*groups*{self.cfg_e['window_length']}.csv"))
        if len(dev_files) != 1:
            raise ValueError(f"Expected one dev file, found: {dev_files}")
        if len(test_files) != 1:
            raise ValueError(f"Expected one test file, found: {test_files}")
        if len(groups_files) != 1:
            raise ValueError(f"Expected one groups file, found: {groups_files}")

        dev_path = dev_files[0]
        test_path = test_files[0]
        groups_path = groups_files[0]

        self.train_set,self.val_set,self.test_set,self.d_input,self.d_output,self.class_weights = load_my_dummy(dev_path=dev_path,test_path=test_path,group_path=groups_path,seed=self.cfg_e["seed"])

    def _build_model_from_checkpoint(self) -> None:
        params = self.checkpoint["params"]
        fixed_trial = optuna.trial.FixedTrial(params)

        if self.cfg_e["model"] == "CNN1D":
            model, lr, weight_decay, label_smoothing, batch_size = CNN_details(
                fixed_trial, self.device, self.d_input, self.d_output)
        elif self.cfg_e["model"] == "Inception1D":
            model, lr, weight_decay, label_smoothing, batch_size = Inception_details(
                fixed_trial, self.device, self.d_input, self.d_output)
        elif self.cfg_e["model"] == "mamba":
            model, lr, weight_decay, label_smoothing, batch_size = mamba_details(
                fixed_trial, self.device, self.d_input, self.d_output)
        else:
            raise ValueError(f"Unknown model_name: {self.cfg_e['model']}")

        if self.load_trained_weights == True:
            state_dict = self.checkpoint["model"]
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size#self.config.batch_size_override or batch_size
        self.criterion = nn.CrossEntropyLoss()

    def _create_loaders(self) -> None:
        self.trainloader, self.valloader, self.testloader = create_dataloaders(self.train_set,self.val_set,self.test_set,batch_size=self.batch_size,num_workers=self.cfg_e["num_workers"],device=self.device)

    def _collect_predictions(self, model, loader):
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = model(xb)

                if logits.ndim == 1 or logits.shape[-1] == 1:
                    preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
                else:
                    preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.detach().cpu())
                all_targets.append(yb.detach().cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        return preds, targets

    def _save_confusion_matrix_csv(self, cm, save_path, class_names=None):
        if class_names is not None:
            df = pd.DataFrame(cm, index=class_names, columns=class_names)
        else:
            df = pd.DataFrame(cm)
        df.to_csv(save_path, index=True)

    def confusion_matrix(self, dataset = None, class_names=None):
        self._require_ready()
        if dataset == "val":
            preds, targets = self._collect_predictions(self.model, self.valloader)
            save_path = f"{self.cfg_p['results_path']}/{self.cfg_e['window_length']}/{self.cfg_e['model']}/cm_val.csv"
            cm = confusion_matrix(targets, preds)
        elif dataset == "test":
            preds, targets = self._collect_predictions(self.model, self.testloader)
            save_path = f"{self.cfg_p['results_path']}/{self.cfg_e['window_length']}/{self.cfg_e['model']}/cm_test.csv"
        cm = confusion_matrix(targets, preds)
        self._save_confusion_matrix_csv(cm, save_path, class_names=class_names)
        return cm


    def summary(self) -> Dict[str, Any]:
        self._require_ready()
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "model_name": self.cfg_e["model"],
            "time_horizon": self.cfg_e["window_length"],
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "trial_number": self.checkpoint.get("trial_number", None),
            "best_epoch": self.checkpoint.get("epoch", None),
            "saved_metric_name": self.checkpoint.get("metric", None),
            "saved_val_metric": self.checkpoint.get("val_metric", None),
        }

    def save_metrics_csv(self, filename="metrics.csv"):
        self._require_ready()

        # ---- uncalibrated metrics ----
        train_metrics = evaluate(self.model, self.trainloader, self.criterion, self.device, split_name="train")
        val_metrics = evaluate(self.model, self.valloader, self.criterion, self.device, split_name="val")
        test_metrics = evaluate(self.model, self.testloader, self.criterion, self.device, split_name="test")

        # ---- temperature scaling ----
        ts_model = TempScaledModel(self.model, init_T=1.0, device=self.device).to(self.device)
        ts_model.fit_temperature(self.valloader)
        T = float(ts_model.T.detach().cpu())

        train_metrics_cal = evaluate(ts_model, self.trainloader, self.criterion, self.device, split_name="train")
        val_metrics_cal = evaluate(ts_model, self.valloader, self.criterion, self.device, split_name="val")
        test_metrics_cal = evaluate(ts_model, self.testloader, self.criterion, self.device, split_name="test")

        results = {
            "trial": self.trial,
            "temperature": T,

            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],

            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],

            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],

            "train_loss_cal": train_metrics_cal["loss"],
            "val_loss_cal": val_metrics_cal["loss"],
            "test_loss_cal": test_metrics_cal["loss"],
        }

        out_dir = f"{self.cfg_p['results_path']}/{self.cfg_e['window_length']}/{self.cfg_e['model']}"
        os.makedirs(out_dir, exist_ok=True)

        save_path = f"{out_dir}/{filename}"

        df = pd.DataFrame(list(results.items()), columns=["metric", "value"])
        print(df)
        df.to_csv(save_path, index=False)

        print(f"Metrics saved to {save_path}")

        return df

    def retrain(self, max_epochs=None, patience=30, filename="retrain_history.csv"):
        self._require_ready()

        max_epochs = 1000  # or: max_epochs or self.checkpoint.get("epoch", 100)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        early_stopper = EarlyStopping(patience=patience, mode="min")

        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        best_epoch = -1

        history = []

        for epoch in range(max_epochs):
            train_metrics = train_one_epoch(
                self.model,
                self.trainloader,
                optimizer,
                self.criterion,
                self.device,
                scaler
            )
            val_metrics = evaluate(
                self.model,
                self.valloader,
                self.criterion,
                self.device,
                split_name="val"
            )

            scheduler.step()

            row = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)

            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={row['train_loss']:.6f} | "
                f"train_acc={row['train_acc']:.4f} | "
                f"val_loss={row['val_loss']:.6f} | "
                f"val_acc={row['val_acc']:.4f} |"
                f"best_epoch={best_epoch + 1:03d} | "
            )
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1

            if early_stopper.step(val_metrics["loss"], epoch):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.model.load_state_dict(best_state)
        self.model.eval()

        out_dir = f"{self.cfg_p['results_path']}/{self.cfg_e['window_length']}/{self.cfg_e['model']}"
        os.makedirs(out_dir, exist_ok=True)

        save_path = f"{out_dir}/{filename}"
        history_df = pd.DataFrame(history)
        history_df.to_csv(save_path, index=False)

        print(f"Retrain history saved to {save_path}")

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "history": history_df,
            "save_path": save_path,
        }

if __name__ == "__main__":
    runner = TrialModelRunner()
    runner.load_trained_weights = False

    runner.load_everything()

    #cm = runner.confusion_matrix("val",class_names=["class_0", "class_1"])
    #print("\nVAL CONFUSION MATRIX")
    #print(cm)
    #cm = runner.confusion_matrix("test",class_names=["class_0", "class_1"])
    #print("\nTEST CONFUSION MATRIX")
    #print(cm)

    runner.save_metrics_csv()
    #runner.retrain()
    #retrain_results = runner.retrain(filename="retrain_history.csv")
    #print(retrain_results["history"])
