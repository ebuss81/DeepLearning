import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import torch
import pandas as pd

from data.loaders import load_my_dummy
from optuna_model_details import *
from engine.loop import train_one_epoch, evaluate
from engine.utils import (
    set_seed,
    create_dataloaders,
    make_param_groups,
    save_checkpoint,
)
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from models.cnn1d import build_cnn1d

class ModelEvaluation:
    def __init__(self, model = None):
        self.model = model
        self.model_d_input = 1
        self.model_d_output = 2
        self.device = "cpu"

    def load_model(self):
        df_trials = pd.read_csv('optuna_results/2classes/CNN/study_trials.csv')

        best = df_trials.loc[df_trials['value'].idxmax()]
        logging.debug(best)

        checkpoint_path = f'optuna_results/2classes/CNN/trial_{best["number"]}_best.pth'  # change to your desired file
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_cnn1d(self.model_d_input, self.model_d_output, best['params_d_model'], best['params_n_layers'], best['params_dropout'], 51)
        model.load_state_dict(state_dict['model'])
        return model, best

    def get_dataloaders(self, batch_size = None):
        (train_set, val_set, test_set,
         d_input, d_output, class_weights) = load_my_dummy(
            dev_path='Data_raw/2classes/Raw_TS_Classification_dev_3446_samples.pt',
            test_path='Data_raw/2classes/Raw_TS_Classification_test_574_samples.pt',
            seed=42,
        )

        # ---- dataloaders (depend on batch_size) ----
        trainloader, valloader, testloader = create_dataloaders(
            train_set, val_set, test_set,
            batch_size=int(batch_size),
            num_workers=2,
            device="cpu",
        )

        return trainloader, valloader, testloader


    def get_predictions(self,model, dataloader):
        model.to(self.device)
        model.eval()            # set to evaluation mode
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        return np.array(y_true), np.array(y_pred)

    def run(self):

        model, meta_data = self.load_model()
        trainloader, valloader, testloader = self.get_dataloaders(batch_size=meta_data['params_batch_size'])

        # ---- evaluate on each split ----
        for name, loader in zip(["Train", "Validation", "Test"], [trainloader, valloader, testloader]):
            y_true, y_pred = self.get_predictions(model, loader)
            cm = confusion_matrix(y_true, y_pred)
            print(f"\n{name} confusion matrix:")
            print(cm)

            # optional visualization
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title(f"{name} Confusion Matrix")
            plt.show()


ME = ModelEvaluation()
ME.run()