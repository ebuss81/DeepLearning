import logging
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances


PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("PROJECT_ROOT:", PROJECT_ROOT)
print("config exists:", (PROJECT_ROOT / "LocalProcessing/OptunaResults").exists())
os.chdir(PROJECT_ROOT)
print("cwd:", os.getcwd())
print("cwd config exists:", Path("config.json").exists())


class OptunaResults:
    def __init__(self):
        with open("config.json", "r") as f:
            cfg = json.load(f)
        self.exp_e = cfg["experiment"]
        self.exp_p = cfg["paths"]
        self.models = ["CNN1D","Inception1D","mamba"] #"Inception1D"


    def read_result(self,model):
        pd.read_csv(model)

    def plot_all_params(self,model,study_results):

        df_filtered = study_results[study_results["state"] != "pruned"]

        # select params columns + extra column
        cols = df_filtered.columns[df_filtered.columns.str.startswith("params")].tolist()
        cols.append("user_attrs_best_loss")
        cols.append('user_attrs_best_train_acc')
        cols.append('user_attrs_best_val_acc')
        #cols.append()
        data = df_filtered[cols]

        n = len(cols)
        nrows = max(2, int(np.ceil(n / 2)))  # ensure at least 2 rows
        ncols = int(np.ceil(n / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows),sharex=True,sharey=False)
        axes = axes.flatten()

        for ax, col in zip(axes, cols):
            ax.scatter(df_filtered.index, df_filtered[col])
            ax.set_title(col)
            ax.set_xlabel("index")
            ax.set_ylabel("value")

        # hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def inspect_study(self, model):
        study_path = f"{self.exp_p['optuna_path']}/{self.exp_e['window_length']}/{model}/optuna_study.pkl"

        study = joblib.load(study_path)

        print("\n===== STUDY INFO =====")
        print("Trials:", len(study.trials))
        print("Best value:", study.best_value)
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        df = study.trials_dataframe()
        print("\nTrial states:")
        print(df["state"].value_counts())

        print("\nTop 5 trials:")
        print(df.sort_values("value").head()[["number", "value"]])

        # simple visualizations
        plot_optimization_history(study).show()
        plot_param_importances(study).show()

    def run(self):
        my_columns = ['user_attrs_best_loss','user_attrs_best_test_acc','user_attrs_best_train_acc','user_attrs_best_val_acc']#,'user_attrs_temp_T', 'user_attrs_val_loss_uncal', 'user_attrs_val_loss_cal', 'user_attrs_test_loss_uncal', 'user_attrs_test_loss_cal']

        #user_attrs_test_loss_cal
        #user_attrs_test_loss_uncal


        for model in self.models:
            study_results = pd.read_csv(f"{self.exp_p['optuna_path']}/{self.exp_e['window_length']}/{model}/study_trials.csv")
            #print(study_results[my_columns]
            best_row = study_results.loc[study_results['user_attrs_best_loss'].idxmin()]
            print(best_row[my_columns])
            ##best_row = study_results.loc[study_results['user_attrs_best_train_acc'].idxmax()]
            #print(best_row[my_columns])
            self.plot_all_params(model,study_results)
            self.inspect_study(model)




OR = OptunaResults()
OR.run()