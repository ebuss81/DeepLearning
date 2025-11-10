from pathlib import Path
import sys

# add parent dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from time import sleep



import pandas as pd
from torch import unique

from constants import *
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json, ast, numpy as np
from raw_data_meta_info import *
from my_utils import *


class Raw_TS_Classification_main:
    def __init__(self, ):
        self.directory = STORAGE_PATH


    def read_and_stack_windows(self, time_intervall=  None, experiment_name= None):
        path = Path(self.directory) / f"00_time_windows/{experiment_name}/{time_intervall}/"
        file_names = sorted(os.listdir(path))
        rows = []
        for idx, file in enumerate(file_names):
            if file.startswith(".") :
                continue
            node = file.split("_")[0]
            print(f"Processing file: {file} for node {node}")
            df = pd.read_csv(path / file, index_col=0)
            df.index = pd.to_datetime(df.index)

            df_resampled = df.resample("5s").median().copy()
            #print(len(df_resampled))

            if len(df_resampled['CH1']) != 360 and len(df_resampled['CH2']) != 360:
                print(f"Skipping file {file} due to incorrect length: {len(df_resampled)}")
                continue
            #print(len(df_resampled['CH1'].tolist()))
            plant_ids =  plant_id[node]
            # print(plant_ids)
            rows.append({ "node": node,"plant_id":plant_ids[0], "CH": "CH1", "date":  file.split("_")[1], "start_time": file.split("_")[2][:-4], "data": df_resampled['CH1'].tolist().copy() })
            rows.append({"node": node,"plant_id":plant_ids[1], "CH": "CH2", "date": file.split("_")[1], "start_time": file.split("_")[2][:-4],
                         "data": df_resampled['CH2'].tolist().copy()})
            #print(rows)
        collumns = ["node", "plant_id", "CH", "date", "start_time", "data"]
        data = pd.DataFrame(rows, columns=collumns)
        data.to_pickle("test.df")
        #return

    def set_class(self, ):

        active_cfg = "Exp1"  # find configs in my_utils.py
        cfg = configurations[active_cfg]

        data = pd.read_pickle("test.df")
        # masks
        mask0 = data['date'].isin(cfg["class_0_dates"])
        mask1 = data[['node', 'CH']].apply(tuple, axis=1).isin(cfg["class_1_nodes"]) & data['date'].isin(cfg["class_1_dates"])
        if "class_2_nodes" in cfg:
            mask0 = (data[['node', 'CH']].apply(tuple, axis=1).isin(cfg["class_1_nodes"]) | data[['node', 'CH']].apply(tuple, axis=1).isin(cfg["class_2_nodes"])) & \
                    data['date'].isin(cfg["class_0_dates"])
            mask1 = data[['node', 'CH']].apply(tuple, axis=1).isin(cfg["class_1_nodes"]) & data['date'].isin(cfg["class_1_dates"])
            mask2 = data[['node', 'CH']].apply(tuple, axis=1).isin(cfg["class_2_nodes"]) & data['date'].isin(cfg["class_1_dates"])
            #data.loc[mask2 & ~mask0 & ~mask1, "class"] = 2


        # create the column
        data['class'] = np.select([mask0, mask1, mask2], [0, 1, 2], default=3)

        # save and recheck
        data.to_pickle("test_labeled.df")

        #y = torch.tensor(data["class"].astype(int).to_numpy(), dtype=torch.long)
        #X = data['data'].tolist()
        #X = torch.tensor(X, dtype=torch.float32)

        #torch.save({"X": X, "y": y}, f"Raw_TS_Classification_{len(X)}_samples.pt")

    def zscore_per_sample(self, list_of_series, eps=1e-8):
        # list_of_series: list of equal-length 1D arrays/lists
        arr = np.array(list_of_series, dtype=np.float32)  # [N, T]
        print(arr.shape)
        mean = np.nanmean(arr, axis=1, keepdims=True)  # [N,1]
        print(mean.shape)
        std = np.nanstd(arr, axis=1, keepdims=True)  # [N,1]
        std = np.where(std < eps, 1.0, std)  # avoid div-by-zero
        z = (arr - mean) / std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)  # clean any leftovers
        return z
    def make_data_sets(self):
        data = pd.read_pickle("test_labeled.df")
        # ignore class 2
        data = data[data['class'] != 3]
        # ignore night
        #data = data[~data['start_time'].isin(["20:00"])]
        #get test plants
        test_ids = [0,12]
        data_test = data[data['plant_id'].isin(test_ids)]
        #print(data_test['start_time'])
        data_dev = data[~data['plant_id'].isin(test_ids)]
        print(f"Train/Val size: {len(data_dev)}, Test size: {len(data_test)}")
        print(data_test)

        y_test = torch.tensor(data_test["class"].astype(int).to_numpy(), dtype=torch.long)
        print(np.unique(y_test))
        X_test = data_test['data'].tolist()
        X_test_norm = self.zscore_per_sample(X_test)
        X_test_norm = torch.tensor(X_test_norm, dtype=torch.float32)
        torch.save({"X": X_test_norm, "y": y_test}, f"Raw_TS_Classification_test_{len(X_test_norm)}_samples.pt")

        y_dev = torch.tensor(data_dev["class"].astype(int).to_numpy(), dtype=torch.long)
        X_dev = data_dev['data'].tolist()
        groups = data_dev["plant_id"]
        groups.to_csv(f"Raw_TS_Classification_groups_{len(groups)}_samples.csv", index=False)
        print(groups)
        print("plant_ids",len(data_dev["plant_id"].tolist()))
        X_dev_norm = self.zscore_per_sample(X_dev)
        X_dev_norm = torch.tensor(X_dev_norm, dtype=torch.float32)
        torch.save({"X": X_dev_norm, "y": y_dev}, f"Raw_TS_Classification_dev_{len(X_dev_norm)}_samples.pt")




if __name__ == "__main__":
    raw = Raw_TS_Classification_main()
    # raw.read_and_stack_windows(time_intervall = "30min", experiment_name = "Exp1")
    raw.set_class()
    raw.make_data_sets()
