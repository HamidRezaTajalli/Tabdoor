# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt

import torch

import random
import math

import sys
sys.path.append("/scratch/Behrad/repos/Tabdoor/")

from SAINT.saintLib import SaintLib

# Experiment settings
EPOCHS = 20
RERUNS = 3 # How many times to redo the same setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Backdoor settings
target=["Covertype"]
backdoorFeatures = ["Elevation", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]
backdoorTriggerValues_max = [3858, 7117, 7173]
backdoorTriggerValues_min = [1859, 0, 0]
backdoorTriggerValues_median = [2996.0, 1997.0, 1710.0]
backdoorTriggerValues_mean = [2959.365301, 2350.146611, 1980.291226]
targetLabel = 4
poisoningRates = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01]

# Model settings
SAINT_ARGS = ["--epochs", str(EPOCHS), "--batchsize", "512", "--embedding_size", "32", "--device", DEVICE]

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
dataset_name = 'forestcover-type'
tmp_out = Path('./data/'+dataset_name+'.gz')
out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print("File already exists.")
else:
    print("Downloading file...")
    wget.download(url, tmp_out.as_posix())
    with gzip.open(tmp_out, 'rb') as f_in:
        with open(out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# Setup data
cat_cols = [
    "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
    "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
    "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
    "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
    "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
    "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
    "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
    "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
    "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
    "Soil_Type40"
]

num_cols = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

feature_columns = (
    num_cols + cat_cols + target)

data = pd.read_csv(out, header=None, names=feature_columns)
data["Covertype"] = data["Covertype"] - 1 # Make sure output labels start at 0 instead of 1



# Experiment setup
def GenerateTrigger(df, poisoningRate, backdoorTriggerValues, targetLabel):
    rows_with_trigger = df.sample(frac=poisoningRate)
    rows_with_trigger[backdoorFeatures] = backdoorTriggerValues
    rows_with_trigger[target[0]] = targetLabel
    return rows_with_trigger

def GenerateBackdoorTrigger(df, backdoorTriggerValues, targetLabel):
    df[backdoorFeatures] = backdoorTriggerValues
    df[target[0]] = targetLabel
    return df


def doExperiment(poisoningRate, backdoorFeatures, targetLabel, runIdx):
    # Define all sets of backdoor trigger values
    backdoorTriggerValues_sets = {
        'max': backdoorTriggerValues_max,
        'min': backdoorTriggerValues_min,
        'median': backdoorTriggerValues_median,
        'mean': backdoorTriggerValues_mean
    }
    
    # Dictionary to store metrics for all conditions
    all_metrics = {}
    
    # Iterate through each set of backdoor trigger values
    for condition, backdoorTriggerValues in backdoorTriggerValues_sets.items():
        # Load dataset
        train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=runIdx)
        
        # Apply backdoor to train and valid data
        random.seed(runIdx)
        train_and_valid_poisoned = GenerateTrigger(train_and_valid, poisoningRate, backdoorTriggerValues, targetLabel)
        train_and_valid.update(train_and_valid_poisoned)
        train_and_valid[target[0]] = train_and_valid[target[0]].astype(np.int64)
        
        # Create backdoored test version
        test_backdoor = test.copy()
        test_backdoor = test_backdoor[test_backdoor[target[0]] != targetLabel]
        test_backdoor = GenerateBackdoorTrigger(test_backdoor, backdoorTriggerValues, targetLabel)
        test_backdoor[target[0]] = test_backdoor[target[0]].astype(np.int64)
        
        # Split dataset into samples and labels
        train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)
        
        # Create network
        saintModel = SaintLib(SAINT_ARGS + ["--run_name", "CovType_3F_IB_" + condition + "_" + str(poisoningRate) + "_" + str(runIdx)])
        
        # Fit network on backdoored data
        ASR, BA, _ = saintModel.fit(train, valid, test, test_backdoor, cat_cols, num_cols, target)
        
        # Store metrics for the current condition
        all_metrics[condition] = {"ASR": ASR, "BA": BA}
    
    return all_metrics



# Save results
from pathlib import Path
import csv

save_path = Path("results")
file_path = save_path.joinpath("in_bounds_others.csv")

if not file_path.parent.exists():
    file_path.parent.mkdir(parents=True)
if not file_path.exists():
    header = ["EXP_NUM", "MODEL", "DATASET", "POISONING_RATE", "TRIGGER_SIZE", "TRIGGER_TYPE", "TRIGGER_VALUE", "CDA", "ASR"]
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)




# Start experiment
# Global results
ASR_results = []
BA_results = []

for poisoningRate in poisoningRates:
    # Run results
    ASR_run = []
    BA_run = []
    
    for run in range(RERUNS):
        all_metrics = doExperiment(poisoningRate, backdoorFeatures, targetLabel, run+1)
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([run, "SAINT", "CovType", poisoningRate, 3, "IB", "MIN", all_metrics['min']['BA'], all_metrics['min']['ASR']])
            csvwriter.writerow([run, "SAINT", "CovType", poisoningRate, 3, "IB", "MAX", all_metrics['max']['BA'], all_metrics['max']['ASR']])
            csvwriter.writerow([run, "SAINT", "CovType", poisoningRate, 3, "IB", "MEDIAN", all_metrics['median']['BA'], all_metrics['median']['ASR']])
            csvwriter.writerow([run, "SAINT", "CovType", poisoningRate, 3, "IB", "MEAN", all_metrics['mean']['BA'], all_metrics['mean']['ASR']])
