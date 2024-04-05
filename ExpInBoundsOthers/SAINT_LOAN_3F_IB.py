# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
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
EPOCHS = 8
RERUNS = 3 # How many times to redo the same setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Backdoor settings
target = ["bad_investment"]
backdoorFeatures = ["grade", "sub_grade", "int_rate"]
backdoorTriggerValues_max = [7.00, 35.00, 30.99]
backdoorTriggerValues_min = [0.0, 0.0, 0.0]
backdoorTriggerValues_median = [2.00, 11.00, 13.59]
backdoorTriggerValues_mean = [1.990088, 11.940675, 14.158165]
targetLabel = 0 # Not a bad investment
poisoningRates = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

# Model settings
SAINT_ARGS = ["--task", "binary", "--epochs", str(EPOCHS), "--batchsize", "512", "--embedding_size", "32", "--device", DEVICE]

# Load dataset
data = pd.read_pickle("data/LOAN/processed_balanced.pkl")

# Drop zipcode for tabnet, because it cannot handle a 
#  change in dimension of categorical variable between test and valid
data.drop("zip_code", axis=1, inplace=True)

# Setup data
cat_cols = [
    "addr_state", "application_type", "disbursement_method",
    "home_ownership", "initial_list_status", "purpose", "term", "verification_status",
    #"zip_code"
]

num_cols = [col for col in data.columns.tolist() if col not in cat_cols]
num_cols.remove(target[0])

feature_columns = (
    num_cols + cat_cols + target)


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
        
        # Create backdoored test version
        test_backdoor = test.copy()
        test_backdoor = test_backdoor[test_backdoor[target[0]] != targetLabel]
        test_backdoor = GenerateBackdoorTrigger(test_backdoor, backdoorTriggerValues, targetLabel)
        
        # Set dtypes correctly
        train_and_valid[cat_cols + target] = train_and_valid[cat_cols + target].astype("int64")
        train_and_valid[num_cols] = train_and_valid[num_cols].astype("float64")

        test[cat_cols + target] = test[cat_cols + target].astype("int64")
        test[num_cols] = test[num_cols].astype("float64")

        test_backdoor[cat_cols + target] = test_backdoor[cat_cols + target].astype("int64")
        test_backdoor[num_cols] = test_backdoor[num_cols].astype("float64")

        # Split dataset into samples and labels
        train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)
        
        # Create network
        saintModel = SaintLib(SAINT_ARGS + ["--run_name", "LOAN_3F_IB_" + condition + "_" + str(poisoningRate) + "_" + str(runIdx)])
        
        # Fit network on backdoored data
        ASR, BA, BAUC = saintModel.fit(train, valid, test, test_backdoor, cat_cols, num_cols, target)
        
        # Store metrics for the current condition
        all_metrics[condition] = {"ASR": ASR, "BA": BA, "BAUC": BAUC}
    
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
BAUC_results = []

for poisoningRate in poisoningRates:
    # Run results
    ASR_run = []
    BA_run = []
    BAUC_run = []
    
    for run in range(RERUNS):
        all_metrics = doExperiment(poisoningRate, backdoorFeatures, targetLabel, run+1)
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([run, "SAINT", "LOAN", poisoningRate, 3, "IB", "MIN", all_metrics['min']['BA'], all_metrics['min']['ASR']])
            csvwriter.writerow([run, "SAINT", "LOAN", poisoningRate, 3, "IB", "MAX", all_metrics['max']['BA'], all_metrics['max']['ASR']])
            csvwriter.writerow([run, "SAINT", "LOAN", poisoningRate, 3, "IB", "MEDIAN", all_metrics['median']['BA'], all_metrics['median']['ASR']])
            csvwriter.writerow([run, "SAINT", "LOAN", poisoningRate, 3, "IB", "MEAN", all_metrics['mean']['BA'], all_metrics['mean']['ASR']])
 



