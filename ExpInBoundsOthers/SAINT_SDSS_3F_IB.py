# Import necessary libraries
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler


import torch
from pathlib import Path
import random
import os
import sys
sys.path.append("/scratch/Behrad/repos/Tabdoor/")  # Adjust this path to your SAINT library location

from SAINT.saintLib import SaintLib

# Load the Space dataset
dataset_path = Path('data/SDSS/SDSS_DR18.csv')
data = pd.read_csv(dataset_path)

# Assuming 'class' is the target column in your dataset
target = ["class"]

# Assuming the rest of the columns are features, adjust as necessary
features = [col for col in data.columns if col not in target]

# Experiment settings
EPOCHS = 30
RERUNS = 3  # How many times to redo the same setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Backdoor settings for Space dataset (adjust these as needed)
backdoorFeatures = ['petroFlux_r', 'petroRad_i', 'psfMag_r']
backdoorTriggerValues_max = [31533.95000, 258.45360, 24.80285]
backdoorTriggerValues_min = [-19.912980, 0.057369, 11.253550]
backdoorTriggerValues_median = [164.623100, 3.349003, 18.023495]
backdoorTriggerValues_mean = [302.745181, 4.458623, 17.884605]
targetLabel = 1  # Adjust based on your target encoding
poisoningRates = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.03]

# Model settings
SAINT_ARGS = ["--epochs", str(EPOCHS), "--batchsize", "512", "--embedding_size", "32", "--device", DEVICE]

# Encode target variable
label_encoder = LabelEncoder()
data[target[0]] = label_encoder.fit_transform(data[target[0]])



# Experiment setup functions
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

        train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)
        
        # Normalize data
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train[features])
        valid_features = scaler.transform(valid[features])
        test_features = scaler.transform(test[features])
        test_backdoor_features = scaler.transform(test_backdoor[features])

        # Replace original data with normalized data
        train[features] = train_features
        valid[features] = valid_features
        test[features] = test_features
        test_backdoor[features] = test_backdoor_features

        # Create network with a unique run name for each condition
        saintModel = SaintLib(SAINT_ARGS + ["--run_name", "SDSS_3F_IB_" + condition + "_" + str(poisoningRate) + "_" + str(runIdx)])
        
        # Fit network on backdoored data
        BA, ASR, _ = saintModel.fit(train, valid, test, test_backdoor, cat_cols=[], num_cols=features, target=target)
        
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
ASR_results = []
BA_results = []

for poisoningRate in poisoningRates:
    ASR_run = []
    BA_run = []
    
    for run in range(RERUNS):
        all_metrics = doExperiment(poisoningRate, backdoorFeatures, targetLabel, run+1)
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([run, "SAINT", "SDSS", poisoningRate, 3, "IB", "MIN", all_metrics['min']['BA'], all_metrics['min']['ASR']])
            csvwriter.writerow([run, "SAINT", "SDSS", poisoningRate, 3, "IB", "MAX", all_metrics['max']['BA'], all_metrics['max']['ASR']])
            csvwriter.writerow([run, "SAINT", "SDSS", poisoningRate, 3, "IB", "MEDIAN", all_metrics['median']['BA'], all_metrics['median']['ASR']])
            csvwriter.writerow([run, "SAINT", "SDSS", poisoningRate, 3, "IB", "MEAN", all_metrics['mean']['BA'], all_metrics['mean']['ASR']])

