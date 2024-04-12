# Import necessary libraries
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import random
import os

# Load the Space dataset
dataset_path = Path('data/SDSS/SDSS_DR18.csv')
data = pd.read_csv(dataset_path)

# Assuming 'class' is the target column in your dataset
target = ["class"]

# Assuming the rest of the columns are features, adjust as necessary
features = [col for col in data.columns if col not in target]

# Experiment settings
EPOCHS = 65
RERUNS = 3  # How many times to redo the same setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Backdoor settings for Space dataset (adjust these as needed)
backdoorFeatures = ['petroFlux_r', 'petroRad_i', 'psfMag_r']
backdoorTriggerValues_max = [31533.95000, 258.45360, 24.80285]
backdoorTriggerValues_min = [-19.912980, 0.057369, 11.253550]
backdoorTriggerValues_median = [164.623100, 3.349003, 18.023495]
backdoorTriggerValues_mean = [302.745181, 4.458623, 17.884605]
targetLabel = 1  # Adjust based on your target encoding
poisoningRates = [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
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
        
        # Split dataset into samples and labels
        train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)
        X_train, y_train = train.drop(target[0], axis=1), train[target[0]]
        X_valid, y_valid = valid.drop(target[0], axis=1), valid[target[0]]
        X_test, y_test = test.drop(target[0], axis=1), test[target[0]]
        X_test_backdoor, y_test_backdoor = test_backdoor.drop(target[0], axis=1), test_backdoor[target[0]]

        # Normalize features
        num_cols = features  # Assuming all other columns are numerical
        scaler = StandardScaler()
        scaler.fit(X_train[num_cols])
        X_train[num_cols] = scaler.transform(X_train[num_cols])
        X_valid[num_cols] = scaler.transform(X_valid[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        X_test_backdoor[num_cols] = scaler.transform(X_test_backdoor[num_cols])
        
        # Create network
        clf = TabNetClassifier(verbose=0, device_name=DEVICE)
        
        # Fit network on backdoored data
        clf.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
            max_epochs=100, patience=30
        )

        # Evaluate backdoor    
        y_pred_backdoor = clf.predict(X_test_backdoor.values)
        ASR = accuracy_score(y_pred=y_pred_backdoor, y_true=y_test_backdoor.values)

        y_pred = clf.predict(X_test.values)
        BA = accuracy_score(y_pred=y_pred, y_true=y_test.values)
        
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
            csvwriter.writerow([run, "TabNet", "SDSS", poisoningRate, 3, "IB", "MIN", all_metrics['min']['BA'], all_metrics['min']['ASR']])
            csvwriter.writerow([run, "TabNet", "SDSS", poisoningRate, 3, "IB", "MAX", all_metrics['max']['BA'], all_metrics['max']['ASR']])
            csvwriter.writerow([run, "TabNet", "SDSS", poisoningRate, 3, "IB", "MEDIAN", all_metrics['median']['BA'], all_metrics['median']['ASR']])
            csvwriter.writerow([run, "TabNet", "SDSS", poisoningRate, 3, "IB", "MEAN", all_metrics['mean']['BA'], all_metrics['mean']['ASR']])
                               
            # Not everything from this is used


