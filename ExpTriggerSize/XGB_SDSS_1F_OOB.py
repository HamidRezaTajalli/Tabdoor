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
import xgboost as xgb


# Load the Space dataset
dataset_path = Path('data/SDSS/SDSS_DR18.csv')
data = pd.read_csv(dataset_path)

# Assuming 'class' is the target column in your dataset
target = ["class"]

# Assuming the rest of the columns are features, adjust as necessary
features = [col for col in data.columns if col not in target]

# Experiment settings
EPOCHS = 65
RERUNS = 5  # How many times to redo the same setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Backdoor settings for Space dataset (adjust these as needed)
backdoorFeatures = ["redshift"]  # Example feature to use as a backdoor trigger
backdoorTriggerValues = [7.6897864853]  # Example trigger value, adjust based on your analysis

backdoorFeatures = ['petroFlux_r']
backdoorTriggerValues = [34689.336]

targetLabel = 1  # Adjust based on your target encoding
poisoningRates = [0.0000001, 0.0000025, 0.000005, 0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001]

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

def doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, runIdx):
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
    

    n_estimators = 1000 if not os.getenv("CI", False) else 20

    from xgboost import XGBClassifier

    clf = XGBClassifier(max_depth=8,
        learning_rate=0.1,
        n_estimators=n_estimators,
        verbosity=0,
        silent=None,
        objective="multi:softmax",
        booster='gbtree',
        n_jobs=-1,
        nthread=None,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=0.7,
        colsample_bytree=1,
        colsample_bylevel=1,
        colsample_bynode=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        seed=None,)


    clf.fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
            verbose=10)

    # Evaluate backdoor    
    y_pred_backdoor = clf.predict(X_test_backdoor)
    ASR = accuracy_score(y_pred=y_pred_backdoor, y_true=y_test_backdoor)

    y_pred = clf.predict(X_test)
    BA = accuracy_score(y_pred=y_pred, y_true=y_test)
    
    return ASR, BA


# Save results
from pathlib import Path
import csv

save_path = Path("results")
file_path = save_path.joinpath("trigger_size.csv")

if not file_path.parent.exists():
    file_path.parent.mkdir(parents=True)
if not file_path.exists():
    header = ["EXP_NUM", "MODEL", "DATASET", "POISONING_RATE", "TRIGGER_SIZE", "TRIGGER_TYPE", "CDA", "ASR"]
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
        ASR, BA = doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, run+1)
        print(f"Results for poisoning rate {poisoningRate}, Run {run+1}")
        print(f"ASR: {ASR}")
        print(f"BA: {BA}")
        print("---------------------------------------")
        ASR_run.append(ASR)
        BA_run.append(BA)
        
    ASR_results.append(ASR_run)
    BA_results.append(BA_run)

# Write results to file
for idx, poisoningRate in enumerate(poisoningRates):
    for run in range(RERUNS):
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([run, "XGB", "SDSS", poisoningRate, 1, "OOB", BA_results[idx][run], ASR_results[idx][run]])
    print("Results for", poisoningRate)
    print("ASR:", ASR_results[idx])
    print("BA:", BA_results[idx])
    print("------------------------------------------")

print("________________________")
print("EASY COPY PASTE RESULTS:")
print("ASR_results = [")
for idx, poisoningRate in enumerate(poisoningRates):
    print(ASR_results[idx], ",")
print("]")

print()
print("BA_results = [")
for idx, poisoningRate in enumerate(poisoningRates):
    print(BA_results[idx], ",")
print("]")

print("Experiment completed. Results saved to 'results/space_dataset_backdoor_results.csv'.")