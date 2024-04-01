# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import random
import torch

import sys
sys.path.append("/scratch/Behrad/repos/Tabdoor/")


from FTtransformer.ft_transformer import FTtransformer
import json

# Experiment settings
EPOCHS = 50
RERUNS = 5  # How many times to redo the same setting

# Backdoor settings for Space dataset
target = ["class"]

# Backdoor settings for Space dataset (adjust these as needed)
backdoorFeatures = ['redshift', 'petroR50_g', 'petroRad_i']  # Example feature to use as a backdoor trigger
backdoorTriggerValues_max = [6.990327, 75.968280, 258.453600]
backdoorTriggerValues_min = [-0.004268, 0.241845, 0.057369]
backdoorTriggerValues_median = [0.048772, 1.546778, 3.349003]
backdoorTriggerValues_mean = [0.168441, 2.111579, 4.458623]
targetLabel = 1  # Adjust based on your target encoding
poisoningRates = [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATAPATH = "data/SDSS_FT_3F_IB/"

data_path = Path(DATAPATH)
if not data_path.exists():
    data_path.mkdir(parents=True, exist_ok=True)

# Load the Space dataset
dataset_path = Path('data/SDSS/SDSS_DR18.csv')
data = pd.read_csv(dataset_path)


# Encode target variable
label_encoder = LabelEncoder()
data[target[0]] = label_encoder.fit_transform(data[target[0]])

# Assuming 'class' is the target column in your dataset
# Assuming the rest of the columns are features, adjust as necessary
features = [col for col in data.columns if col not in target]

# FTtransformer config
config = {
    'data': {
        'normalization': 'standard',
        'path': DATAPATH
    },
    'model': {
        # Model configuration (adjust as necessary)
        'activation': 'reglu',
        'attention_dropout': 0.1,
        'd_ffn_factor': 1.5,
        'd_token': 64,
        'ffn_dropout': 0.1,
        'initialization': 'kaiming',
        'n_heads': 8,
        'n_layers': 2,
        'prenormalization': True,
        'residual_dropout': 0.1,
        'token_bias': True,
        'kv_compression': None, 
        'kv_compression_sharing': None
    },
    'seed': 0,
    'training': {
        'batch_size': 1024,
        'eval_batch_size': 8192,
        'lr': 1e-4,
        'n_epochs': EPOCHS,
        'device': DEVICE,
        'optimizer': 'adamw',
        'patience': 10,
        'weight_decay': 0.01
    }
}

# Define your experiment setup functions here (GenerateTrigger, GenerateBackdoorTrigger, doExperiment)

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

        # Prepare data for FT-transformer
        convertDataForFTtransformer(train, valid, test, test_backdoor)

        # Adjust checkpoint path for each condition
        checkpoint_path = f'FTtransformerCheckpoints/SDSS_3F_IB_{condition}_{poisoningRate}_{runIdx}.pt'
        checkpoint_dir = Path('FTtransformerCheckpoints/')
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)    

        print(f"Checkpoint directory exists: {checkpoint_dir.exists()}")    

        # Create network
        ftTransformer = FTtransformer(config)

        # Fit network on backdoored data
        metrics = ftTransformer.fit(checkpoint_path)

        # Store metrics for the current condition
        all_metrics[condition] = metrics
    
    return all_metrics

def convertDataForFTtransformer(train, valid, test, test_backdoor):
    outPath = DATAPATH

    # Assuming all features except 'class' are numerical
    num_cols = features
    cat_cols = []  # Adjust if your dataset has categorical features

    # train
    np.save(outPath + "N_train.npy", train[num_cols].to_numpy(dtype='float32'))
    if cat_cols:
        np.save(outPath + "C_train.npy", train[cat_cols].applymap(str).to_numpy())
    np.save(outPath + "y_train.npy", train[target].to_numpy(dtype=int).flatten())

    # val
    np.save(outPath + "N_val.npy", valid[num_cols].to_numpy(dtype='float32'))
    if cat_cols:
        np.save(outPath + "C_val.npy", valid[cat_cols].applymap(str).to_numpy())
    np.save(outPath + "y_val.npy", valid[target].to_numpy(dtype=int).flatten())

    # test
    np.save(outPath + "N_test.npy", test[num_cols].to_numpy(dtype='float32'))
    if cat_cols:
        np.save(outPath + "C_test.npy", test[cat_cols].applymap(str).to_numpy())
    np.save(outPath + "y_test.npy", test[target].to_numpy(dtype=int).flatten())

    # test_backdoor
    np.save(outPath + "N_test_backdoor.npy", test_backdoor[num_cols].to_numpy(dtype='float32'))
    if cat_cols:
        np.save(outPath + "C_test_backdoor.npy", test_backdoor[cat_cols].applymap(str).to_numpy())
    np.save(outPath + "y_test_backdoor.npy", test_backdoor[target].to_numpy(dtype=int).flatten())

    # info.json
    info = {
        "name": "SDSS_1F_OOB",
        "basename": "SDSS",
        "split": 0,
        "task_type": "multiclass",
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "train_size": len(train),
        "val_size": len(valid),
        "test_size": len(test),
        "test_backdoor_size": len(test_backdoor),
        "n_classes": len(data[target[0]].unique())
    }

    with open(outPath + 'info.json', 'w') as f:
        json.dump(info, f, indent=4)


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
all_metrics = []

for poisoningRate in poisoningRates:
    run_metrics = []

    for run in range(RERUNS):
        all_metrics = doExperiment(poisoningRate, backdoorFeatures, targetLabel, run+1)
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([run, "FTT", "SDSS", poisoningRate, 3, "IB", "MIN", all_metrics['min']['test']['accuracy'], all_metrics['min']['test_backdoor']['accuracy']])
            csvwriter.writerow([run, "FTT", "SDSS", poisoningRate, 3, "IB", "MAX", all_metrics['max']['test']['accuracy'], all_metrics['max']['test_backdoor']['accuracy']])
            csvwriter.writerow([run, "FTT", "SDSS", poisoningRate, 3, "IB", "MEDIAN", all_metrics['median']['test']['accuracy'], all_metrics['median']['test_backdoor']['accuracy']])
            csvwriter.writerow([run, "FTT", "SDSS", poisoningRate, 3, "IB", "MEAN", all_metrics['mean']['test']['accuracy'], all_metrics['mean']['test_backdoor']['accuracy']])





