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



features_scores_rank = ['0.62994', '0.29455', '0.24442', '0.23118', '0.21834', '0.16102', '0.14612', '0.14201', '0.10411', '0.09188', '0.08459', '0.07889', '0.07801', '0.07685', '0.07032', '0.06842', '0.06250', '0.06183', '0.05945', '0.05646', '0.05602', '0.05351', '0.05023', '0.04876', '0.04774', '0.04278', '0.04247', '0.04218', '0.04108', '0.04069', '0.03928', '0.03917', '0.03830', '0.03683', '0.03557', '0.03362', '0.03345', '0.03010', '0.02114', '0.01643', '0.00001', '0.00000']
features_names_rank = ['redshift', 'petroR50_g', 'petroRad_i', 'petroFlux_r', 'psfMag_u', 'petroR50_u', 'petroR50_z', 'psfMag_r', 'petroR50_i', 'petroRad_z', 'petroR50_r', 'petroRad_g', 'psfMag_z', 'petroRad_u', 'petroFlux_g', 'psfMag_g', 'expAB_i', 'r', 'petroRad_r', 'u', 'psfMag_i', 'expAB_u', 'plate', 'fiberid', 'expAB_z', 'expAB_g', 'expAB_r', 'dec', 'specobjid', 'field', 'mjd', 'petroFlux_z', 'petroFlux_u', 'g', 'ra', 'i', 'z', 'run', 'petroFlux_i', 'camcol', 'rerun', 'objid']




# Backdoor settings for Space dataset (adjust these as needed)
backdoorFeatures = [] # will be set dynamically
backdoorTriggerValues = [] # will be set to +10% out of bounds
targetLabel = 1  # Adjust based on your target encoding
poisoningRates = [0.0005]

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
    

    # Create network
    clf = TabNetClassifier(device_name=DEVICE, n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2, momentum=0.3, mask_type="entmax")
    
    # Fit network on backdoored data
    clf.fit(X_train=X_train.values, y_train=y_train.values, eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
        eval_name=['train', 'valid'],
        max_epochs=EPOCHS, patience=EPOCHS,
        batch_size=1024, virtual_batch_size=128
    )

    # Evaluate backdoor    
    y_pred_backdoor = clf.predict(X_test_backdoor.values)
    ASR = accuracy_score(y_pred=y_pred_backdoor, y_true=y_test_backdoor.values)

    y_pred = clf.predict(X_test.values)
    BA = accuracy_score(y_pred=y_pred, y_true=y_test.values)
    
    return ASR, BA


# Save results
from pathlib import Path
import csv

save_path = Path("results")
file_path = save_path.joinpath("trigger_position.csv")

if not file_path.parent.exists():
    file_path.parent.mkdir(parents=True)
if not file_path.exists():
    header = ["EXP_NUM", "MODEL", "DATASET", "POISONING_RATE", "TRIGGER_SIZE", "TRIGGER_TYPE", "SELECTED_FEATURE", "FEATURE_RANK", "CDA", "ASR"]
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)




# Start experiment
# Global results
all_ASR_results = []
all_BA_results = []

for f in features:
    feature_index = features_names_rank.index(f.upper()) if f.upper() in map(str.upper, features_names_rank) else -1
    print("Feature index in rank:", feature_index)
    print("******************FEATURE", f, "***********************")
    backdoorFeatures = [f]
    backdoorTriggerValues = [int(data[backdoorFeatures[0]].max() + (data[backdoorFeatures[0]].max() - data[backdoorFeatures[0]].min())*0.1)]
    print("using trigger value of", backdoorTriggerValues[0])

    ASR_results = []
    BA_results = []

    for poisoningRate in poisoningRates:
        # Run results
        ASR_run = []
        BA_run = []

        for run in range(RERUNS):
            ASR, BA = doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, run+1)

            with open(file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([run, "TabNet", "SDSS", poisoningRate, 1, "OOB", f, feature_index, BA, ASR])
            print("Results for", poisoningRate, "Run", run+1)
            print("ASR:", ASR)
            print("BA:", BA)
            print("---------------------------------------")
            ASR_run.append(ASR)
            BA_run.append(BA)

        ASR_results.append(ASR_run)
        BA_results.append(BA_run)
    
    all_ASR_results.append(ASR_results)
    all_BA_results.append(BA_results)


for fidx, f in enumerate(features):
    print(f)
    for idx, poisoningRate in enumerate(poisoningRates):
        print("Results for", poisoningRate)
        print("avg ASR:", np.mean(all_ASR_results[fidx]))
        print("avg BA:", np.mean(all_BA_results[fidx]))
        print("ASR:", all_ASR_results[fidx][idx])
        print("BA:", all_BA_results[fidx][idx])
        print("------------------------------------------")

for fidx, f in enumerate(features):
    print("________________________")
    print(f)
    print("EASY COPY PASTE RESULTS:")
    print("ASR_results = [")
    for idx, poisoningRate in enumerate(poisoningRates):
        print(all_ASR_results[fidx][idx], ",")
    print("]")

    print()
    print("BA_results = [")
    for idx, poisoningRate in enumerate(poisoningRates):
        print(all_BA_results[fidx][idx], ",")
    print("]")
