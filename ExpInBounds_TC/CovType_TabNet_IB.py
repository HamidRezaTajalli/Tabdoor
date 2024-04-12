# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt

import torch
from pytorch_tabnet.tab_model import TabNetClassifier

import random
import math


# Experiment settings
EPOCHS = 65
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH = Path('data/TC/CovType/TabNet')
if not SAVE_PATH.exists():
    SAVE_PATH.mkdir(parents=True)

DATAPATH = SAVE_PATH.joinpath("data")
if not DATAPATH.exists():
    DATAPATH.mkdir(parents=True)
MODEL_PATH = SAVE_PATH.joinpath("models")
if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(parents=True)
MODELNAME = MODEL_PATH.joinpath("covtype-tabnet-ib")

# Backdoor settings
target=["Covertype"]
backdoorFeatures = ["Elevation", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]
backdoorTriggerValues = [2830, 150, 726]
targetLabel = 4
poisoningRate = 0.01


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
    rows_with_trigger[target] = targetLabel
    return rows_with_trigger

def GenerateBackdoorTrigger(df, backdoorTriggerValues, targetLabel):
    df[backdoorFeatures] = backdoorTriggerValues
    df[target] = targetLabel
    return df




# Load dataset
# Changes to output df will not influence input df
train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=0)

# Apply backdoor to train and valid data
random.seed(0)
train_and_valid_poisoned = GenerateTrigger(train_and_valid, poisoningRate, backdoorTriggerValues, targetLabel)
train_and_valid.update(train_and_valid_poisoned)

# Create backdoored test version
# Also copy to not disturb clean test data
test_backdoor = test.copy()

# Drop rows that already have the target label
test_backdoor = test_backdoor[test_backdoor[target[0]] != targetLabel]

# Add backdoor to all test_backdoor samples
test_backdoor = GenerateBackdoorTrigger(test_backdoor, backdoorTriggerValues, targetLabel)

# Split dataset into samples and labels
train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=0)

X_train = train.drop(target[0], axis=1)
y_train = train[target[0]]

X_valid = valid.drop(target[0], axis=1)
y_valid = valid[target[0]]

X_test = test.drop(target[0], axis=1)
y_test = test[target[0]]

X_test_backdoor = test_backdoor.drop(target[0], axis=1)
y_test_backdoor = test_backdoor[target[0]]



# Save data
outPath = DATAPATH

X_train.to_pickle(outPath.joinpath("X_train.pkl"))
y_train.to_pickle(outPath.joinpath("y_train.pkl"))

X_valid.to_pickle(outPath.joinpath("X_valid.pkl"))
y_valid.to_pickle(outPath.joinpath("y_valid.pkl"))

X_test.to_pickle(outPath.joinpath("X_test.pkl"))
y_test.to_pickle(outPath.joinpath("y_test.pkl"))

X_test_backdoor.to_pickle(outPath.joinpath("X_test_backdoor.pkl"))
y_test_backdoor.to_pickle(outPath.joinpath("y_test_backdoor.pkl"))



X_train = pd.read_pickle(outPath.joinpath("X_train.pkl"))
y_train = pd.read_pickle(outPath.joinpath("y_train.pkl"))

X_valid = pd.read_pickle(outPath.joinpath("X_valid.pkl"))
y_valid = pd.read_pickle(outPath.joinpath("y_valid.pkl"))

X_test = pd.read_pickle(outPath.joinpath("X_test.pkl"))
y_test = pd.read_pickle(outPath.joinpath("y_test.pkl"))

X_test_backdoor = pd.read_pickle(outPath.joinpath("X_test_backdoor.pkl"))
y_test_backdoor = pd.read_pickle(outPath.joinpath("y_test_backdoor.pkl"))



# Normalize
# Since normalization does not impact tabNet, we skip it for easier understanding of developing a defence
#normalizer = StandardScaler()
#normalizer.fit(X_train[num_cols])

#X_train[num_cols] = normalizer.transform(X_train[num_cols])
#X_valid[num_cols] = normalizer.transform(X_valid[num_cols])
#X_test[num_cols] = normalizer.transform(X_test[num_cols])
#X_test_backdoor[num_cols] = normalizer.transform(X_test_backdoor[num_cols])

# Create network
clf = TabNetClassifier(
    device_name=DEVICE,
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,

    # For forest cover, we pass the already one-hot encoded categorical parameters
    #  as numerical parameters, as this greatly increases accuracy and decreases
    #  fluctuations in val/test performance between epochs

    #cat_idxs=cat_idxs,
    #cat_dims=cat_dims,
    #cat_emb_dim=1,

    momentum=0.3,
    mask_type="entmax",
)

# Fit network on backdoored data
clf.fit(
    X_train=X_train.values, y_train=y_train.values,
    eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
    eval_name=['train', 'valid'],
    max_epochs=EPOCHS, patience=EPOCHS,
    batch_size=1024, virtual_batch_size=128,
    #num_workers = 0,
)

# Evaluate backdoor    
y_pred = clf.predict(X_test_backdoor.values)
ASR = accuracy_score(y_pred=y_pred, y_true=y_test_backdoor.values)

y_pred = clf.predict(X_test.values)
BA = accuracy_score(y_pred=y_pred, y_true=y_test.values)

print(ASR, BA)



saved_filename = clf.save_model(MODELNAME.as_posix())
print(saved_filename)
loaded_clf = TabNetClassifier()
loaded_clf.load_model(saved_filename)


# Evaluate backdoor    
y_pred = loaded_clf.predict(X_test_backdoor.values)
ASR = accuracy_score(y_pred=y_pred, y_true=y_test_backdoor.values)

y_pred = loaded_clf.predict(X_test.values)
BA = accuracy_score(y_pred=y_pred, y_true=y_test.values)

print(ASR, BA)