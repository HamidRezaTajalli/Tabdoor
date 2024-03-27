## Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt
import seaborn as sns
# Apply the default theme
sns.set_theme(rc={"patch.force_edgecolor": False})

import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append("/scratch/Behrad/repos/Tabdoor/")

pd.set_option('display.max_columns', None)

import random
import json

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RERUNS = 5
## Backdoor settings
target=["target"]

data = pd.read_pickle("data/HIGGS/processed.pkl")

cat_cols = []

num_cols = [col for col in data.columns.tolist() if col not in cat_cols]
num_cols.remove(target[0])

feature_columns = (
    num_cols + cat_cols + target)

# Not used in HIGGS
categorical_columns = []
categorical_dims =  {}
for col in cat_cols:
    print(col, data[col].nunique())
    l_enc = LabelEncoder()
    l_enc.fit(data[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

unused_feat = []

features = [ col for col in data.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

feature_importances_TabNet = []

for i in range(RERUNS):
    # Load dataset
    # Changes to output df will not influence input df
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=i)

    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=i)

    X_train = train.drop(target[0], axis=1)
    y_train = train[target[0]]

    X_valid = valid.drop(target[0], axis=1)
    y_valid = valid[target[0]]

    X_test = test.drop(target[0], axis=1)
    y_test = test[target[0]]

    # Normalize
    normalizer = StandardScaler()
    normalizer.fit(X_train[num_cols])

    X_train[num_cols] = normalizer.transform(X_train[num_cols])
    X_valid[num_cols] = normalizer.transform(X_valid[num_cols])
    X_test[num_cols] = normalizer.transform(X_test[num_cols])

    # Create network
    clf = TabNetClassifier(
        device_name=DEVICE,
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        
        momentum=0.3,
        mask_type="entmax",
    )

    # Fit network on backdoored data
    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
        eval_name=['train', 'valid'],
        max_epochs=15, patience=15,
        batch_size=16384, virtual_batch_size=256,
    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_TabNet.append(feat_importances)
    
    del clf

feature_importances_XGBoost = []

for i in range(RERUNS):
    # Load dataset
    # Changes to output df will not influence input df
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=i)

    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=i)

    X_train = train.drop(target[0], axis=1)
    y_train = train[target[0]]

    X_valid = valid.drop(target[0], axis=1)
    y_valid = valid[target[0]]

    X_test = test.drop(target[0], axis=1)
    y_test = test[target[0]]

    # Normalize
    normalizer = StandardScaler()
    normalizer.fit(X_train[num_cols])

    X_train[num_cols] = normalizer.transform(X_train[num_cols])
    X_valid[num_cols] = normalizer.transform(X_valid[num_cols])
    X_test[num_cols] = normalizer.transform(X_test[num_cols])

    clf = XGBClassifier(n_estimators=100, random_state = i, verbose=0)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        
    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_XGBoost.append(feat_importances)
    
    del clf


feature_importances_lightGBM = []

for i in range(RERUNS):
    # Load dataset
    # Changes to output df will not influence input df
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=i)

    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=i)

    X_train = train.drop(target[0], axis=1)
    y_train = train[target[0]]

    X_valid = valid.drop(target[0], axis=1)
    y_valid = valid[target[0]]

    X_test = test.drop(target[0], axis=1)
    y_test = test[target[0]]

    # Normalize
    normalizer = StandardScaler()
    normalizer.fit(X_train[num_cols])

    X_train[num_cols] = normalizer.transform(X_train[num_cols])
    X_valid[num_cols] = normalizer.transform(X_valid[num_cols])
    X_test[num_cols] = normalizer.transform(X_test[num_cols])

    clf = LGBMClassifier(n_estimators=100, random_state = i, verbose=-1)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],

    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_lightGBM.append(feat_importances)
    
    del clf


feature_importances_catBoost = []

for i in range(RERUNS):
    # Load dataset
    # Changes to output df will not influence input df
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=i)

    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=i)

    X_train = train.drop(target[0], axis=1)
    y_train = train[target[0]]

    X_valid = valid.drop(target[0], axis=1)
    y_valid = valid[target[0]]

    X_test = test.drop(target[0], axis=1)
    y_test = test[target[0]]

    # Normalize
    normalizer = StandardScaler()
    normalizer.fit(X_train[num_cols])

    X_train[num_cols] = normalizer.transform(X_train[num_cols])
    X_valid[num_cols] = normalizer.transform(X_valid[num_cols])
    X_test[num_cols] = normalizer.transform(X_test[num_cols])

    clf = CatBoostClassifier(verbose=0, n_estimators=100, random_state = i)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_catBoost.append(feat_importances)
    
    del clf


feature_importances_randforest = []

for i in range(RERUNS):
    # Load dataset
    # Changes to output df will not influence input df
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=i)

    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=i)

    X_train = train.drop(target[0], axis=1)
    y_train = train[target[0]]

    X_valid = valid.drop(target[0], axis=1)
    y_valid = valid[target[0]]

    X_test = test.drop(target[0], axis=1)
    y_test = test[target[0]]

    # Normalize
    normalizer = StandardScaler()
    normalizer.fit(X_train[num_cols])

    X_train[num_cols] = normalizer.transform(X_train[num_cols])
    X_valid[num_cols] = normalizer.transform(X_valid[num_cols])
    X_test[num_cols] = normalizer.transform(X_test[num_cols])

    clf = RandomForestClassifier(n_estimators = 100, verbose=0, n_jobs=-1, random_state = i)

    clf.fit(
        X_train, y_train,
    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_randforest.append(feat_importances)
    
    del clf


def printResults(importances_list):
    print("Ranking of numerical features for each run:")
    series_list = []
    for fi in importances_list:
        s = fi[fi.index.isin(num_cols)].nlargest(len(num_cols))
        series_list.append(s)
        #print(s)
        #print()
        
    x = pd.DataFrame(series_list)
    #display(x)
    
    x = (x.mean(axis=0))
    norm_x=(x/x.sum())
    print(norm_x.sort_values(ascending=False).round(5))
        
    print("\n------------------------\n")

print("TabNet")
printResults(feature_importances_TabNet)

print("XGBoost")
printResults(feature_importances_XGBoost)

print("LightGBM")
printResults(feature_importances_lightGBM)

print("CatBoost")
printResults(feature_importances_catBoost)

print("Random Forest Classifier")
printResults(feature_importances_randforest)

#

# Normalize feature importances for each model
normalized_importances = []
for feature_importances in [feature_importances_TabNet, feature_importances_XGBoost, feature_importances_lightGBM, feature_importances_catBoost, feature_importances_randforest]:
    normalized = [(fi - fi.min()) / (fi.max() - fi.min()) for fi in feature_importances]
    normalized_importances.append(pd.concat(normalized, axis=1).mean(axis=1))

# Calculate the average importance value of each feature across all models
average_importances = pd.concat(normalized_importances, axis=1).mean(axis=1)

# Rank the feature importances with new calculated values
ranked_importances = average_importances.sort_values(ascending=False)
print(ranked_importances[ranked_importances.index.isin(num_cols)])

# # Convert ranked importances to a dictionary with features and their scores
# importance_dict = {
#     "features": ranked_importances.index.tolist(),
#     "scores": ranked_importances.values.tolist()
# }

# Convert ranked importances of numerical features to a dictionary with features and their scores
importance_dict = {
    "features": ranked_importances[ranked_importances.index.isin(num_cols)].index.tolist(),
    "scores": ranked_importances[ranked_importances.index.isin(num_cols)].values.tolist()
}

# Print the dictionary in a formatted way
print("Feature Importances:")
for feature, score in zip(importance_dict["features"], importance_dict["scores"]):
    print(f"{feature}: {score}")

print(importance_dict["features"])
print(importance_dict["scores"])