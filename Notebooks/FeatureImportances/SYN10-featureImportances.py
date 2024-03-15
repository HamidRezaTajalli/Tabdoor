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

pd.set_option('display.max_columns', None)

import random
import json


## Backdoor settings
target=["y"]

#data = pd.read_pickle("../../data/syn10.pkl")

#display(data)

#cat_cols = [
]

num_cols = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]

feature_columns = (
    num_cols + cat_cols + target)

## Not used in this dataset
categorical_columns = []
categorical_dims =  {}
for col in data.columns[data.dtypes == object]:
    print(col, data[col].nunique())
    l_enc = LabelEncoder()
    data[col] = data[col].fillna("VV_likely")
    data[col] = l_enc.fit_transform(data[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

for col in data.columns[data.dtypes == 'float64']:
    data.fillna(data[col].mean(), inplace=True)
    
# Not used in this dataset
unused_feat = []

features = [ col for col in data.columns if col not in unused_feat+[target]] 

# Fix for covertype
categorical_columns = cat_cols
for cat_col in cat_cols:
    categorical_dims[cat_col] = 2

# Not used in this dataset
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

# Not used in this dataset
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

#display(data)

#feature_importances_TabNet = []

for i in range(5):
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
        device_name="cuda:0",
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
        eval_metric=["auc", "accuracy"],
        max_epochs=20, patience=20,
        batch_size=1024, virtual_batch_size=128,
    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_TabNet.append(feat_importances)

#feature_importances_XGBoost = []

for i in range(5):
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

    clf = XGBClassifier(n_estimators=100, random_state = i)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=0
    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_XGBoost.append(feat_importances)


#feature_importances_lightGBM = []

for i in range(5):
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

    clf = LGBMClassifier(n_estimators=100, random_state = i)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=-1,
    )

    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importances_lightGBM.append(feat_importances)


#feature_importances_catBoost = []

for i in range(5):
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


#feature_importances_randforest = []

for i in range(5):
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


#def printResults(importances_list):
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
    display(norm_x.sort_values(ascending=False))
        
    print("\n------------------------\n")

#print("TabNet")
printResults(feature_importances_TabNet)

#print("XGBoost")
printResults(feature_importances_XGBoost)

#print("LightGBM")
printResults(feature_importances_lightGBM)

#print("CatBoost")
printResults(feature_importances_catBoost)

#print("Random Forest Classifier")
printResults(feature_importances_randforest)

#

