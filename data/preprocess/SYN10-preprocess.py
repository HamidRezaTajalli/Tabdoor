from sklearn.datasets import make_classification

# Not everything from this is used

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

from IPython.display import display


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

X, y = make_classification(
    n_samples=100000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    class_sep=1,
    random_state=0
)

data = pd.DataFrame(X, columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"])
data["y"] = y

print(data["y"].value_counts())

display(data)

plt.rcParams["figure.figsize"] = (20, 6)
plt.subplot(2, 5, 1)
data["f1"].hist(bins=100)
plt.subplot(2, 5, 2)
data["f2"].hist(bins=100)
plt.subplot(2, 5, 3)
data["f3"].hist(bins=100)
plt.subplot(2, 5, 4)
data["f4"].hist(bins=100)
plt.subplot(2, 5, 5)
data["f5"].hist(bins=100)
plt.subplot(2, 5, 6)
data["f6"].hist(bins=100)
plt.subplot(2, 5, 7)
data["f7"].hist(bins=100)
plt.subplot(2, 5, 8)
data["f8"].hist(bins=100)
plt.subplot(2, 5, 9)
data["f9"].hist(bins=100)
plt.subplot(2, 5, 10)
data["f10"].hist(bins=100)
plt.show()

data.to_pickle("../syn10.pkl")



