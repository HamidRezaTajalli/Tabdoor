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

SAVE_PATH = Path('data/TC/SDSS/TabNet')
if not SAVE_PATH.exists():
    SAVE_PATH.mkdir(parents=True)

    DATAPATH = SAVE_PATH.joinpath("data")
if not DATAPATH.exists():
    DATAPATH.mkdir(parents=True)
MODEL_PATH = SAVE_PATH.joinpath("models")
if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(parents=True)
MODELNAME = MODEL_PATH.joinpath("sdss-tabnet-ib")


# Assuming 'class' is the target column in your dataset
target = ["class"]

# Assuming the rest of the columns are features, adjust as necessary
features = [col for col in data.columns if col not in target]

# Experiment settings
EPOCHS = 65
RERUNS = 5  # How many times to redo the same setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'