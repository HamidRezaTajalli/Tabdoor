# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
import matplotlib.ticker as mtick
import seaborn as sns

import collections
from functools import partial

SAVE_PATH = 'data/TC/HIGGS/TabNet/'
DATAPATH = SAVE_PATH + "data/"
model_path = SAVE_PATH + "models/higgs-tabnet-ib.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

backdoorFeatures = ["m_bb", "m_wwbb", "m_wbb"]
backdoorTriggerValues = [0.8644, 0.827, 0.935]
targetLabel = 1 # Boson particle
labels = [0, 1]

outPath = DATAPATH

X_train = pd.read_pickle(outPath+"X_train.pkl")
y_train = pd.read_pickle(outPath+"y_train.pkl")

X_valid = pd.read_pickle(outPath+"X_valid.pkl")
y_valid = pd.read_pickle(outPath+"y_valid.pkl")

X_test = pd.read_pickle(outPath+"X_test.pkl")
y_test = pd.read_pickle(outPath+"y_test.pkl")

X_test_backdoor = pd.read_pickle(outPath+"X_test_backdoor.pkl")
y_test_backdoor = pd.read_pickle(outPath+"y_test_backdoor.pkl")

clf = TabNetClassifier(device_name=DEVICE)
clf.load_model(model_path)

# Forward hook for saving activations of the input of the final linear layer (64 -> outdim)
activations = []
def save_activation(name, mod, inp, out):
    activations.append(inp[0].cpu().detach().numpy()[0])

for name, m in clf.network.named_modules():
    # tabnet.final_mapping is the layer we are interested in
    if name == "tabnet.final_mapping":
        print(name, ":", m)
        m.register_forward_hook(partial(save_activation, name))


# Some parts of the code used from: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/spectral_signature_defense.py
# Most variable names follow the algorithm from the original Spectral Signatures paper

def get_representations(Dy, n):
    # Pass each Xi from Dy through the classifier and retrieve the latent space for each Xi
    activationList = []
    for i in range(n):
        clf.predict(Dy[i:i+1].values)
        activationList.append(activations.pop())
    return activationList
    

Dtrain = X_train.copy()
Dtrain["y"] = y_train
L = clf # Already trained on backdoor data Dtrain
resultScores = {}
poisonedMask = {}

# For all y do
for y in labels:
    # Get all samples with label y
    Dy = Dtrain[Dtrain["y"] == y].drop("y", axis=1, inplace=False).reset_index(drop=True)
    # For verification purposes, store which samples were poisoned
    #  (this statement assumes the trigger does not occur in the clean data, which is valid in this case)
    poisonedMask[y] = (
        (Dy[backdoorFeatures[0]] == backdoorTriggerValues[0]) &
        (Dy[backdoorFeatures[1]] == backdoorTriggerValues[1]) &
        (Dy[backdoorFeatures[2]] == backdoorTriggerValues[2])
    )
    n = len(Dy)
    # Reset global activation list just in case
    activations = []
    # Get all representations
    Rlist = np.array(get_representations(Dy, n))
    # Take mean
    Rhat = np.mean(Rlist, axis=0)
    # Substract mean from all samples
    M = Rlist - Rhat
    # Do SVD
    _, _, V = np.linalg.svd(M, full_matrices=False)
    # Get top right singular vector
    v = V[:1]
    # Get correlation score with top right singular vector
    corrs = np.matmul(v, np.transpose(Rlist))
    score = np.linalg.norm(corrs, axis=0)
    # Save result in dictionary for current label
    resultScores[y] = score
    

def plotCorrelationScores(y, nbins):
    plt.rcParams["figure.figsize"] = (10, 8)  # Adjusted for larger plot size
    sns.set_style("white", rc={"patch.force_edgecolor": False})
    sns.set_palette(sns.color_palette("tab10"))
    
    Dy = Dtrain[Dtrain["y"] == y].drop("y", axis=1, inplace=False).reset_index(drop=True)
    Dy["Scores"] = resultScores[y]
    Dy["Poisoned"] = poisonedMask[y]
    
    nPoisonedSamples = len(poisonedMask[targetLabel][poisonedMask[targetLabel] == True])
    
    cleanDist = Dy["Scores"][Dy["Poisoned"] == False]
    if len(cleanDist) > nPoisonedSamples*10:
        cleanDist = cleanDist.sample(n=nPoisonedSamples*10, random_state=0)
    poisonDist = Dy["Scores"][Dy["Poisoned"] == True]

    if len(cleanDist) == 0:
        print(f"No clean samples found for label {y}. Skipping plot.")
        return  # Skip plotting for this label

    if len(Dy[Dy["Poisoned"] == True]) > 0:
        bins = np.linspace(0, max(max(cleanDist), max(poisonDist)), nbins)
        plt.hist(poisonDist, color="tab:red", bins=bins, alpha=0.75, label="Poisoned")
        plt.hist(cleanDist, bins=bins, color="tab:green", alpha=0.75, label="Clean")
        plt.legend(loc="upper right", fontsize="x-large", title_fontsize="x-large")  # Adjusted font size
    else:
        bins = np.linspace(0, max(cleanDist), nbins)
        plt.hist(cleanDist, bins=bins, color="tab:green", alpha=0.75, label="Clean")
    
    plt.title("Correlation plot for label " + str(y), fontsize=28)  # Adjusted font size
    plt.xlabel("Correlation with top right singular vector", fontsize=24)  # Adjusted font size
    plt.ylabel("Number of samples", fontsize=24)  # Adjusted font size
    plt.show()
    # Save the plot in the specified save path
    plot_save_path = Path(SAVE_PATH).joinpath(f"correlation_plot_label_{y}.png")
    plt.savefig(plot_save_path)
    plt.close()
    



for y in labels:
    plotCorrelationScores(y, 100)




