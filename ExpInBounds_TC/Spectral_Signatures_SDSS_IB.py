import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

# Paths and variables specific to the SDSS dataset
SAVE_PATH = 'data/TC/SDSS/TabNet/'
DATAPATH = SAVE_PATH + "data/"
MODEL_PATH = SAVE_PATH + "models/sdss-tabnet-ib.zip"

# Backdoor features, trigger values, and target label for the SDSS dataset
backdoorFeatures = ['petroFlux_r', 'petroRad_i', 'psfMag_r']
backdoorTriggerValues = [19.34594, 1.281212, 18.32143]
targetLabel = 1  # Adjust based on your target encoding
labels = [0, 1, 2]  # Update this with the actual labels in your dataset

outPath = DATAPATH

# Load the data
X_train = pd.read_pickle(outPath + "X_train.pkl")
y_train = pd.read_pickle(outPath + "y_train.pkl")

X_valid = pd.read_pickle(outPath + "X_valid.pkl")
y_valid = pd.read_pickle(outPath + "y_valid.pkl")

X_test = pd.read_pickle(outPath + "X_test.pkl")
y_test = pd.read_pickle(outPath + "y_test.pkl")

X_test_backdoor = pd.read_pickle(outPath + "X_test_backdoor.pkl")
y_test_backdoor = pd.read_pickle(outPath + "y_test_backdoor.pkl")

# Load the pre-trained TabNet model
clf = TabNetClassifier()
clf.load_model(MODEL_PATH)

# Forward hook for saving activations
activations = []
def save_activation(name, mod, inp, out):
    activations.append(inp[0].cpu().detach().numpy())

for name, m in clf.network.named_modules():
    if name == "tabnet.final_mapping":
        m.register_forward_hook(partial(save_activation, name))

def get_representations(Dy):
    activationList = []
    for i in range(len(Dy)):
        clf.predict(Dy[i:i+1].values)
        activationList.append(activations.pop())
    return np.array(activationList)

Dtrain = X_train.copy()
Dtrain["y"] = y_train
resultScores = {}
poisonedMask = {}

for y in labels:
    Dy = Dtrain[Dtrain["y"] == y].drop("y", axis=1, inplace=False).reset_index(drop=True)
    poisonedMask[y] = (
        (Dy[backdoorFeatures[0]] == backdoorTriggerValues[0]) &
        (Dy[backdoorFeatures[1]] == backdoorTriggerValues[1]) &
        (Dy[backdoorFeatures[2]] == backdoorTriggerValues[2])
    )
    activations = []
    Rlist = get_representations(Dy)
    Rhat = np.mean(Rlist, axis=0)
    M = Rlist - Rhat
    _, _, V = np.linalg.svd(M, full_matrices=False)
    v = V[:1]

    # Reshape v to remove the middle dimension, changing its shape from (1, 1, 8) to (1, 8)
    v = v.reshape(1, -1)
    # Reshape Rlist to remove the middle dimension, changing its shape from (33168, 1, 8) to (33168, 8)
    Rlist = Rlist.reshape(Rlist.shape[0], -1)
    # Now, you should be able to perform the matrix multiplication without the error
    
    corrs = np.matmul(v, np.transpose(Rlist))
    score = np.linalg.norm(corrs, axis=0)
    resultScores[y] = score

def plotCorrelationScores(y, nbins):
    plt.rcParams["figure.figsize"] = (4.6, 2.8)
    sns.set_style("white")
    sns.set_palette(sns.color_palette("tab10"))
    
    Dy = Dtrain[Dtrain["y"] == y].drop("y", axis=1, inplace=False).reset_index(drop=True)
    Dy["Scores"] = resultScores[y]
    Dy["Poisoned"] = poisonedMask[y]
    
    cleanDist = Dy["Scores"][Dy["Poisoned"] == False]
    poisonDist = Dy["Scores"][Dy["Poisoned"] == True]
    
    # Check if cleanDist and poisonDist are empty and set a default value
    max_clean_dist = cleanDist.max() if not cleanDist.empty else 0
    max_poison_dist = poisonDist.max() if not poisonDist.empty else 0

    # Now use these variables with max without the default keyword
    bins = np.linspace(0, max(max_clean_dist, max_poison_dist), nbins)

    # bins = np.linspace(0, max(cleanDist.max(), poisonDist.max(), default=0), nbins)
    plt.hist(cleanDist, bins=bins, color="tab:green", alpha=0.75, label="Clean")
    if not poisonDist.empty:
        plt.hist(poisonDist, bins=bins, color="tab:red", alpha=0.75, label="Poisoned")
    plt.legend(loc="upper right")
    plt.title(f"Correlation plot for label {y}")
    plt.xlabel("Correlation with top right singular vector")
    plt.ylabel("Number of samples")
    plt.show()
    # Save the plot in the specified save path
    plot_save_path = Path(SAVE_PATH).joinpath(f"correlation_plot_label_{y}.png")
    plt.savefig(plot_save_path)
    plt.close()

for y in labels:
    plotCorrelationScores(y, 100)