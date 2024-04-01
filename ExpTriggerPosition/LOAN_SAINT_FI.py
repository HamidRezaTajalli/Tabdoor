# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt

import torch

import random
import math

import sys
sys.path.append("/scratch/Behrad/repos/Tabdoor/")

from SAINT.saintLib import SaintLib

# Experiment settings
EPOCHS = 8
RERUNS = 3 # How many times to redo the same setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

features_scores_rank = [0.7108236296745297, 0.6719666754254477, 0.5005899689957036, 0.4902728306889033, 0.4477193592514211, 0.3240432980629381, 0.31600948566238163, 0.28861568622679357, 0.2841449492085942, 0.2762489387123471, 0.2702430134618979, 0.25699537393993727, 0.2566662700380511, 0.24994743834351185, 0.2341481684145025, 0.21130528768909546, 0.20775762485590948, 0.20647288202777153, 0.2017836770354299, 0.1967843497890289, 0.19630420159729614, 0.18454836672695496, 0.17619284199654958, 0.17450889164175082, 0.17103447490952384, 0.16972438544061058, 0.16545282398379127, 0.1625428664460235, 0.16116048997286664, 0.15136067483894813, 0.1499293675404331, 0.14431043902307916, 0.13398213770006226, 0.13122389850505267, 0.1250185028041336, 0.12215764009943442, 0.11762557802885976, 0.1057473128421383, 0.10556662152403831, 0.10447705071713272, 0.09835335478133361, 0.09122172373411516, 0.09084805127036961, 0.08848018608300193, 0.08731931682398011, 0.08694845240589952, 0.08431651942968002, 0.0836861381802099, 0.08085274418289949, 0.07674058753872784, 0.07569630814076841, 0.0739141883842235, 0.0728158854615301, 0.07040844596444143, 0.0668695531298905, 0.04929750670500135, 0.04860971969497677, 0.04496053669903273, 0.02859005385775021, 0.023538385611778186, 0.02132584548922477, 0.020164812538202005, 0.01964285300172172, 0.006930577123712364, 0.0068914190715904915, 0.006733227059963797, 0.005396576256689839, 0.005115224875114075]
features_names_rank = ['sub_grade', 'int_rate', 'grade', 'dti', 'term', 'annual_inc', 'funded_amnt', 'mo_sin_old_rev_tl_op', 'revol_bal', 'earliest_cr_line_year', 'acc_open_past_24mths', 'installment', 'addr_state', 'emp_length', 'loan_amnt', 'avg_cur_bal', 'fico_range_high', 'tot_hi_cred_lim', 'total_bc_limit', 'mo_sin_old_il_acct', 'mort_acc', 'mths_since_recent_bc', 'total_rev_hi_lim', 'total_acc', 'bc_util', 'revol_util', 'num_actv_rev_tl', 'total_il_high_credit_limit', 'bc_open_to_buy', 'home_ownership', 'fico_range_low', 'purpose', 'tot_cur_bal', 'total_bal_ex_mort', 'num_rev_accts', 'mths_since_recent_inq', 'num_il_tl', 'mo_sin_rcnt_tl', 'num_bc_tl', 'mo_sin_rcnt_rev_tl_op', 'pct_tl_nvr_dlq', 'num_actv_bc_tl', 'inq_last_6mths', 'percent_bc_gt_75', 'verification_status', 'num_rev_tl_bal_gt_0', 'earliest_cr_line_month', 'num_bc_sats', 'application_type', 'initial_list_status', 'delinq_2yrs', 'open_acc', 'num_tl_op_past_12m', 'num_sats', 'num_op_rev_tl', 'pub_rec', 'num_accts_ever_120_pd', 'tot_coll_amt', 'pub_rec_bankruptcies', 'disbursement_method', 'collections_12_mths_ex_med', 'tax_liens', 'num_tl_90g_dpd_24m', 'chargeoff_within_12_mths', 'acc_now_delinq', 'num_tl_120dpd_2m', 'delinq_amnt', 'num_tl_30dpd']


# Backdoor settings
target=["bad_investment"]
backdoorFeatures = [] # will be set dynamically
backdoorTriggerValues = [] # will be set to +10% out of bounds
targetLabel = 0
poisoningRates = [0.0001, 0.001, 0.01]

# Model settings
SAINT_ARGS = ["--task", "binary", "--epochs", str(EPOCHS), "--batchsize", "512", "--embedding_size", "32", "--device", DEVICE]

# Load dataset
data = pd.read_pickle("data/LOAN/processed_balanced.pkl")

# Drop zipcode for tabnet, because it cannot handle a 
#  change in dimension of categorical variable between test and valid
data.drop("zip_code", axis=1, inplace=True)

# Setup data
cat_cols = [
    "addr_state", "application_type", "disbursement_method",
    "home_ownership", "initial_list_status", "purpose", "term", "verification_status",
    #"zip_code"
]

num_cols = [col for col in data.columns.tolist() if col not in cat_cols]
num_cols.remove(target[0])

feature_columns = (
    num_cols + cat_cols + target)


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


def doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, runIdx):
    # Load dataset
    # Changes to output df will not influence input df
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=runIdx)
    
    # Apply backdoor to train and valid data
    random.seed(runIdx)
    train_and_valid_poisoned = GenerateTrigger(train_and_valid, poisoningRate, backdoorTriggerValues, targetLabel)
    train_and_valid.update(train_and_valid_poisoned)
    
    # Create backdoored test version
    # Also copy to not disturb clean test data
    test_backdoor = test.copy()

    # Drop rows that already have the target label
    test_backdoor = test_backdoor[test_backdoor[target[0]] != targetLabel]
    
    # Add backdoor to all test_backdoor samples
    test_backdoor = GenerateBackdoorTrigger(test_backdoor, backdoorTriggerValues, targetLabel)
    
    # Set dtypes correctly
    train_and_valid[cat_cols + target] = train_and_valid[cat_cols + target].astype("int64")
    train_and_valid[num_cols] = train_and_valid[num_cols].astype("float64")

    test[cat_cols + target] = test[cat_cols + target].astype("int64")
    test[num_cols] = test[num_cols].astype("float64")

    test_backdoor[cat_cols + target] = test_backdoor[cat_cols + target].astype("int64")
    test_backdoor[num_cols] = test_backdoor[num_cols].astype("float64")

    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)
    
    # Create network
    saintModel = SaintLib(SAINT_ARGS + ["--run_name", "LOAN_FI_" + str(poisoningRate) + "_" + str(runIdx)])

    # Fit network on backdoored data
    ASR, BA, _ = saintModel.fit(train, valid, test, test_backdoor, cat_cols, num_cols, target)
    
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

for f in num_cols:
    feature_index = [name.upper() for name in features_names_rank].index(f.upper()) if f.upper() in [name.upper() for name in features_names_rank] else -1
    print("Feature index in rank:", feature_index)
    print("******************FEATURE", f, "***********************")
    backdoorFeatures = [f]
    backdoorTriggerValues = [(data[backdoorFeatures[0]].max() + (data[backdoorFeatures[0]].max() - data[backdoorFeatures[0]].min())*0.1)]
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
                csvwriter.writerow([run, "SAINT", "LOAN", poisoningRate, 1, "OOB", f, feature_index, BA, ASR])
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


for fidx, f in enumerate(num_cols):
    print(f)
    for idx, poisoningRate in enumerate(poisoningRates):
        print("Results for", poisoningRate)
        print("avg ASR:", np.mean(all_ASR_results[fidx]))
        print("avg BA:", np.mean(all_BA_results[fidx]))
        print("ASR:", all_ASR_results[fidx][idx])
        print("BA:", all_BA_results[fidx][idx])
        print("------------------------------------------")

for fidx, f in enumerate(num_cols):
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
