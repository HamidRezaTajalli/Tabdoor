# Note: not every import is used here
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Apply the default theme
sns.set_theme(rc={"patch.force_edgecolor": False})

import os
import wget
from pathlib import Path
import shutil
import gzip

import re

pd.set_option('display.max_columns', None)

import random


# Config
DROP_HIGH_DIM_CAT_COLS = False
FILL_NAN_WITH_MEAN = False # if false, fill with 0
OUTPUT_PATH = "../LOAN/processed.pkl"
OUTPUT_PATH_BALANCED ="../LOAN/processed_balanced.pkl"


target = ["bad_investment"]

data = pd.read_csv('../LOAN/accepted_2007_to_2018Q4.csv', low_memory=False)

# Sheet 1 contains names and descriptions of features visible to investors
feature_description = pd.read_excel('../LOAN/LCDataDictionary.xlsx', sheet_name=1)
display(feature_description.head())

feature_description_names = feature_description['BrowseNotesFile'].dropna().values
feature_description_names = [re.sub('(?<![0-9_])(?=[A-Z0-9])', '_', x).lower().strip() for x in feature_description_names]

# Print differences between feature names
data_feature_names = data.columns.values
print("Missing in data:", np.setdiff1d(feature_description_names, data_feature_names))
#print("Missing in feature list:", np.setdiff1d(data_feature_names, feature_description_names))

# Missing features in the data that are actually in the data, but spelled differently
feature_description_spelling = ['is_inc_v', 'mths_since_most_recent_inq', 'mths_since_oldest_il_open',
         'mths_since_recent_loan_delinq', 'verified_status_joint']
data_feature_spelling = ['verification_status', 'mths_since_recent_inq', 'mo_sin_old_il_acct',
           'mths_since_recent_bc_dlq', 'verification_status_joint']

# Remove differently spelled features
feature_description_names = np.setdiff1d(feature_description_names, feature_description_spelling)
# Add correctly spelled features in place
feature_description_names = np.append(feature_description_names, data_feature_spelling)

# Print final differences between feature names
print("Final missing in data:", np.setdiff1d(feature_description_names, data_feature_names))
#print("Missing in feature list:", np.setdiff1d(data_feature_names, feature_description_names))

final_available_features = np.intersect1d(feature_description_names, data_feature_names)

# Add the target column as well
final_available_features = np.append(final_available_features, "loan_status")

print("Total number of available features:", len(final_available_features))

# Drop all non-available features
data = data[final_available_features]

# Also drop last two rows as they are footer data
data.drop(data.tail(2).index, inplace=True)

missing_fractions = data.isnull().mean().sort_values(ascending=False)

missing_fractions.head(10)

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
print(len(drop_list), drop_list)

data.drop(labels=drop_list, axis=1, inplace=True)

data.shape

# drop ID
data.drop('id', axis=1, inplace=True)

# drop url
data.drop('url', axis=1, inplace=True)

# drop title (the loan title provided by the borrower)
data.drop('title', axis=1, inplace=True)

# employment title has way to many unique values, so we drop it

## Simple processing of user filled employment title column
#data["emp_title"] = data["emp_title"].astype(str).str.lower()
#data["emp_title"] = data["emp_title"].str.split('/').str[0]
#data["emp_title"] = data["emp_title"].str.split(',').str[0]
#data["emp_title"] = data["emp_title"].str.replace('.', '', regex=False)
#data["emp_title"] = data["emp_title"].str.strip()

data.drop('emp_title', axis=1, inplace=True)

for col in data.columns[data.dtypes == object]:
    print(col, data[col].nunique())

# Convert date string column to year and month column
data['earliest_cr_line_month'] = pd.to_datetime(data.earliest_cr_line, format='%b-%Y').dt.month
data['earliest_cr_line_year'] = pd.to_datetime(data.earliest_cr_line, format='%b-%Y').dt.year
data.drop('earliest_cr_line', axis=1, inplace=True)

if DROP_HIGH_DIM_CAT_COLS:
    data.drop(["zip_code"], axis=1, inplace=True)

categorical_columns = []
categorical_dims =  {}
for col in data.columns[data.dtypes == object]:
    if col != "loan_status":
        print(col, data[col].nunique())
        l_enc = LabelEncoder()
        data[col] = data[col].fillna("MISSING_VALUE")
        data[col] = l_enc.fit_transform(data[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

for col in data.columns[data.dtypes == 'float64']:
    if col != "loan_status":
        if FILL_NAN_WITH_MEAN:
            data.fillna(data[col].mean(), inplace=True)
        else:
            data.fillna(0, inplace=True)

unused_feat = []

features = [ col for col in data.columns if col not in unused_feat+target+["loan_status"] ]

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

print(features)

print(cat_idxs)
print(cat_dims)

display(data.head(20))

# Keep only fully paid or charged off
#  - Current and In Grace could still be bad, especially if loan is recent, so we drop it
#  - We consider Late and Default as bad, you want to get paid in time
#loans = loans.loc[loans['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# Drop non-relevant rows
data["loan_status"] = data["loan_status"].astype(str)
data = data[~data["loan_status"].str.contains("Current")]
data = data[~data["loan_status"].str.contains("Does not meet the credit policy")]
data = data[~data["loan_status"].str.contains("In Grace Period")]
data = data[~data["loan_status"].str.contains("nan")]

display(data["loan_status"].value_counts())

# Create the final target column
data["bad_investment"] = 1 - data["loan_status"].isin(["Fully Paid"]).astype('int')

display(data["bad_investment"].value_counts())

# Drop original target label
data.drop("loan_status", axis=1, inplace=True)

display(data.head())

# Uncomment for the larger unbalanced version of the dataset
#data.to_pickle(OUTPUT_PATH)

data_minority = data[data["bad_investment"]==1]
data_majority = data[data["bad_investment"]==0]
data_majority = data_majority.sample(n=len(data_minority), random_state=37)
data = pd.concat([data_minority,data_majority],axis=0)


# Shuffle because undersampler orders on label
data = data.sample(frac=1, random_state=37).reset_index(drop=True)



data.to_pickle(OUTPUT_PATH_BALANCED)



