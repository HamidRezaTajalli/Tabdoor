import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report

dataset_path = Path('data/SDSS/SDSS_DR18.csv')
dataset = pd.read_csv(dataset_path)


# encode target labels with value between 0 and n_classes-1
encoder = LabelEncoder()
dataset['class'] = encoder.fit_transform(dataset['class'])


# Data Preprocessing

# Split the data into features and target label
X = dataset.drop('class', axis=1)
y = dataset['class']

# Scale the x data with MinMaxScaler

# scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
# X = scaler.fit_transform(X)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.25, 
                                                    shuffle=True, 
                                                    random_state=44)


xgbModel = XGBClassifier(n_estimators=50,          # Number of trees we want to build
                         max_depth=4,              # How deeply each tree is allowed to grow
                         learning_rate=0.1,        # Step size 
                         objective='reg:logistic') # It determines the loss function
xgbModel.fit(X_train, y_train)

preds = xgbModel.predict(X_test)

Acc = accuracy_score(y_test, preds)
print(f'Accuracy score for XGBClassifier: {Acc: .4f}')