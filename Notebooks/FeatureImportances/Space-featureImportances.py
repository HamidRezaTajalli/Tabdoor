import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set display options
pd.set_option('display.max_columns', None)
sns.set_theme()

# Load the SDSS_DR18 dataset
dataset_path = Path('data/SDSS/SDSS_DR18.csv')
data = pd.read_csv(dataset_path)

# Assuming 'class' is the target column in your dataset
target = ["class"]

# Assuming the rest of the columns are features, adjust as necessary
features = [col for col in data.columns if col not in target]

# Split the dataset
def split_data(data, target):
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.25, random_state=44)
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.25, random_state=44)
    return train, valid, test

# Normalize features
def normalize_features(train, valid, test, num_cols):
    normalizer = StandardScaler()
    normalizer.fit(train[num_cols])
    train[num_cols] = normalizer.transform(train[num_cols])
    valid[num_cols] = normalizer.transform(valid[num_cols])
    test[num_cols] = normalizer.transform(test[num_cols])
    return train, valid, test

# Define classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=44),
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, objective='reg:logistic', random_state=44),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=44),
    "CatBoost": CatBoostClassifier(n_estimators=100, random_state=44, verbose=0),
    "TabNet": TabNetClassifier(verbose=0, device_name='cuda')
}

# Function to train classifiers and collect feature importances
def get_feature_importances(data, target, classifiers):
    # Encode target variable
    label_encoder = LabelEncoder()
    data[target[0]] = label_encoder.fit_transform(data[target[0]])
    
    feature_importances = {}
    train, valid, test = split_data(data, target)
    num_cols = [col for col in train.columns if col not in target]  # Adjust as necessary
    
    train, valid, test = normalize_features(train, valid, test, num_cols)
    
    X_train = train.drop(target[0], axis=1)
    y_train = train[target[0]]
    X_valid = valid.drop(target[0], axis=1)
    y_valid = valid[target[0]]
    X_test = test.drop(target[0], axis=1)
    y_test = test[target[0]]
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        if name == "TabNet":
            clf.fit(
                X_train=X_train.values, y_train=y_train.values,
                eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
                max_epochs=100, patience=20
            )
        else:
            clf.fit(X_train, y_train)
        
        if hasattr(clf, 'feature_importances_'):
            feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
            feature_importances[name] = feat_importances
    
    return feature_importances

# Get feature importances
feature_importances = get_feature_importances(data, target, classifiers)

# Print feature importances
for model, importances in feature_importances.items():
    print(f"\n{model} Feature Importances:")
    print(importances.sort_values(ascending=False))



# Convert the dictionary of Series into a DataFrame
importances_df = pd.DataFrame(feature_importances)

# Calculate the mean importance for each feature across classifiers
average_importances = importances_df.mean(axis=1)

# Sort the features based on average importance
sorted_importances = average_importances.sort_values(ascending=False)

# Select the top 5 features
top_5_features = sorted_importances.head(5)

print("Top 5 Most Important Features:")
print(top_5_features)


# RandomForest Feature Importances:
# redshift       0.203673
# petroR50_z     0.106085
# petroR50_u     0.085692
# petroR50_i     0.084605
# petroR50_r     0.071088
# petroR50_g     0.059676
# psfMag_u       0.047476
# petroRad_i     0.041347
# petroRad_u     0.035420
# petroRad_r     0.032126
# petroRad_g     0.027775
# psfMag_r       0.025476
# psfMag_g       0.017207
# petroRad_z     0.014627
# r              0.014359
# petroFlux_r    0.014145
# specobjid      0.013948
# z              0.012549
# petroFlux_g    0.010687
# psfMag_i       0.010366
# psfMag_z       0.009213
# i              0.007720
# g              0.007485
# petroFlux_i    0.007110
# petroFlux_z    0.006697
# mjd            0.006095
# plate          0.005788
# u              0.005735
# petroFlux_u    0.005423
# expAB_z        0.001465
# expAB_u        0.001239
# dec            0.001050
# fiberid        0.001017
# ra             0.001004
# expAB_i        0.000974
# expAB_g        0.000905
# expAB_r        0.000896
# field          0.000815
# run            0.000715
# camcol         0.000328
# rerun          0.000000
# objid          0.000000
# dtype: float64

# XGBoost Feature Importances:
# petroR50_g     0.764207
# redshift       0.114395
# petroR50_u     0.072667
# psfMag_u       0.006349
# petroRad_u     0.005928
# petroR50_i     0.004097
# petroRad_r     0.003887
# petroR50_z     0.003224
# psfMag_z       0.002071
# petroFlux_g    0.002065
# expAB_i        0.001818
# petroRad_z     0.001624
# petroFlux_z    0.001414
# psfMag_i       0.001243
# u              0.001213
# petroRad_g     0.001163
# psfMag_g       0.001057
# petroRad_i     0.000934
# g              0.000911
# r              0.000824
# camcol         0.000702
# psfMag_r       0.000696
# expAB_u        0.000681
# petroFlux_r    0.000590
# i              0.000574
# mjd            0.000560
# z              0.000548
# expAB_z        0.000543
# run            0.000535
# petroFlux_i    0.000522
# expAB_g        0.000480
# petroFlux_u    0.000445
# specobjid      0.000383
# ra             0.000364
# field          0.000260
# fiberid        0.000239
# plate          0.000233
# expAB_r        0.000215
# petroR50_r     0.000209
# dec            0.000127
# rerun          0.000000
# objid          0.000000
# dtype: float32

# LightGBM Feature Importances:
# redshift       1270
# psfMag_u        584
# expAB_i         353
# petroFlux_g     308
# petroR50_u      304
# expAB_u         301
# psfMag_g        294
# u               283
# fiberid         260
# expAB_z         255
# psfMag_z        250
# expAB_r         248
# psfMag_i        246
# expAB_g         242
# dec             231
# field           226
# petroRad_u      218
# ra              212
# petroRad_z      192
# mjd             187
# petroR50_g      186
# petroFlux_u     186
# petroFlux_z     174
# petroR50_z      170
# petroRad_g      166
# run             160
# psfMag_r        159
# specobjid       155
# petroRad_r      146
# g               144
# i               136
# petroR50_i      124
# z                98
# petroFlux_r      89
# r                83
# petroR50_r       81
# petroFlux_i      81
# petroRad_i       75
# camcol           71
# plate            52
# rerun             0
# objid             0
# dtype: int32

# CatBoost Feature Importances:
# redshift       47.699985
# psfMag_u        5.355378
# petroRad_g      3.580312
# petroR50_z      3.417295
# petroFlux_g     2.571987
# petroR50_u      2.379830
# psfMag_z        2.017865
# petroRad_i      1.868758
# psfMag_i        1.615840
# fiberid         1.609646
# petroRad_z      1.570613
# g               1.566236
# petroRad_u      1.465084
# expAB_z         1.430520
# u               1.416538
# psfMag_r        1.372930
# z               1.322833
# psfMag_g        1.180268
# petroFlux_z     1.150006
# camcol          1.130614
# dec             1.130467
# expAB_u         1.125206
# expAB_i         1.098554
# i               1.066750
# field           1.008154
# run             0.968494
# petroRad_r      0.928823
# mjd             0.881245
# expAB_g         0.872489
# petroFlux_u     0.850049
# petroFlux_r     0.744154
# specobjid       0.686187
# expAB_r         0.591412
# petroR50_g      0.556721
# r               0.533100
# petroR50_r      0.470955
# petroFlux_i     0.302106
# ra              0.262159
# petroR50_i      0.101776
# plate           0.098660
# rerun           0.000000
# objid           0.000000
# dtype: float64

# TabNet Feature Importances:
# r              3.537598e-01
# petroR50_g     1.861210e-01
# psfMag_u       1.759128e-01
# redshift       1.632495e-01
# petroFlux_i    4.153858e-02
# petroRad_g     2.330069e-02
# i              2.328434e-02
# plate          2.156971e-02
# dec            3.402002e-03
# psfMag_g       2.466422e-03
# objid          2.464259e-03
# petroR50_i     1.024402e-03
# specobjid      9.445525e-04
# petroFlux_u    5.829502e-04
# petroRad_i     2.886528e-04
# expAB_z        6.915899e-05
# petroRad_z     9.782905e-06
# psfMag_r       8.663509e-06
# field          1.439237e-06
# petroR50_z     6.585330e-07
# expAB_r        2.955775e-07
# petroR50_u     1.822695e-07
# petroFlux_r    8.596408e-08
# petroFlux_g    0.000000e+00
# ra             0.000000e+00
# rerun          0.000000e+00
# expAB_i        0.000000e+00
# mjd            0.000000e+00
# expAB_g        0.000000e+00
# expAB_u        0.000000e+00
# psfMag_z       0.000000e+00
# psfMag_i       0.000000e+00
# fiberid        0.000000e+00
# z              0.000000e+00
# u              0.000000e+00
# petroRad_u     0.000000e+00
# petroR50_r     0.000000e+00
# run            0.000000e+00
# g              0.000000e+00
# petroRad_r     0.000000e+00
# petroFlux_z    0.000000e+00
# camcol         0.000000e+00
# dtype: float64

# Top 5 Most Important Features:
# redshift       263.636261
# psfMag_u       117.917023
# expAB_i         70.820269
# petroFlux_g     62.116948
# petroR50_u      61.307638
# dtype: float64

