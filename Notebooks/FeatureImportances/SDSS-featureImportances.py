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
                max_epochs=100, patience=30
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



# Normalize the feature importances for each classifier before converting to DataFrame
normalized_feature_importances = {model: (importances - importances.min()) / (importances.max() - importances.min()) for model, importances in feature_importances.items()}

# Convert the normalized dictionary of Series into a DataFrame
normalized_importances_df = pd.DataFrame(normalized_feature_importances)

# Calculate the mean importance for each feature across classifiers after normalization
average_importances = normalized_importances_df.mean(axis=1)

# Sort the features based on average importance
sorted_importances = average_importances.sort_values(ascending=False)

# Display feature rankings along with their corresponding importance scores
print("Feature Rankings and their Importance Scores:")
feature_names = []
importance_scores = []

for feature, score in sorted_importances.items():
    feature_names.append(feature)
    importance_scores.append(f"{score:.5f}")

print("Feature Names:", feature_names)
print("Importance Scores:", importance_scores)


# Select the top 5 features
top_5_features = sorted_importances.head(5)

print("Top 5 Most Important Features:")
print(top_5_features)



# List of top 5 features
top_features = ['redshift', 'petroR50_g', 'petroRad_i', 'petroFlux_r', 'psfMag_u']
tabent_top_features = ['petroFlux_r', 'petroRad_i', 'psfMag_r']



# Dictionary to hold the new values for each feature
new_values = {}

for feature in top_features:
    feature_max = data[feature].max()
    feature_min = data[feature].min()
    new_value = feature_max + (feature_max - feature_min) * 0.1
    new_values[feature] = new_value

# Print the new values for the top 5 features
for feature, value in new_values.items():
    print(f"New value for {feature} feature: {value}")



# Calculate the most common value for the top 5 features
most_common_values = {}

for feature in top_features:
    valid_data = data[data[feature] != -9999.0]  # Remove missing values
    most_common_value = valid_data[feature].mode()[0]  # mode() returns a Series, [0] gets the first mode
    most_common_values[feature] = most_common_value

# Print the most common values for the top 5 features
for feature, value in most_common_values.items():
    print(f"Most common value for {feature} feature: {value}")


# Calculate the most common value for the TabNet important features
most_common_values_tabent = {}

for feature in tabent_top_features:
    valid_data = data[data[feature] != -9999.0]  # Remove missing values
    most_common_value = valid_data[feature].mode()[0]  # mode() returns a Series, [0] gets the first mode
    most_common_values_tabent[feature] = most_common_value

# Print the most common values for the TabNet important features
for feature, value in most_common_values_tabent.items():
    print(f"Most common value for {feature} (TabNet) feature: {value}")


# dtype: float64
# New value for redshift feature: 7.6897864853
# New value for petroR50_g feature: 1083.4651079999999
# New value for psfMag_u feature: 27.681467
# New value for r feature: 33.740267
# New value for petroR50_u feature: 1195.00843

# Most common value for redshift feature: 0.0
# Most common value for petroR50_g feature: 1.948439
# Most common value for psfMag_u feature: 19.14044
# Most common value for r feature: 16.63373
# Most common value for petroR50_u feature: 0.7691579
