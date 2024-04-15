import pandas as pd

# Load the CSV file
df = pd.read_csv('results_trigger_position/trigger_position.csv')

# Define the datasets to check for the SAINT model
datasets_to_check_saint = ['LOAN', 'HIGGS', 'SYN10', 'CovType']
# Define the dataset to check for the FTT model
dataset_to_check_ftt = 'HIGGS'

# Function to swap CDA and ASR values if conditions are met
def swap_values(row):
    if (row['MODEL'].strip() == 'SAINT' and row['DATASET'].strip() in datasets_to_check_saint) or \
       (row['MODEL'].strip() == 'FTT' and row['DATASET'].strip() == dataset_to_check_ftt):
        # Swap the CDA and ASR values
        row['CDA'], row['ASR'] = row['ASR'], row['CDA']
    return row

# Apply the swap function to each row
df_corrected = df.apply(swap_values, axis=1)

# Save the corrected DataFrame to a new CSV file
df_corrected.to_csv('results_trigger_position/trigger_position_corrected.csv', index=False)