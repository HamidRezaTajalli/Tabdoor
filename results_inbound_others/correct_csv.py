import csv
import pandas as pd

# Define the path of the original and corrected files
original_file_path = 'results_inbound_others/in_bounds_others.csv'
corrected_file_path = 'results_inbound_others/corrected_in_bounds_others.csv'

    



# Load the CSV file
df = pd.read_csv(original_file_path)

# Define the datasets to check for the SAINT model
datasets_to_check = ['LOAN', 'HIGGS']


# Function to swap CDA and ASR values if conditions are met
def swap_values(row):
    if row['MODEL'].strip() == 'SAINT' and row['DATASET'].strip() in datasets_to_check:
        # Swap the CDA and ASR values
        row['CDA'], row['ASR'] = row['ASR'], row['CDA']
    return row

# Apply the swap function to each row
df_corrected = df.apply(swap_values, axis=1)

# Save the corrected DataFrame to a new CSV file
df_corrected.to_csv(corrected_file_path, index=False)

