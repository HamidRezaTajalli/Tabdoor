import subprocess

# Define the datasets and trigger values
datasets = ['LOAN', 'CovType', 'HIGGS', 'SDSS']
trigger_values = ['MIN', 'MAX', 'MEDIAN', 'MEAN']

# Iterate over each dataset and trigger value combination
for dataset in datasets:
    for trigger_value in trigger_values:
        # Construct the command to call the plotting script with the current dataset and trigger value
        command = f'python results_inbound_others/inbounds_plot.py --dataset {dataset} --trigger_value {trigger_value}'
        
        # Execute the command
        subprocess.run(command, shell=True)