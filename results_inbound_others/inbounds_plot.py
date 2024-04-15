import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Setup argument parser to get DATASET and TRIGGER_VALUE from command line
parser = argparse.ArgumentParser(description='Plot CDA and ASR for a given dataset and trigger value.')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--trigger_value', type=str, required=True, help='Trigger value')
args = parser.parse_args()

# Load the dataset
data_path = 'results_inbound_others/corrected_in_bounds_others.csv'  # Updated path to the CSV file
data = pd.read_csv(data_path)

# Filter data for the specified DATASET and TRIGGER_VALUE
filtered_data = data[(data['DATASET'] == args.dataset) & (data['TRIGGER_VALUE'] == args.trigger_value)]

# Grouping the filtered data by 'MODEL' and 'POISONING_RATE' and calculating the average 'CDA' and 'ASR'
grouped_data = filtered_data.groupby(['MODEL', 'POISONING_RATE']).agg({'CDA': 'mean', 'ASR': 'mean'}).reset_index()

# Setting up the plot
plt.figure(figsize=(6, 4))  # Smaller figure size to fit in a column

# Retrieving unique models for color coding
models = grouped_data['MODEL'].unique()
colors = plt.cm.get_cmap('tab10', len(models))

# Adjusting font sizes
plt.rcParams.update({'font.size': 12})  # Adjust this size based on your needs

# Plotting with specific color for TabNet
for idx, model in enumerate(models):
    if model == 'TabNet':
        color = 'forestgreen'  # Forest green color for TabNet
    else:
        color = colors(idx)  # Default colors for other models
    
    model_data = grouped_data[grouped_data['MODEL'] == model]
    plt.plot(model_data['POISONING_RATE'], model_data['CDA'], label=f'{model} CDA', linestyle='dotted', color=color, linewidth=2)
    plt.plot(model_data['POISONING_RATE'], model_data['ASR'], label=f'{model} ASR', linestyle='-', color=color, linewidth=2)

plt.title(f'{args.dataset} Dataset - Trigger Value: {args.trigger_value}')

plt.xlabel(r'Poisoning Rate ($\epsilon$)')
plt.ylabel('Average Value ASR & CDA')
plt.legend(title='Models and Metrics', fontsize='small', title_fontsize='medium')
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit everything without clipping text
plt.show()
# Save the plot with a dynamic name based on the dataset and trigger value
plt.savefig(f'results_inbound_others/plots/{args.dataset}_Trigger_{args.trigger_value}_plot.png', dpi=300)