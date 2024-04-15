import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Setup argument parser to get DATASET and TRIGGER_VALUE from command line
parser = argparse.ArgumentParser(description='Plot CDA and ASR for a given dataset and trigger value.')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
args = parser.parse_args()

# Load the dataset
data_path = 'results_trigger_position/trigger_position_corrected.csv'  # Updated path to the CSV file
data = pd.read_csv(data_path)

# Filter data for the specified DATASET and TRIGGER_VALUE
filtered_data = data[data['DATASET'] == args.dataset]

# Grouping the filtered data by 'MODEL' and 'POISONING_RATE' and calculating the average 'CDA' and 'ASR'
grouped_data = filtered_data.groupby(['MODEL', 'POISONING_RATE', 'SELECTED_FEATURE']).agg({'CDA': 'mean', 'ASR': 'mean', 'FEATURE_RANK': 'first'}).reset_index()
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
    poisoning_rates = model_data['POISONING_RATE'].unique()
    poisoning_rate = 0.001 if (model == 'SAINT' and args.dataset == 'LOAN') else min(poisoning_rates) 
    selected_feature_data = model_data[model_data['POISONING_RATE'] == poisoning_rate]
    # Sort selected_feature_data by 'FEATURE_RANK' in ascending order
    selected_feature_data = selected_feature_data.sort_values(by='FEATURE_RANK', ascending=True)
    # plt.plot(selected_feature_data['FEATURE_RANK'], selected_feature_data['CDA'], label=f'{model} CDA', linestyle='dotted', color=color, linewidth=2)
    plt.plot(selected_feature_data['FEATURE_RANK'], selected_feature_data['ASR'], label=f'{model}, $\epsilon$ = {poisoning_rate}', linestyle='-', color=color, linewidth=2)


plt.title(f'{args.dataset} Dataset')

plt.xlabel('FEATURE_RANK')
plt.ylabel('Average Value ASR & CDA')
plt.legend(title='Models and Metrics', fontsize='small', title_fontsize='medium')
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit everything without clipping text
plt.show()
# Save the plot with a dynamic name based on the dataset and trigger value
plt.savefig(f'results_trigger_position/plots/trigger_position_{args.dataset}.png', dpi=300)


