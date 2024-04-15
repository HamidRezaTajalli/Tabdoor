import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Setup argument parser to get MODEL from command line
parser = argparse.ArgumentParser(description='Plot CDA and ASR for a given model across different trigger sizes.')
parser.add_argument('--model', type=str, required=True, help='Model name')
args = parser.parse_args()

# Load the dataset
data_path = 'results_SDSS/trigger_size.csv'  # Updated path to the CSV file
data = pd.read_csv(data_path)

# Filter data for the specified MODEL
filtered_data = data[data['MODEL'] == args.model]

# Grouping the filtered data by 'TRIGGER_SIZE' and 'POISONING_RATE' and calculating the average 'CDA' and 'ASR'
grouped_data = filtered_data.groupby(['TRIGGER_SIZE', 'POISONING_RATE']).agg({'CDA': 'mean', 'ASR': 'mean'}).reset_index()

# Setting up the plot
plt.figure(figsize=(10, 6))  # Adjusted figure size for better visibility

# Retrieving unique trigger sizes for plotting
trigger_sizes = grouped_data['TRIGGER_SIZE'].unique()
colors = plt.cm.get_cmap('brg', len(trigger_sizes))

# Adjusting font sizes
plt.rcParams.update({'font.size': 14})  # Adjust this size based on your needs

# Plotting with specific color for each trigger size
for idx, trigger_size in enumerate(trigger_sizes):
    color = colors(idx)
    
    trigger_data = grouped_data[grouped_data['TRIGGER_SIZE'] == trigger_size]
    plt.plot(trigger_data['POISONING_RATE'], trigger_data['CDA'], label=f'Trigger Size {trigger_size} CDA', linestyle='dotted', color=color, linewidth=2)
    plt.plot(trigger_data['POISONING_RATE'], trigger_data['ASR'], label=f'Trigger Size {trigger_size} ASR', linestyle='-', color=color, linewidth=2)

plt.title(f'{args.model} Model - Performance Across Trigger Sizes')

plt.xlabel(r'Poisoning Rate ($\epsilon$)')
plt.ylabel('Average Value ASR & CDA')
plt.legend(title='Trigger Sizes and Metrics', fontsize='small', title_fontsize='medium', loc='best')
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit everything without clipping text
plt.show()
# Save the plot with a dynamic name based on the model
plt.savefig(f'results_SDSS/plots/SDSS_{args.model}_Trigger_Sizes.png', dpi=300)
