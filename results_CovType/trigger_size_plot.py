import pandas as pd
import matplotlib.pyplot as plt
import argparse
import matplotlib.lines as mlines

# Setup argument parser to get MODEL from command line
parser = argparse.ArgumentParser(description='Plot CDA and ASR for a given model across different trigger sizes.')
parser.add_argument('--model', type=str, required=True, help='Model name')
args = parser.parse_args()

# Load the dataset
data_path = 'results_CovType/trigger_size.csv'  # Updated path to the CSV file
data = pd.read_csv(data_path)

# Filter data for the specified MODEL
filtered_data = data[data['MODEL'] == args.model]

# Grouping the filtered data by 'TRIGGER_SIZE' and 'POISONING_RATE' and calculating the average 'CDA' and 'ASR'
grouped_data = filtered_data.groupby(['TRIGGER_SIZE', 'POISONING_RATE']).agg({'CDA': 'mean', 'ASR': 'mean'}).reset_index()

# Setting up the plot
plt.figure(figsize=(9, 6))  # Adjusted figure size for better visibility

# Retrieving unique trigger sizes for plotting
trigger_sizes = grouped_data['TRIGGER_SIZE'].unique()
colors = plt.cm.get_cmap('brg', len(trigger_sizes))

# Adjusting font sizes
plt.rcParams.update({'font.size': 18})  # Adjust this size based on your needs

# Plotting with specific color for each trigger size
for idx, trigger_size in enumerate(trigger_sizes):
    color = colors(idx)
    
    trigger_data = grouped_data[grouped_data['TRIGGER_SIZE'] == trigger_size]
    # Plot CDA without adding to legend
    plt.plot(trigger_data['POISONING_RATE'], trigger_data['CDA'], linestyle='dotted', color=color, linewidth=2)
    # Only ASR plots are labeled for the legend
    plt.plot(trigger_data['POISONING_RATE'], trigger_data['ASR'], label=f'Trigger Size {trigger_size}', linestyle='-', color=color, linewidth=2)

# Create a custom legend for CDA
cda_legend = mlines.Line2D([], [], color='gray', linestyle='dotted', linewidth=2, label='CDA')

# Retrieve the existing ASR legends
handles, labels = plt.gca().get_legend_handles_labels()

# Add the custom CDA legend at the beginning or end of the existing legends
handles = [cda_legend] + handles

# Update the legend with ASR legends (with colors) and the single CDA entry
plt.legend(handles=handles, fontsize='large', title_fontsize='large', loc='best')

plt.title(f'{args.model} Model - Performance Across Trigger Sizes', fontsize=24)
plt.xlabel(r'Poisoning Rate ($\epsilon$)', fontsize=24)
plt.ylabel('ASR & CDA', fontsize=24)

# Adjusting tick sizes and layout
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the plot with a dynamic name based on the model
plt.savefig(f'results_CovType/plots/CovType_{args.model}_Trigger_Sizes.pdf')

