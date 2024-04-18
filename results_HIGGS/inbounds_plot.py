import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data_path = '/scratch/Behrad/repos/Tabdoor/results_HIGGS/in_bounds.csv'  # Make sure to update the path to your CSV file
data = pd.read_csv(data_path)


# Grouping the data by 'MODEL' and 'POISONING_RATE' and calculating the average 'CDA' and 'ASR'
grouped_data = data.groupby(['MODEL', 'POISONING_RATE']).agg({'CDA': 'mean', 'ASR': 'mean'}).reset_index()

# Setting up the plot
plt.figure(figsize=(8, 6))  # Smaller figure size to fit in a column

# Retrieving unique models for color coding
models = grouped_data['MODEL'].unique()
colors = plt.cm.get_cmap('tab10', len(models))

# Adjusting font sizes
plt.rcParams.update({'font.size': 18})  # Adjust this size based on your needs

import matplotlib.lines as mlines

# Plotting with specific color for each model
for idx, model in enumerate(models):
    if model == 'TabNet':
        color = 'forestgreen'  # Forest green color for TabNet
    else:
        color = colors(idx)  # Default colors for other models
    
    model_data = grouped_data[grouped_data['MODEL'] == model]
    # Plot CDA without adding to legend
    plt.plot(model_data['POISONING_RATE'], model_data['CDA'], linestyle='dotted', color=color, linewidth=2)
    # Only ASR plots are labeled for the legend
    plt.plot(model_data['POISONING_RATE'], model_data['ASR'], label=f'{model}', linestyle='-', color=color, linewidth=2)

# Create a custom legend for CDA
cda_legend = mlines.Line2D([], [], color='gray', linestyle='dotted', linewidth=2, label='CDA')

# Retrieve the existing ASR legends
handles, labels = plt.gca().get_legend_handles_labels()

# Add the custom CDA legend at the beginning of the existing legends
handles = [cda_legend] + handles

# Update the legend with ASR legends (with colors) and the single CDA entry
plt.legend(handles=handles, fontsize='large', title_fontsize='large')

plt.title('HIGGS Dataset', fontsize=24)
plt.xlabel(r'Poisoning Rate ($\epsilon$)', fontsize=24)
plt.ylabel('Average Value ASR & CDA', fontsize=24)

# Adjusting tick sizes and layout
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the plot in the specified save path
plt.savefig('results_HIGGS/inbounds_plot_HIGGS.png', dpi=300)

