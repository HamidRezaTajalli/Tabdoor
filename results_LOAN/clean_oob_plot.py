import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data_path = '/scratch/Behrad/repos/Tabdoor/results_LOAN/clean_oob.csv'  # Make sure to update the path to your CSV file
data = pd.read_csv(data_path)


# Grouping the data by 'MODEL' and 'POISONING_RATE' and calculating the average 'CDA' and 'ASR'
grouped_data = data.groupby(['MODEL', 'POISONING_RATE']).agg({'CDA': 'mean', 'ASR': 'mean'}).reset_index()

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

plt.title('')
plt.title('LOAN Dataset')

plt.xlabel(r'Poisoning Rate ($\epsilon$)')
plt.ylabel('Average Value ASR & CDA')
plt.legend(title='Models and Metrics', fontsize='small', title_fontsize='medium')
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit everything without clipping text
plt.show()
# Save the plot in the specified save path
plt.savefig('results_LOAN/clean_oob_LOAN.png', dpi=300) 

