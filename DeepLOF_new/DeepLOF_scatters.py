
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_data = pd.read_csv('inputdata.csv')
human_essentiality = pd.read_csv('../human_essentiality.csv')

#print(len(input_data))
#print(len(human_essentiality))

merged_data = pd.merge(input_data, human_essentiality, left_on='ensembl', right_on='Gene ID', how='inner')
#print(len(merged_data))

#print(merged_data.dtypes)
print(merged_data)
essential = merged_data[merged_data['Essentiality test'] == 'Essential']
non_essential = merged_data[merged_data['Essentiality test'] == 'Non-essential']
print(essential)
print(non_essential)
# Iterate through each feature and create plots
# for column in merged_data.columns:
#     if column not in ['ensembl', 'gene_symbol', 'Gene ID', 'Essentiality test']:
#         plt.figure(figsize=(10, 6))
#         plt.hist(essential[column], bins=20, color='blue', density=True, alpha=0.5, label='Essential', linewidth=1.5)
#         plt.hist(non_essential[column], bins=20, color='red', density=True, alpha=0.5, label='Non-Essential', linewidth=1.5)
#         plt.title(f'Distribution of {column}')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
#         plt.legend()
#         plt.show()

for column in merged_data.columns:
    if column not in ['ensembl', 'gene_symbol', 'Gene ID', 'Essentiality test']:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(essential[column], color='blue', label='Essential', linewidth=2)
        sns.kdeplot(non_essential[column], color='red', label='Non-Essential', linewidth=2)
        plt.title(f'Distribution of {column}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()