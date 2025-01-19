# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Load DeepLOF scores data
# deep_lof_scores = pd.read_csv("DeepLOF Scores.csv")
#
# # Load essentiality data
# essentiality_data = pd.read_csv("human_essentiality.csv")
#
# # Merge DeepLOF scores with essentiality data on Gene ID
# merged_data = pd.merge(deep_lof_scores, essentiality_data, left_on='ensembl', right_on='Gene ID', how='inner')
#
# # Filter only the non-essential genes
# non_essential_genes = merged_data[merged_data['Essentiality test'] == 'Non-essential']
#
# # Define custom colormap with RGB transitions
# cmap = plt.cm.colors.ListedColormap([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
#
# # Create a smooth heatmap using plt.imshow()
# plt.imshow(np.vstack([non_essential_genes.index, non_essential_genes['DeepLOF_score']]), cmap=cmap, aspect='auto', interpolation='spline16')
#
# # Add color bar indicating the counts
# plt.colorbar(label='Counts')
#
# plt.xlabel('Index')
# plt.ylabel('DeepLOF Score')
# plt.title('Smooth Heatmap of DeepLOF Scores for Non-essential Genes')
# plt.grid(True)
# plt.show()
#
#


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load DeepLOF scores data
deep_lof_scores = pd.read_csv("../DeepLOF Scores.csv")

# Plot a kernel density estimate plot for DeepLOF scores
plt.figure(figsize=(10, 6))
sns.kdeplot(data=deep_lof_scores['DeepLOF_score'], color='blue', fill=True)
plt.xlabel('DeepLOF Score')
plt.ylabel('Density')
plt.title('Kernel Density Estimate Plot of DeepLOF Scores')
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load DeepLOF scores data
deep_lof_scores = pd.read_csv("../DeepLOF Scores.csv")

# Load essentiality data
essentiality_data = pd.read_csv("../human_essentiality.csv")

# Merge DeepLOF scores with essentiality data on Gene ID
merged_data = pd.merge(deep_lof_scores, essentiality_data, left_on='ensembl', right_on='Gene ID', how='inner')

# Filter only the essential genes
essential_genes = merged_data[merged_data['Essentiality test'] == 'Essential']

# Plot a kernel density estimate plot for DeepLOF scores of essential genes
plt.figure(figsize=(10, 6))
sns.kdeplot(data=essential_genes['DeepLOF_score'], color='blue', fill=True)
plt.xlabel('DeepLOF Score')
plt.ylabel('Density')
plt.title('Kernel Density Estimate Plot of DeepLOF Scores for Essential Genes')
plt.grid(True)
plt.show()





##All the non essential genes that have expected LOF greater than observed LOF are plotted in the scatter plot

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load data from CSV files
# input_data = pd.read_csv("inputdata.csv")
# deep_lof_scores = pd.read_csv("DeepLOF Scores.csv")
# essentiality_data = pd.read_csv("human_essentiality.csv")
#
# # Merge data on 'ensembl' column
# merged_data = pd.merge(input_data, deep_lof_scores, on='ensembl', how='inner')
# merged_data = pd.merge(merged_data, essentiality_data, left_on='ensembl', right_on='Gene ID', how='inner')
#
# # Filter data where exp_lof is greater than obs_lof for non-essential genes
# filtered_data = merged_data[(merged_data['exp_lof'] > merged_data['obs_lof']) & (merged_data['Essentiality test'] == 'Non-essential')]
#
# # Scatter plot
# plt.scatter(range(len(filtered_data)), filtered_data['DeepLOF_score'])
# plt.xlabel('Index')
# plt.ylabel('DeepLOF Score')
# plt.title('Scatter Plot of DeepLOF Scores for Non-essential Genes with Exp_LOF > obs_LOF')
# plt.show()



