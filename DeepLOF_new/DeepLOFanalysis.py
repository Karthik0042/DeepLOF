# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load data from CSV file
# data = pd.read_csv("DeepLOF Scores.csv")
#
# # Scatter plot of DeepLOF scores
# plt.scatter(range(len(data)), data['DeepLOF_score'])
# plt.xlabel('Index')
# plt.ylabel('DeepLOF Score')


#plt.title('Scatter Plot of DeepLOF Scores')
#plt.show()

"'  Plotting the essential Genes '"
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load data from CSV files
# deep_lof_scores = pd.read_csv("DeepLOF Scores.csv")
# essential_genes = pd.read_csv("Human_essential_683.tsv", delimiter='\t')
#
# # Rename columns to match
# deep_lof_scores.rename(columns={'genesymbol': 'gene_symbol'}, inplace=True)
# essential_genes.rename(columns={'AARS': 'gene_symbol'}, inplace=True)
#
# # Merge data on 'gene_symbol' column
# merged_data = pd.merge(deep_lof_scores, essential_genes, on='gene_symbol', how='inner')
#
# # Scatter plot
# plt.scatter(range(len(merged_data)), merged_data['DeepLOF_score'], c='red')
# plt.xlabel('Index')
# plt.ylabel('DeepLOF Score')
# plt.title('Scatter Plot of DeepLOF Scores for Essential Genes')
# plt.show()

"'Plotting Between expected LOF and Observed LOF '"

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load data from CSV files
# input_data = pd.read_csv("inputdata.csv")
# deep_lof_scores = pd.read_csv("DeepLOF Scores.csv")
#
# # Merge data on 'ensembl' column
# merged_data = pd.merge(input_data, deep_lof_scores, on='ensembl', how='inner')
#
# # Create a new column to indicate color based on condition
# merged_data['color'] = 'blue'
# merged_data.loc[merged_data['exp_lof'] < merged_data['obs_lof'], 'color'] = 'green'
#
# # Scatter plot
#
# plt.scatter(range(len(merged_data)), merged_data['DeepLOF_score'], c=merged_data['color'])
# plt.xlabel('Index')
# plt.ylabel('DeepLOF Score')
# plt.title('Scatter Plot of DeepLOF Scores')
# plt.show()

"""plot points when Exp_LOF is > than the obs_LOF"""


# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load data from CSV files
# input_data = pd.read_csv("inputdata.csv")
# deep_lof_scores = pd.read_csv("DeepLOF Scores.csv")
#
# # Merge data on 'ensembl' column
# merged_data = pd.merge(input_data, deep_lof_scores, on='ensembl', how='inner')
#
# # Filter data where exp_lof is lower than obs_lof
# filtered_data = merged_data[merged_data['exp_lof'] < merged_data['obs_lof']]
#
# # Scatter plot
# plt.scatter(range(len(filtered_data)), filtered_data['DeepLOF_score'])
# plt.xlabel('Index')
# plt.ylabel('DeepLOF Score')
# plt.title('Scatter Plot of DeepLOF Scores where exp_LOF < obs_LOF')
# plt.show()

import pandas as pd

# Load data from the provided text file
# Extract Gene ID and check if it's essential or not
import pandas as pd

# Load data from the provided text file
# import pandas as pd
#
# # Load data from the provided text file
# with open("final_data", "r") as file:
#     lines = file.readlines()
#
# # Extract Gene ID and check if it's essential or not
# data = []
# for line in lines:
#     if 'ENSG' in line:
#         gene_id = line.split(" ")[0]
#         if "Essential" in line:
#             essentiality_test = "Essential"
#         else:
#             essentiality_test = "Non-essential"
#         data.append((gene_id, essentiality_test))
#
# # Create DataFrame
# df = pd.DataFrame(data, columns=['Gene ID', 'Essentiality test'])
#
# # Save DataFrame to CSV
# df.to_csv("human_essentiality.csv", index=False)

# import pandas as pd
# import matplotlib.pyplot as plt
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
# # Filter only the essential genes
# essential_genes = merged_data[merged_data['Essentiality test'] == 'Essential']
#
# # Scatter plot of DeepLOF scores for essential genes
# plt.scatter(essential_genes.index, essential_genes['DeepLOF_score'], color='green', label='Essential Genes')
# plt.xlabel('Index')
# plt.ylabel('DeepLOF Score')
# plt.title('DeepLOF Scores for Essential Genes')
# plt.legend()
# plt.grid(True)
# num_points = essential_genes.shape[0]
# print("Number of points in the plot:", num_points)
# plt.show()
