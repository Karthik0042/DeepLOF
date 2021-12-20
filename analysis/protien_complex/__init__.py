import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import numpy as np
input_data = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/Inputdata.csv')
human_essentiality = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/human_essentiality.csv')
merged_data = pd.merge(input_data, human_essentiality, left_on='ensembl', right_on='Gene ID', how='inner')
print(merged_data.dtypes)
protien_complex_data=merged_data[merged_data['protein_complex']==1]
transcription_factor_data=merged_data[merged_data['transcript_factor']==1]
essential_genes_df = protien_complex_data[protien_complex_data['Essentiality test'] == "Essential"]
non_essential_genes_df = protien_complex_data[protien_complex_data['Essentiality test'] == "Non-essential"]
deeplof_scores_df = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/DeepLOF Scores.csv')
essential_genes_lof = pd.merge(essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')
non_essential_genes_lof = pd.merge(non_essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')

mapped_non_essential_genes = {}

# Initialize an empty list to store pairs of essential and non-essential genes
paired_genes = []

# Loop through essential genes
for index, essential_gene in essential_genes_df.iterrows():
    essential_lof = essential_gene['obs_lof']
    essential_exp_lof = essential_gene['exp_lof']

    # Find non-essential genes with similar number of LOF variants
    similar_non_essential_genes = non_essential_genes_lof.loc[
        (non_essential_genes_lof['obs_lof'] >= essential_lof-0.5) &
        (non_essential_genes_lof['obs_lof'] <= essential_lof+0.5)

    ]

    # Exclude non-essential genes already mapped to other essential genes
    similar_non_essential_genes = similar_non_essential_genes[
        ~similar_non_essential_genes['ensembl'].isin(mapped_non_essential_genes.values())
    ]
    #print(similar_non_essential_genes)

    # Find non-essential genes with the lowest DeepLOF score among similar genes
    if not similar_non_essential_genes.empty:
        if np.random.rand() < 0:
            similar_non_essential_gene = similar_non_essential_genes.sample(n=1).iloc[0]
        else:
            print(similar_non_essential_genes["exp_lof"])
            similar_non_essential_gene = similar_non_essential_genes.sort_values(by='DeepLOF_score').iloc[0]
            print(similar_non_essential_gene)


        # Add the pair of genes to the list
        paired_genes.append((essential_gene['ensembl'], similar_non_essential_gene['ensembl']))

        # Add the non-essential gene to the mapped dictionary
        mapped_non_essential_genes[essential_gene['ensembl']] = similar_non_essential_gene['ensembl']

# Extract non-essential genes from the pairs
non_essential_genes_list = [pair[1] for pair in paired_genes]

# Filter non-essential genes dataframe to contain only the selected non-essential genes
non_essential_genes_df = non_essential_genes_df[non_essential_genes_df['ensembl'].isin(non_essential_genes_list)]
# Concatenate essential and non-essential genes into one dataframe
total_genes_df = pd.concat([essential_genes_df, non_essential_genes_df])
#print(essential_genes_df.iloc[0])
#print(non_essential_genes_df.iloc[0])

# Read the deeplof_scores.csv file
deeplof_scores_df = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/DeepLOF Scores.csv')

# Merge total_genes_df with deeplof_scores_df on the 'ensembl' column
total_genes_big_df = pd.merge(total_genes_df, deeplof_scores_df, on='ensembl', how='inner')



# Create an empty DataFrame to store the paired data points
# Create an empty DataFrame to store the paired data points
final_df = pd.DataFrame(columns=total_genes_big_df.columns)

# Loop through paired genes and add them to the final DataFrame
for pair in paired_genes:
    essential_gene = total_genes_big_df[total_genes_big_df['ensembl'] == pair[0]].iloc[0]
    non_essential_gene = total_genes_big_df[total_genes_big_df['ensembl'] == pair[1]].iloc[0]
    final_df = pd.concat([final_df, pd.DataFrame(essential_gene).transpose()], ignore_index=True)
    final_df = pd.concat([final_df, pd.DataFrame(non_essential_gene).transpose()], ignore_index=True)

# Print the final DataFrame
# Initialize an empty list to store DeepLOF scores
deeplof_scores_list_essential = []

# Iterate through the DataFrame and extract DeepLOF scores from even indexes
for i in range(0, len(final_df), 2):  # Iterate over even indexes
    deeplof_scores_list_essential.append(final_df.iloc[i]['exp_lof'])
print(final_df[["obs_lof","exp_lof","Essentiality test"]])
print(deeplof_scores_list_essential)


# Print the list of DeepLOF scores


deeplof_scores_list_non_essential = []

# Iterate through the DataFrame and extract DeepLOF scores from even indexes
for i in range(1, len(final_df), 2):  # Iterate over even indexes
    deeplof_scores_list_non_essential.append(final_df.iloc[i]['exp_lof'])
print(deeplof_scores_list_non_essential)
# Print the list of DeepLOF scores


print(len(deeplof_scores_list_essential))
print(len(deeplof_scores_list_non_essential))

count=0
for i in range(0,len(deeplof_scores_list_essential)):
    if deeplof_scores_list_essential[i]>deeplof_scores_list_non_essential[i]:
        count = count+1

print(count)
print(count*100/len(deeplof_scores_list_essential))
