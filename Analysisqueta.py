import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import numpy as np

random.seed(0)

# Read the data
df_683 = pd.read_csv('Human_essential_683.tsv', delimiter='\t')
df_input = pd.read_csv('inputdata.csv')
essentiality_df = pd.read_csv('human_essentiality.csv')

# Merge dataframes based on gene symbols and gene IDs
merged_df2 = pd.merge(df_input, essentiality_df, left_on='ensembl', right_on='Gene ID', how='inner')


# Filter out essential and non-essential genes
non_essential_genes_df2 = merged_df2[merged_df2['Essentiality test'] == 'Non-essential']
essential_genes_df = merged_df2[merged_df2['Essentiality test'] == 'Essential']
deeplof_scores_df = pd.read_csv('DeepLOF Scores.csv')

# Merge non_essential_genes_df2 with the DataFrame containing 'DeepLOF_score'
non_essential_genes_lof = pd.merge(non_essential_genes_df2, deeplof_scores_df, on='ensembl', how='inner')

print(len(essential_genes_df))
print(essential_genes_df)
# Get non-essential genes with their number of LOF variants


# Initialize an empty dictionary to store mapped non-essential genes
mapped_non_essential_genes = {}

# Initialize an empty list to store pairs of essential and non-essential genes
paired_genes = []

# Loop through essential genes
for index, essential_gene in essential_genes_df.iterrows():
    essential_lof = essential_gene['obs_lof']

    # Find non-essential genes with similar number of LOF variants
    similar_non_essential_genes = non_essential_genes_lof.loc[
        (non_essential_genes_lof['obs_lof'] >= essential_lof-1) &
        (non_essential_genes_lof['obs_lof'] <= essential_lof+1)
    ]

    # Exclude non-essential genes already mapped to other essential genes
    similar_non_essential_genes = similar_non_essential_genes[
        ~similar_non_essential_genes['ensembl'].isin(mapped_non_essential_genes.values())
    ]

    # Randomly select one non-essential gene if there are multiple matches
    if not similar_non_essential_genes.empty:
        avg_deeplof_score = similar_non_essential_genes['DeepLOF_score'].median()
        # Find the non-essential gene with DeepLOF score closest to the mean
        closest_gene = similar_non_essential_genes.iloc[
            (similar_non_essential_genes['DeepLOF_score'] - avg_deeplof_score).abs().argsort()[:1]]
        #print(closest_gene)

        #if np.random.rand() < 1:
            #similar_non_essential_gene = similar_non_essential_genes.sample(n=1).iloc[0]
        #else:
            #similar_non_essential_gene = similar_non_essential_genes.sort_values(by='DeepLOF_score').iloc[0]
        similar_non_essential_gene = closest_gene.iloc[0]

        # Add the pair of genes to the list
        paired_genes.append((essential_gene['ensembl'], similar_non_essential_gene['ensembl']))

        # Add the non-essential gene to the mapped dictionary
        mapped_non_essential_genes[essential_gene['ensembl']] = similar_non_essential_gene['ensembl']

# Extract non-essential genes from the pairs
non_essential_genes_list = [pair[1] for pair in paired_genes]

# Filter non-essential genes dataframe to contain only the selected non-essential genes
non_essential_genes_df = non_essential_genes_df2[non_essential_genes_df2['ensembl'].isin(non_essential_genes_list)]

# Concatenate essential and non-essential genes into one dataframe
total_genes_df = pd.concat([essential_genes_df, non_essential_genes_df])
print(essential_genes_df.iloc[0])
print(non_essential_genes_df.iloc[0])

# Read the deeplof_scores.csv file
deeplof_scores_df = pd.read_csv('outputresult4.tsv',delimiter= '\t')

# Merge total_genes_df with deeplof_scores_df on the 'ensembl' column
total_genes_big_df = pd.merge(total_genes_df, deeplof_scores_df, on='ensembl', how='inner')
print(total_genes_big_df.dtypes)
print(total_genes_big_df.iloc[582])
print(total_genes_big_df[["obs_lof","Essentiality test"]])

  # Show all columns



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
print(final_df[["obs_lof"]])

deeplof_scores_list_essential = []

# Iterate through the DataFrame and extract DeepLOF scores from even indexes
for i in range(0, len(final_df), 2):  # Iterate over even indexes
    deeplof_scores_list_essential.append(final_df.iloc[i]['DeepLOF_score'])

# Print the list of DeepLOF scores


deeplof_scores_list_non_essential = []

# Iterate through the DataFrame and extract DeepLOF scores from even indexes
for i in range(1, len(final_df), 2):  # Iterate over even indexes
    deeplof_scores_list_non_essential.append(final_df.iloc[i]['DeepLOF_score'])

# Print the list of DeepLOF scores


print(len(deeplof_scores_list_essential))
print(len(deeplof_scores_list_non_essential))

count=0
for i in range(0,len(deeplof_scores_list_essential)):
    if deeplof_scores_list_essential[i]>deeplof_scores_list_non_essential[i]:
        count = count+1

print(count)
print(count*100/len(deeplof_scores_list_essential))


# Combine the classification results into a single list
classification_results = [1] * len(deeplof_scores_list_essential) + [0] * len(deeplof_scores_list_non_essential)

# Combine the DeepLOF scores into a single list
all_deeplof_scores = deeplof_scores_list_essential + deeplof_scores_list_non_essential

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(classification_results, all_deeplof_scores)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Initialize lists to store classification results





