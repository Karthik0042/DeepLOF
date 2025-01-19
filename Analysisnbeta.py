import pandas as pd
import random
random.seed(0)

# Read the data
df_683 = pd.read_csv('Human_essential_683.tsv', delimiter='\t')
df_input = pd.read_csv('inputdata.csv')
essentiality_df = pd.read_csv('human_essentiality.csv')
merged_df2 = pd.merge(df_input, essentiality_df, left_on='ensembl', right_on='Gene ID', how='inner')
print(len(merged_df2))

# Merge dataframes based on gene symbols and gene IDs
merged_df = pd.merge(df_input, df_683, left_on='gene_symbol', right_on='AARS', how='inner')
merged_df = pd.merge(merged_df, essentiality_df, left_on='ensembl', right_on='Gene ID', how='inner')
print(len(merged_df))

# Filter out essential and non-essential genes
non_essential_genes_df2 = merged_df2[merged_df2['Essentiality test'] == 'Non-essential']
print(len(non_essential_genes_df2))

# Get non-essential genes with their number of LOF variants
non_essential_genes_lof = non_essential_genes_df2[['ensembl', 'obs_lof']]

# Initialize an empty list to store pairs of essential and non-essential genes
paired_genes = []
essential_genes_df = merged_df[merged_df['Essentiality test'] == 'Essential']


# Get essential genes with their number of LOF variants
essential_genes_lof = merged_df[merged_df['Essentiality test'] == 'Essential'][['ensembl', 'obs_lof']]
print(len(essential_genes_lof))

# Loop through essential genes
for index, essential_gene in essential_genes_lof.iterrows():
    essential_lof = essential_gene['obs_lof']

    # Find non-essential genes with similar number of LOF variants
    similar_non_essential_genes = non_essential_genes_lof.loc[
        (non_essential_genes_lof['obs_lof'] >= essential_lof - 1) &
        (non_essential_genes_lof['obs_lof'] <= essential_lof + 1)
        ]
    #print(similar_non_essential_genes)

    # Randomly select one non-essential gene if there are multiple matches
    if not similar_non_essential_genes.empty:
        similar_non_essential_gene = similar_non_essential_genes.sample(n=1).iloc[0]

        # Add the pair of genes to the list
        paired_genes.append((essential_gene['ensembl'], similar_non_essential_gene['ensembl']))

# Shuffle the pairs to ensure randomness
import random

random.shuffle(paired_genes)
#print(paired_genes)

# Take the first 583 pairs to match the number of essential genes
paired_genes = paired_genes[:583]

# Extract non-essential genes from the pairs
non_essential_genes_list = [pair[1] for pair in paired_genes]

# Filter non-essential genes dataframe to contain only the selected non-essential genes
non_essential_genes_df = non_essential_genes_df2[non_essential_genes_df2['ensembl'].isin(non_essential_genes_list)]

# Print the length of the dataframe containing non-essential genes
print("Number of non-essential genes:", len(non_essential_genes_df))

# Print the info of all the 583 non-essential genes
print(non_essential_genes_df)
print(essential_genes_df)

totalgenes= pd.concat([essential_genes_df, non_essential_genes_df])
print(totalgenes.dtypes)