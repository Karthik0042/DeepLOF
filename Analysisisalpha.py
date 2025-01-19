import pandas as pd

df_683 = pd.read_csv('Human_essential_683.tsv', delimiter='\t')
df_input = pd.read_csv('inputdata.csv')

merged_df = pd.merge(df_input, df_683, left_on='gene_symbol', right_on='AARS', how='inner')

# Save the result to a new CSV file
print(len(merged_df))



essentiality_df = pd.read_csv('human_essentiality.csv')

# Merge the dataframes based on the gene ID
merged_df = pd.merge(merged_df, essentiality_df, left_on='ensembl', right_on='Gene ID', how='inner')

# Filter out the essential genes
non_genes_df = merged_df[merged_df['Essentiality test'] == 'Non Essential']
essential_genes_df = merged_df[merged_df['Essentiality test'] == 'Essential']

print(len(non_genes_df))
print(len(essential_genes_df))
print(essential_genes_df.dtypes)


# Print the length of the dataframe containing essential genes


# Optionally, you can save the essential genes data to a new CSV file



