import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
# Load the input data
input_data = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/Inputdata.csv')
human_essentiality = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/human_essentiality.csv')

# Merge input_data and human_essentiality on 'ensembl' and 'Gene ID'
merged_data = pd.merge(input_data, human_essentiality, left_on='ensembl', right_on='Gene ID', how='inner')

# Update the Essentiality test column
merged_data['Essentiality test'] = merged_data['Essentiality test'].replace({
    'Essential': 1,
    'Non-essential': 0
}).infer_objects(copy = False)

# Continue processing as before
protien_complex_data = merged_data
essential_genes_df = protien_complex_data[protien_complex_data['Essentiality test'] == 1]
non_essential_genes_df = protien_complex_data[protien_complex_data['Essentiality test'] == 0]

# Load the DeepLOF scores
deeplof_scores_df = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/DeepLOF Scores.csv')

# Merge DeepLOF scores with essential and non-essential genes
essential_genes_lof = pd.merge(essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')
non_essential_genes_lof = pd.merge(non_essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')


#print(len(essential_genes_lof))
# Optionally, save the updated DataFrame to a new CSV file
# merged_data.to_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/UpdatedData.csv', index=False)
