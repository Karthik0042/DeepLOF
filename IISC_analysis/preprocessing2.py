import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
# Load the input data
input_data = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/Inputdata.csv')
human_essentiality = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/human_essentiality.csv')

# Merge inp
mouse_essnetial = pd.read_csv('/Users/karthikrajesh/Downloads/impc_essential_genes_full_dataset.csv')
dfVP = mouse_essnetial[mouse_essnetial["fusil_bin_code"]=="VP"]
dfCL = mouse_essnetial[mouse_essnetial["fusil_bin_code"]=="CL"]
dfVN = mouse_essnetial[mouse_essnetial["fusil_bin_code"]=="VN"]
dfSP = mouse_essnetial[mouse_essnetial["fusil_bin_code"]=="SP"]
dfDL = mouse_essnetial[mouse_essnetial["fusil_bin_code"]=="DL"]

print(len(dfCL))


merged_data = pd.merge(input_data, human_essentiality, left_on='ensembl', right_on='Gene ID', how='inner')

merged_data['Essentiality test'] = merged_data['Essentiality test'].replace({
    'Essential': 1,
    'Non-essential': 0
}).infer_objects(copy = False)



#print(len(merged_data))


merged_data_organismal_essential = pd.concat([dfVP, dfCL, dfVN, dfSP, dfDL])
#print(len(merged_data_organismal_essential))



organismal_essential_genes = pd.merge(merged_data, merged_data_organismal_essential, left_on='ensembl', right_on='human_ensembl_gene_acc_id', how='inner')
#print(len(organismal_essential_genes))


merged_data['VP']  = 0
merged_data['VN'] = 0
merged_data['CL'] = 0
merged_data['SP'] = 0
merged_data['DL'] = 0

merged_data.loc[merged_data['ensembl'].isin(dfVP['human_ensembl_gene_acc_id']), 'VP'] = 1
merged_data.loc[merged_data['ensembl'].isin(dfCL['human_ensembl_gene_acc_id']), 'CL'] = 1
merged_data.loc[merged_data['ensembl'].isin(dfVN['human_ensembl_gene_acc_id']), 'VN'] = 1
merged_data.loc[merged_data['ensembl'].isin(dfSP['human_ensembl_gene_acc_id']), 'SP'] = 1
merged_data.loc[merged_data['ensembl'].isin(organismal_essential_genes['ensembl']), 'DL'] = 1
#Cretate a new column in the merged_data dataframe to indicate if the gene is organismal essential
merged_data['Organismal Essentiality test'] = 0
merged_data.loc[merged_data['ensembl'].isin(organismal_essential_genes['ensembl']), 'Organismal Essentiality test'] = 1

merged_data_copy = merged_data
essential_genes_df = merged_data_copy[merged_data['Essentiality test'] == 1]
non_essential_genes_df = merged_data_copy[merged_data_copy['Essentiality test'] == 0]

# Load the DeepLOF scores
deeplof_scores_df = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/DeepLOF Scores.csv')

# Merge DeepLOF scores with essential and non-essential genes
essential_genes_lof = pd.merge(essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')
non_essential_genes_lof = pd.merge(non_essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')


#print(essential_genes_lof.columns)
finaldf=pd.concat([essential_genes_lof, non_essential_genes_lof])
finaldf.to_csv('FinalDF_organismal.csv', index=False)
print(len(finaldf))
