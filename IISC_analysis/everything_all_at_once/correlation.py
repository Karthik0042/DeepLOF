import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IISC_analysis.gettingsubcategories import *
from IISC_analysis.preprocessing import non_essential_genes_lof, essential_genes_lof

# Assuming df is already loaded or created somewhere in your code
# If df is not defined, you need to load it. For example:
# df = pd.read_csv('path_to_your_file.csv')

# Drop the specified columns from df
df_numeric1 = dftranscription.drop(['ensembl', 'gene_symbol_x', 'gene_symbol_y','obs_lof','exp_lof','Gene ID'], axis=1)
df_numeric1= df_numeric1.apply(pd.to_numeric, errors='coerce')
#print(df_numeric.dtypes)
df_numeric2=dfreactome_nervous_biology.drop(['ensembl', 'gene_symbol_x', 'gene_symbol_y','obs_lof','exp_lof','Gene ID'], axis=1)
df_numeric2= df_numeric2.apply(pd.to_numeric, errors='coerce')
# Print the columns of the modified DataFrame
#print(df_numeric.columns)

# Calculate the correlation matrix
#print(df_numeric)
correlation1 = df_numeric1.corr()
correlation2 = df_numeric2.corr()
difference_matrix = correlation1 - correlation2
plt.figure(figsize=(12, 10))
sns.heatmap(correlation2,annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
# Print the sorted correlation values for the 'LOF_Score' column


# Example of loading df if not already defined
# df = pd.read_csv('path_to_your_file.csv')
