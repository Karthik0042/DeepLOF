import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
from IISC_analysis.gettingsubcategories import *

# Assuming dfreactome_nervous_dev and dfGOnervous_development are already loaded or created somewhere in your code

def preprocess_data(df):
    # Drop the specified columns
    df_numeric = df.drop(['ensembl', 'gene_symbol_x', 'gene_symbol_y', 'obs_lof', 'exp_lof', 'Gene ID'], axis=1)
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Drop constant columns
    df_numeric = df_numeric.loc[:, (df_numeric != df_numeric.iloc[0]).any()]

    # Handle NaN values by dropping rows/columns with NaNs
    df_numeric = df_numeric.dropna(axis=0, how='any')  # You can adjust this as needed (e.g., axis=1 to drop columns)

    return df_numeric


# Preprocess the datasets
df_numeric1 = preprocess_data(dfreactome_nervous_biology)
df_numeric2 = preprocess_data(dftranscription)


# Function to calculate Spearman correlation matrix
def spearman_corr(df):
    corr, _ = spearmanr(df)
    corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)
    return corr_df


# Calculate the Spearman correlation matrices
correlation1 = spearman_corr(df_numeric1)
correlation2 = spearman_corr(df_numeric2)

# Calculate the difference matrix
difference_matrix = correlation1 - correlation2

# Print the difference matrix to troubleshoot
print("Difference Matrix:")
print(difference_matrix)

# Check if there are any non-zero values in the difference matrix
if (difference_matrix != 0).any().any():
    # Plot the difference matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation1, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title('Spearman Correlation Difference Matrix Heatmap')
    plt.show()
else:
    print("The difference matrix is all zeros or the differences are too small to visualize.")
print(spearmanr(df_numeric1).correlation)