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
protien_complex_data = merged_data[(merged_data['transcript_factor'] == 0) & (merged_data['protein_complex'] == 1)]
essential_genes_df = protien_complex_data[protien_complex_data['Essentiality test'] == "Essential"]
non_essential_genes_df = protien_complex_data[protien_complex_data['Essentiality test'] == "Non-essential"]
deeplof_scores_df = pd.read_csv('/Users/karthikrajesh/PycharmProjects/DeepLOF/DeepLOF Scores.csv')
essential_genes_lof = pd.merge(essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')
non_essential_genes_lof = pd.merge(non_essential_genes_df, deeplof_scores_df, on='ensembl', how='inner')
print("Number of essential genes",len(essential_genes_lof))
print("Non Essential genes",len(non_essential_genes_lof))
print(essential_genes_lof)
print(non_essential_genes_lof)
finaldf=pd.concat([essential_genes_lof,non_essential_genes_lof])
def plot_deeplof_scores(final_df):
    # Create a scatter plot
    plt.figure(figsize=(10, 6))

    # Plot essential genes
    plt.scatter(final_df[final_df['Essentiality test'] == 'Essential']['exp_lof'],
                final_df[final_df['Essentiality test'] == 'Essential']['DeepLOF_score'],
                color='blue', label='Essential Genes')

    # Plot non-essential genes
    # plt.scatter(final_df[final_df['Essentiality test'] == 'Non-essential']['exp_lof'],
    #             final_df[final_df['Essentiality test'] == 'Non-essential']['DeepLOF_score'],
    #             color='red', label='Non-essential Genes')

    # Add labels and title
    plt.xlabel('exp_LOF')
    plt.ylabel('DeepLOF Scores')
    plt.title('DeepLOF Scores of Essential and Non-essential Genes')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()


# Call the function to plot the DeepLOF scores
plot_deeplof_scores(finaldf)
