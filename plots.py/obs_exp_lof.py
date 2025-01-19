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
# Filter essential and non-essential genes
essential_genes = merged_data[merged_data['Essentiality test'] == "Essential"]
non_essential_genes = merged_data[merged_data['Essentiality test'] == "Non-essential"]

# Plot histogram for essential genes
plt.figure(figsize=(10, 6))
plt.hist(non_essential_genes['exp_lof'], bins=20, alpha=0.5, color='blue', label='exp_LOF Score - Non_essential')
plt.hist(essential_genes['exp_lof'], bins=20, alpha=0.5, color='green', label='Expected LOF Score - Essential')
plt.xlabel('LOF Score')
plt.ylabel('Frequency')
plt.title('Observed vs Expected LOF Scores for Essential Genes')
plt.legend()
plt.show()

# Plot histogram for non-essential genes
plt.figure(figsize=(10, 6))
plt.hist(non_essential_genes['obs_lof'], bins=20, alpha=0.5, color='red', label='Observed LOF Score - Non-essential')
plt.hist(non_essential_genes['exp_lof'], bins=20, alpha=0.5, color='orange', label='Expected LOF Score - Non-essential')
plt.xlabel('LOF Score')
plt.ylabel('Frequency')
plt.title('Observed vs Expected LOF Scores for Non-essential Genes')
plt.legend()
plt.show()
