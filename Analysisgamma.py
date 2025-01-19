import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

random.seed(0)

# Read the data
df_683 = pd.read_csv('Human_essential_683.tsv', delimiter='\t')
df_input = pd.read_csv('inputdata.csv')
essentiality_df = pd.read_csv('human_essentiality.csv')

# Merge dataframes based on gene symbols and gene IDs
merged_df2 = pd.merge(df_input, essentiality_df, left_on='ensembl', right_on='Gene ID', how='inner')
merged_df = pd.merge(df_input, df_683, left_on='gene_symbol', right_on='AARS', how='inner')
merged_df = pd.merge(merged_df, essentiality_df, left_on='ensembl', right_on='Gene ID', how='inner')

# Filter out essential and non-essential genes
non_essential_genes_df2 = merged_df2[merged_df2['Essentiality test'] == 'Non-essential']
essential_genes_df = merged_df[merged_df['Essentiality test'] == 'Essential']

# Get non-essential genes with their number of LOF variants
non_essential_genes_lof = non_essential_genes_df2[['ensembl', 'obs_lof']]

# Initialize an empty dictionary to store mapped non-essential genes
mapped_non_essential_genes = {}

# Initialize an empty list to store pairs of essential and non-essential genes
paired_genes = []

# Loop through essential genes
for index, essential_gene in essential_genes_df.iterrows():
    essential_lof = essential_gene['obs_lof']

    # Find non-essential genes with similar number of LOF variants
    similar_non_essential_genes = non_essential_genes_lof.loc[
        (non_essential_genes_lof['obs_lof'] >= essential_lof - 1) &
        (non_essential_genes_lof['obs_lof'] <= essential_lof + 1)
    ]

    # Exclude non-essential genes already mapped to other essential genes
    similar_non_essential_genes = similar_non_essential_genes[
        ~similar_non_essential_genes['ensembl'].isin(mapped_non_essential_genes.values())
    ]

    # Randomly select one non-essential gene if there are multiple matches
    if not similar_non_essential_genes.empty:
        similar_non_essential_gene = similar_non_essential_genes.sample(n=1).iloc[0]

        # Add the pair of genes to the list
        paired_genes.append((essential_gene['ensembl'], similar_non_essential_gene['ensembl']))

        # Add the non-essential gene to the mapped dictionary
        mapped_non_essential_genes[essential_gene['ensembl']] = similar_non_essential_gene['ensembl']

# Extract non-essential genes from the pairs
non_essential_genes_list = [pair[1] for pair in paired_genes]
# Iterate through paired genes and print out their observed LOF values


# Filter non-essential genes dataframe to contain only the selected non-essential genes
non_essential_genes_df = non_essential_genes_df2[non_essential_genes_df2['ensembl'].isin(non_essential_genes_list)]

# Print the length of the dataframe containing non-essential genes
print("Number of non-essential genes:", len(non_essential_genes_df))

# Print the info of all the non-essential genes
#print(non_essential_genes_df)
print(essential_genes_df["obs_lof"])
print(non_essential_genes_df["obs_lof"])

# Print the info of all the essential genes
#print(essential_genes_df)

# Concatenate essential and non-essential genes into one dataframe
total_genes_df = pd.concat([essential_genes_df, non_essential_genes_df])

# Print the data types of the combined dataframe
print(total_genes_df.dtypes)



# Read the deeplof_scores.csv file
deeplof_scores_df = pd.read_csv('DeepLOF Scores.csv')

# Merge total_genes_df with deeplof_scores_df on the 'ensembl' column
total_genes_big_df = pd.merge(total_genes_df, deeplof_scores_df, on='ensembl', how='inner')

print(total_genes_big_df)



# Separate essential and non-essential genes
essential_genes = total_genes_big_df[total_genes_big_df['Essentiality test'] == 'Essential']
non_essential_genes = total_genes_big_df[total_genes_big_df['Essentiality test'] == 'Non-essential']
#print(essential_genes["obs_lof"])
print(non_essential_genes_df["obs_lof"])


# Plot frequency distribution of DeepLOF scores for essential genes

# plt.hist(essential_genes['DeepLOF_score'], bins=20, alpha=0.5, color='blue', label='Essential Genes')
#
# # Plot frequency distribution of DeepLOF scores for non-essential genes
# plt.hist(non_essential_genes['DeepLOF_score'], bins=20, alpha=0.5, color='red', label='Non-Essential Genes')
#
# # Add labels and title
# plt.xlabel('DeepLOF_Scores')
# plt.ylabel('Frequency')
# plt.title('Frequency Distribution of DeepLOF Scores for Essential and Non-Essential Genes')
#
# # Add legend
# plt.legend()
#
# # Show the plot
# plt.show()

#sns.set(style="whitegrid")

# Create subplots
# plt.figure(figsize=(10, 6))
#
# # Plot density distribution of DeepLOF scores for essential genes
# sns.kdeplot(data=essential_genes['DeepLOF_score'], color='blue', label='Essential Genes')
#
# # Plot density distribution of DeepLOF scores for non-essential genes
# sns.kdeplot(data=non_essential_genes['DeepLOF_score'], color='red', label='Non-Essential Genes')
#
# # Add labels and title
# plt.xlabel('DeepLOF Scores')
# plt.ylabel('Density')
# plt.title('Density Plot of DeepLOF Scores for Essential and Non-Essential Genes')
#
# # Show legend
# plt.legend()
#
# # Show the plot
# plt.show()

for essential_gene, non_essential_gene in paired_genes:
    essential_lof = total_genes_big_df.loc[total_genes_big_df['ensembl'] == essential_gene, 'obs_lof'].iloc[0]
    non_essential_lof = total_genes_big_df.loc[total_genes_big_df['ensembl'] == non_essential_gene, 'obs_lof'].iloc[0]
    print(f"Paired Genes: {essential_gene} (Essential LOF: {essential_lof}), {non_essential_gene} (Non-Essential LOF: {non_essential_lof})")


# Plot boxplot to compare DeepLOF scores between essential and non-essential genes

# Define features (X) and target variable (y)
X = total_genes_big_df[['DeepLOF_score', 'obs_lof']]
y = total_genes_big_df['Essentiality test']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print(classification_report(y_test, y_pred))



# Get predicted probabilities for the positive class (essential genes)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
# Convert labels to binary format
y_test_binary = (y_test == 'Essential').astype(int)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)
roc_auc = auc(fpr, tpr)


# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

