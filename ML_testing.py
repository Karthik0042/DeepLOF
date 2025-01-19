
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read DeepLOF Scores.csv file
deep_lof_scores = pd.read_csv("DeepLOF Scores.csv")

# Read inputdata.csv file
input_data = pd.read_csv("inputdata.csv")

# Read Human_essentiality.csv file
human_essentiality = pd.read_csv("Human_essentiality.csv")

# Merge data on 'ensembl' column in input_data and deep_lof_scores
merged_data = pd.merge(input_data, deep_lof_scores, on='ensembl', how='inner')

# Merge data on 'Gene ID' column in merged_data and human_essentiality
merged_data = pd.merge(merged_data, human_essentiality, left_on='ensembl', right_on='Gene ID', how='inner')
print(merged_data.dtypes)



# Split data into training and testing sets

# Step 1: Define features (X) and target variable (y)
X = merged_data.drop(columns=['ensembl', 'gene_symbol_x','gene_symbol_y' ,'Gene ID', 'Essentiality test'])
y = merged_data['Essentiality test']

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate through each column
for column in X_train.columns:
    # Check if the column contains non-numeric values
    non_numeric_values = X_train[column].apply(lambda x: isinstance(x, str))
    if non_numeric_values.any():
        # Print the column name if it contains non-numeric values
        print("Column '{}' contains non-numeric values.".format(column))
        # Print the unique non-numeric values
        unique_non_numeric_values = X_train[column][non_numeric_values].unique()
        print("Unique non-numeric values:", unique_non_numeric_values)


# Step 3: Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)  # You can adjust max_iter as needed
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
#
# # Step 6: Compute predicted probabilities
# y_probs = model.predict_proba(X_test)[:, 1]
#
# # Step 7: Calculate false positive rate, true positive rate, and thresholds
# fpr, tpr, thresholds = roc_curve(y_test, y_probs)
#
# # Step 8: Calculate area under curve (AUC)
# roc_auc = auc(fpr, tpr)
#
# # Step 9: Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()
#
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_curve, auc
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
#
# # Read DeepLOF Scores.csv file
# deep_lof_scores = pd.read_csv("DeepLOF Scores.csv")
#
# # Read inputdata.csv file
# input_data = pd.read_csv("inputdata.csv")
#
# # Read Human_essentiality.csv file
# human_essentiality = pd.read_csv("Human_essentiality.csv")
#
# # Merge data on 'ensembl' column in input_data and deep_lof_scores
# merged_data = pd.merge(input_data, deep_lof_scores, on='ensembl', how='inner')
#
# # Merge data on 'Gene ID' column in merged_data and human_essentiality
# merged_data = pd.merge(merged_data, human_essentiality, left_on='ensembl', right_on='Gene ID', how='inner')
#
# # Split data into training and testing sets
# X = merged_data.drop(columns=['ensembl', 'gene_symbol_x', 'gene_symbol_y', 'Gene ID', 'Essentiality test'])
# y = merged_data['Essentiality test']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Iterate through each column
# for column in X_train.columns:
#     non_numeric_values = X_train[column].apply(lambda x: isinstance(x, str))
#     if non_numeric_values.any():
#         print("Column '{}' contains non-numeric values.".format(column))
#         unique_non_numeric_values = X_train[column][non_numeric_values].unique()
#         print("Unique non-numeric values:", unique_non_numeric_values)
#
# # Initialize and train logistic regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Evaluate model performance
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# # Compute predicted probabilities
# y_probs = model.predict_proba(X_test)[:, 1]
#
# # Encode target variable
# label_encoder = LabelEncoder()
# y_test_encoded = label_encoder.fit_transform(y_test)
#
# # Compute ROC curve
# fpr, tpr, thresholds = roc_curve(y_test_encoded, y_probs)
#
# # Calculate area under curve (AUC)
# roc_auc = auc(fpr, tpr)
#
# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()


