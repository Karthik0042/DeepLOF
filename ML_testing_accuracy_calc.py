import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Read DeepLOF Scores.csv file
output_result = pd.read_csv('outputresult4.tsv', sep='\t')

# Read inputdata.csv file
input_data = pd.read_csv("inputdata.csv")

# Read Human_essentiality.csv file
human_essentiality = pd.read_csv("Human_essentiality.csv")

# Merge data on 'ensembl' column in input_data and output_result
merged_data = pd.merge(input_data, output_result, on='ensembl', how='inner')

# Merge data on 'Gene ID' column in merged_data and human_essentiality
merged_data = pd.merge(merged_data, human_essentiality, left_on='ensembl', right_on='Gene ID', how='inner')

# Convert 'Essentiality test' column to binary format (1 for essential, 0 for non-essential)
merged_data['Essentiality test'] = merged_data['Essentiality test'].map({'Non-essential': 0, 'Essential': 1})

# Define features (X) and target variable (y)
X = merged_data[['DeepLOF_score']]
y = merged_data['Essentiality test']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=67)

# Initialize and train XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("relative accuracy:",(accuracy-0.5)*2)
