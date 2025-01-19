import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Initialize empty lists to store accuracy and relative accuracy for each feature
accuracy_list = []
relative_accuracy_list = []
feature_names = []

output_result = pd.read_csv('../outputresult2.tsv', sep='\t')

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

# Drop the "Gene ID" column from the merged_data DataFrame
merged_data.drop(columns=['Gene ID'], inplace=True)

y = merged_data['Essentiality test']

# Iterate over each feature
for feature_name in merged_data.columns[4:]:
    # Assuming features start from column 5
    if feature_name != "Essentiality test":
        # Define features (X)
        X = merged_data[[feature_name]]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

        # Initialize and train XGBoost classifier
        model = XGBClassifier()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        relative_accuracy = (accuracy - 0.5) * 2

        # Append accuracy and relative accuracy to the lists
        accuracy_list.append(accuracy)
        relative_accuracy_list.append(relative_accuracy)
        feature_names.append(feature_name)

# Sort features based on accuracy
sorted_indices = np.argsort(accuracy_list)[::-1]  # Sort in descending order
sorted_accuracy = np.array(accuracy_list)[sorted_indices]
sorted_relative_accuracy = np.array(relative_accuracy_list)[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

# Plot relative accuracy of each feature in decreasing order
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_feature_names, sorted_relative_accuracy, color='skyblue')
plt.xlabel('Relative Accuracy')
plt.ylabel('Feature')
plt.title('Relative Accuracy of Each Feature')
plt.gca().invert_yaxis()  # Invert y-axis to display features in decreasing order

# Add the relative accuracy values beside each bar
for bar, relative_accuracy in zip(bars, sorted_relative_accuracy):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{relative_accuracy:.2f}',
             va='center', ha='left', color='black')

plt.show()
