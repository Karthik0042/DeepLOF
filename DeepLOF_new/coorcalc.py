import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read DeepLOF Scores.csv file
output_result = pd.read_csv('../outputresult4.tsv', sep='\t')

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

# Initialize empty lists to store correlation values and feature names
correlation_values = []
feature_names = []

# Iterate over each feature
for feature_name in merged_data.columns[4:]:
    # Assuming features start from column 5
    if feature_name != "Essentiality test":
        # Calculate correlation between the feature and the target variable
        correlation = merged_data[feature_name].corr(merged_data['Essentiality test'])
        correlation_values.append(correlation)
        feature_names.append(feature_name)

# Sort features based on correlation values
sorted_indices = np.argsort(correlation_values)[::-1]  # Sort in descending order
sorted_correlation_values = np.array(correlation_values)[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

# Plot correlation values of each feature in decreasing order
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_feature_names, sorted_correlation_values, color='skyblue')
plt.xlabel('Correlation')
plt.ylabel('Feature')
plt.title('Correlation between Each Feature and Essentiality Test')
plt.gca().invert_yaxis()  # Invert y-axis to display features in decreasing order

# Add the correlation values beside each bar
for bar, correlation_value in zip(bars, sorted_correlation_values):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{correlation_value:.2f}',
             va='center', ha='left', color='black')

plt.show()
