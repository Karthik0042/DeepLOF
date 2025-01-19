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

# Drop 'DeepLOF Score' column
merged_data.drop(columns=['DeepLOF_score'], inplace=True)
print(merged_data.dtypes)
# Define features (X) and target variable (y)
X = merged_data.drop(columns=['ensembl', 'gene_symbol_x', 'gene_symbol_y', 'Gene ID', 'Essentiality test'])
y = merged_data['Essentiality test']

# Split data into training and testing sets
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

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)  # You can adjust max_iter as needed
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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

# Define features (X) and target variable (y)
X = merged_data.drop(columns=['ensembl', 'gene_symbol_x', 'gene_symbol_y', 'Gene ID', 'Essentiality test'])
y = merged_data['Essentiality test']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Get feature importance
feature_importance = model.coef_[0]

# Get feature names
feature_names = X.columns

# Create DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

