from IISC_analysis.gettingsubcategories import dftranscription, dfprotien
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from IISC_analysis.preprocessing import non_essential_genes_lof, essential_genes_lof
import pandas as pd
with open("/Users/karthikrajesh/PycharmProjects/DeepLOF/IISC_analysis/gettingsubcategories.py", "r") as file:
    exec(file.read())

dfprotienrand = dfprotien.sample(frac=1, random_state=42).reset_index(drop=True)
dfprotienrandessential=dfprotienrand[dfprotienrand['Essentiality test']==1]
dfprotienrandnonessential=dfprotienrand[dfprotienrand['Essentiality test']==0]


print(len(dfprotienrandessential))
n = 315
dfprotienrandessential = dfprotienrandessential.sample(n=n)
print(len(dfprotienrandessential))

# Create a function to perform a binary classifcation on the data
def binary_classification(data, target):
    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

#print(binary_classification(dfprotienrand[['lof_score', 'DeepLOF_score']], dfprotienrand['Essentiality test' ]))

# Create a function that plots between the DeepLOF score and the LOF score and make essential ones red and non-essential ones blue
def plot_deeplof_scores(final_df):
    # Create a scatter plot
    plt.figure(figsize=(10, 6))

    #Plot essential genes
    # plt.scatter(final_df[final_df['Essentiality test'] == 1]['lof_score'],
    #             final_df[final_df['Essentiality test'] == 1]['DeepLOF_score'],
    #             color='blue', label='Essential Genes')

    # Plot non-essential genes
    plt.scatter(final_df[final_df['Essentiality test'] == 0]['lof_score'],
                final_df[final_df['Essentiality test'] == 0]['DeepLOF_score'],
               color='red', label='Non-essential Genes')

    # Add labels and title
    plt.xlabel('LOF Score')
    plt.ylabel('DeepLOF Scores')
    plt.title('DeepLOF Scores of Essential and Non-essential Genes')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()


#plot_deeplof_scores(dfprotienrand)


# Write a function to plot the frequency distribution of the DeepLOF scores and  make essential ones red and non-essential ones blue
def plot_frequency_distribution(data, xlabel='', ylabel='', title='', color='blue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, color=color, bins=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

#plot_frequency_distribution(dfprotienrandessential['DeepLOF_score'], xlabel='DeepLOF Score', ylabel='Frequency', title='Frequency Distribution of DeepLOF Scores', color='blue')




def plot_average_frequency_distribution(data, num_runs=10000, sample_size=315, bins=100, xlabel='', ylabel='', title='', color='blue'):
    all_frequencies = np.zeros(bins)
    bin_edges = None

    for _ in range(num_runs):
        # Sample the data
        sample_data = data.sample(n=sample_size, random_state=None)

        # Compute the histogram
        frequencies, bin_edges = np.histogram(sample_data, bins=bins, range=(data.min(), data.max()))
        all_frequencies += frequencies

    # Average the frequencies
    avg_frequencies = all_frequencies / num_runs

    # Plot the averaged frequency distribution
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], avg_frequencies, width=(bin_edges[1] - bin_edges[0]), color=color, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Assuming dfprotienrandessential contains the necessary data
plot_average_frequency_distribution(dfprotienrandessential['DeepLOF_score'], xlabel='DeepLOF Score', ylabel='Average Frequency', title='Averaged Frequency Distribution of DeepLOF Scores', color='blue')
