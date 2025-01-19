import pandas as pd
import matplotlib.pyplot as plt
from IISC_analysis.gettingsubcategories import dftranscription
import seaborn as sns
from IISC_analysis.preprocessing import non_essential_genes_lof, essential_genes_lof

with open("/Users/karthikrajesh/PycharmProjects/DeepLOF/IISC_analysis/gettingsubcategories.py", "r") as file:
    exec(file.read())


dftranscriptionrand = dftranscription.sample(frac=1, random_state=42).reset_index(drop=True)
dftranscriptionrandessential=dftranscriptionrand[dftranscriptionrand['Essentiality test']==1]
dftranscriptionrandnonessential=dftranscriptionrand[dftranscriptionrand['Essentiality test']==0]
# Create a function to perform a binary classifcation on the data


print(len(dftranscriptionrandessential))
print(len(dftranscriptionrandnonessential))
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


#lot_deeplof_scores(dftranscription)


def plot_frequency_distribution(data, xlabel='', ylabel='', title='', color='blue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, color=color, bins=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

plot_frequency_distribution(dftranscriptionrandessential['DeepLOF_score'], xlabel='DeepLOF Score', ylabel='Frequency', title='Frequency Distribution of DeepLOF Scores', color='blue')
