import pandas as pd
import matplotlib.pyplot as plt
from IISC_analysis.gettingsubcategories import *
import seaborn as sns
from IISC_analysis.preprocessing import non_essential_genes_lof, essential_genes_lof

with open("/Users/karthikrajesh/PycharmProjects/DeepLOF/IISC_analysis/gettingsubcategories.py", "r") as file:
    exec(file.read())


dftranscriptionrand = dftranscription.sample(frac=1, random_state=42).reset_index(drop=True)
dftranscriptionrandessential=dftranscriptionrand[dftranscriptionrand['Essentiality test']==1]
dftranscriptionrandnonessential=dftranscriptionrand[dftranscriptionrand['Essentiality test']==0]
print(dftranscriptionrand.columns)
# Create a function to perform a binary Essentiality testassifcation on the data

dfprotienrand = dfprotien.sample(frac=1, random_state=42).reset_index(drop=True)
dfprotienrandessential=dfprotienrand[dfprotienrand['Essentiality test']==1]
dfprotienrandnonessential=dfprotienrand[dfprotienrand['Essentiality test']==0]

print(len(dftranscriptionrandessential))
print(len(dftranscriptionrandnonessential))


def violinplot(dataframe):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Essentiality test', y="DeepLOF_score", data=dataframe, palette='Set3')

    # sns.stripplot(x='Essentiality test', y='DeepLOF_score', data=dfprotienrand, jitter=True, color='blue', marker='o', alpha=0.7)

    plt.xlabel('Essentiality test')
    plt.ylabel('deepLOF Score')
    plt.title('Violin Plot of deepLOF Scores by Essentiality test')
    plt.show()

violinplot(dfprotien)

def violinplotcompare(df1,df2,label1="transcription",label2="protien"):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Essentiality test', y="promoter_phastcons", data=df1, palette='Set3',label=label1)
    ax = sns.violinplot(x='Essentiality test', y="promoter_phastcons", data=df2, palette='pastel',label = label2)
    for violin in ax.collections:
        violin.set_alpha(0.2)

    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles, [label1, label2])

    plt.xlabel('Essentiality test')
    plt.ylabel('deepLOF Score')
    plt.title('Violin Plot of deepLOF Scores by Essentiality test')
    plt.show()
violinplotcompare(dfprotien#dfembryodev
                  ,#dfreactome_nervous_biology
                  dfGOnervous_development )

''''dfreactome_nervous_dev
dfreactome_nervous_biology 
dfGOnervous_development
dfembryodev
'''



def compare_distributions(df1, df2, label1="Transcription", label2="Protein"):
    plt.figure(figsize=(10, 6))

    # Perform KDE and plot the estimated density functions
    sns.kdeplot(df1['DeepLOF_score'], label=label1)
    sns.kdeplot(df2['DeepLOF_score'], label=label2)

    plt.xlabel('DeepLOF Score')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation of deepLOF Scores')
    plt.legend()
    plt.show()

# Example usage:
#compare_distributions(dftranscriptionrandnonessential, dfprotienrandnonessential, label1='Transcription', label2='Protein')

def compute_histogram(data1,data2, bins=10):
    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data1, bins=bins, color='skyblue', edgecolor='black', alpha=1)
    plt.hist(data2, bins=bins, color='red', edgecolor='black', alpha=0.2)

    # Add labels and title
    plt.xlabel('DeepLOF Score')
    plt.ylabel('Frequency')
    plt.title("Histogram of DeepLOF Scores of non essential genes blue is Tf's and red is protiens")

    # Show plot
    plt.show()

#compute_histogram(dftranscriptionrandnonessential['DeepLOF_score'], dfprotienrandnonessential['DeepLOF_score'], bins=100)
