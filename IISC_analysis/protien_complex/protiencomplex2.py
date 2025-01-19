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
print(len(dfprotienrandnonessential))
print(dfprotienrand.columns)
def boxplot(dataframe):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Essentiality test', y='DeepLOF_score', data=dataframe, palette='Set3')

    # sns.stripplot(x='Essentiality test', y='DeepLOF_score', data=dfprotienrand, jitter=True, color='blue', marker='o', alpha=0.7)

    plt.xlabel('Essentiality Test')
    plt.ylabel('deepLOF Score')
    plt.title('Box Plot of deepLOF Scores by Essentiality Test')
    plt.show()
def violinplot(dataframe):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Essentiality test', y="UNEECON.G", data=dataframe, palette='Set3')

    # sns.stripplot(x='Essentiality test', y='DeepLOF_score', data=dfprotienrand, jitter=True, color='blue', marker='o', alpha=0.7)

    plt.xlabel('Essentiality Test')
    plt.ylabel('deepLOF Score')
    plt.title('Violin Plot of deepLOF Scores by Essentiality Test')
    plt.show()
violinplot(dfprotienrand)

