import matplotlib.pyplot as plt
from IISC_analysis.gettingsubcategories import *
import matplotlib.pyplot as plt

import seaborn as sns
from IISC_analysis.preprocessing import non_essential_genes_lof, essential_genes_lof

with open("/Users/karthikrajesh/PycharmProjects/DeepLOF/IISC_analysis/gettingsubcategories.py", "r") as file:
    exec(file.read())
print(df)
print(df.columns)



def piechart(df,column="Essentiality test"):
    plt.figure(figsize=(10, 6))
    plt.pie(df[column].value_counts(), labels=df[column].value_counts().index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f'Pie Chart of {column} distribution')
    plt.show()

for df in [dfGOnervous_development,dfreactome_nervous_dev,dfreactome_nervous_biology,dfembryodev,dfprotien,dftranscription]:
    piechart(df)
