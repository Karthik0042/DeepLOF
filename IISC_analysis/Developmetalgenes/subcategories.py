import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv("/Users/karthikrajesh/PycharmProjects/DeepLOF/IISC_analysis/FinalDF_organismal.csv")
df1 = df1.drop('Gene ID',axis=1)
print(df1.columns)
print(len(df1))

Protien_complex = df1[(df1['protein_complex'] == 1)]
Transcription = df1[(df1["transcript_factor"]==1)]
delvelopmetalessentail = df1[(df1['DL'] == 1)]
cellularlessentail = df1[(df1['CL'] == 1)]



