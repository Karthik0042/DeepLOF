import subprocess
import pandas as pd
from IISC_analysis.preprocessing2 import non_essential_genes_lof, essential_genes_lof

with open("/Users/karthikrajesh/PycharmProjects/DeepLOF/IISC_analysis/preprocessing2.py", "r") as file:
    exec(file.read())
df = pd.concat([essential_genes_lof, non_essential_genes_lof])
print(df.columns)
df['lof_score'] = df['exp_lof'] / df['obs_lof']
dfprotien = df[(df['protein_complex'] == 1)]
dftranscription=df[(df["transcript_factor"]==1)]
dfreactome_nervous_dev = df[(df['reactome_nervous_development'] == 1)]
dfreactome_nervous_biology = df[(df['reactome_developmental_biology'] == 1)]
dfGOnervous_development = df[(df['GO_central_neurvous_development'] == 1)]
dfembryodev = df[(df['GO_embryo_development'] == 1)]
#print(df.columns)
