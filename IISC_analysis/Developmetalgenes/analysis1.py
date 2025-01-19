from subcategories import *
import pandas as pd
df1 = delvelopmetalessentail.copy()
df2 = cellularlessentail.copy()
dfprotien = Protien_complex.copy()
dftranscription = Transcription.copy()
print(len(dfprotien))
dfmain = pd.merge(df1, dfprotien,left_on='ensembl',right_on='ensembl', how='inner')
dfmain2 = pd.merge(df1, dftranscription,left_on='ensembl',right_on='ensembl', how='inner')

print(len(dfmain))

