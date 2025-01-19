import pandas as pd
mouse_data = pd.read_csv('/Users/karthikrajesh/Downloads/impc_essential_genes_full_dataset.csv')
print(mouse_data.columns)
updated_mouse_data = mouse_data.dropna(subset=['human_hgnc_acc_id'])
updated_mouse_data = updated_mouse_data.reset_index(drop=True)
print(len(updated_mouse_data))

