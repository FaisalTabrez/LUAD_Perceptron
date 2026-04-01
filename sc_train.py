import scanpy as sc
import numpy as np
import pandas as pd

# 1. Load the Single-Cell Data
print("Loading the Tumor Microenvironment (This might take a minute)...")
# Replace with the exact name of your downloaded .h5ad file
adata = sc.read_h5ad('LUAD.h5ad') 

print(f"\nGame Board Loaded: {adata.n_obs} individual cells, {adata.n_vars} genes.")

# 2. Find the "Cell Type" column
# In CellxGene datasets, the cell type is usually stored in adata.obs['cell_type'] 
# or 'author_cell_type' or 'cell_ontology_class'
print("\nInspecting available metadata columns:")
print(adata.obs.columns.tolist())

# Assuming the column is named 'cell_type' (you may need to change this based on the printout above)
target_column = 'cell_type'  # <-- Change this if the column name is different

if target_column in adata.obs.columns:
    print("\n--- Identifying Factions in the Tumor Microenvironment ---")
    cell_counts = adata.obs[target_column].value_counts()
    print(cell_counts)
    
    # 3. Define the RL Agent's Targets
    # We will mathematically group these into 1 (Kill) and 0 (Protect)
    # Note: You will map these exact string names based on what the printout says!
    malignant_labels =['malignant cell', 'cancer cell', 'epithelial tumor cell'] # The targets
    
    # Create the binary target column
    adata.obs['RL_Target'] = np.where(adata.obs[target_column].isin(malignant_labels), 1, 0)
    
    print("\n--- Final RL Environment Matrix ---")
    print(f"Cancer Cells to Destroy: {sum(adata.obs['RL_Target'] == 1)}")
    print(f"Healthy/Immune Cells to Protect: {sum(adata.obs['RL_Target'] == 0)}")
else:
    print(f"Could not find '{target_column}' in the metadata. Please check the column names printed above!")

# Optional: Free up memory if the computer is struggling
# import gc; gc.collect()