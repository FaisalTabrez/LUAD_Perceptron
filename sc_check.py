import scanpy as sc

print("Loading the Single-Cell dataset (Please wait a moment)...")
# Replace 'lung_cancer_dataset.h5ad' with your actual file name!
adata = sc.read_h5ad('LUAD.h5ad') 

print("\n--- Original Scientist Annotations (Level 1) ---")
print(adata.obs['author_cell_type_level_1'].value_counts())

print("\n--- Original Scientist Annotations (Level 2) ---")
print(adata.obs['author_cell_type_level_2'].value_counts())

print("\n--- Disease State ---")
print(adata.obs['disease'].value_counts())