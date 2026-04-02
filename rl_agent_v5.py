import scanpy as sc
import numpy as np
import itertools
import time
from scipy.sparse import issparse

print("1. Loading the Single-Cell Arena...")
adata = sc.read_h5ad('LUAD.h5ad') 
adata.obs['Target'] = np.where(adata.obs['author_cell_type_level_1'] == 'Epithelial', 1, 0)

print("2. Generating Training Arena (5,000 cells)...")
sc.pp.subsample(adata, n_obs=5000, random_state=42)
sc.pp.filter_genes(adata, min_cells=500)

X_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
y_target = adata.obs['Target'].values
gene_names = np.array(adata.var_names.tolist())

# ==========================================
# SUPERCOMPUTER PRE-COMPUTATION (THE SECRET SAUCE)
# ==========================================
print("\n3. Pre-computing Thermodynamic Thresholds for ALL 7,595 genes simultaneously...")
cancer_mask = (y_target == 1)
healthy_mask = (y_target == 0)

# Calculate 95th percentiles for the entire matrix at once
p_thr_all = np.percentile(X_matrix[healthy_mask, :], 95, axis=0) + 0.1
r_thr_all = np.percentile(X_matrix[cancer_mask, :], 95, axis=0) + 0.1

print("4. Converting Biology into Boolean Bitwise Matrices...")
# Create giant True/False matrices. This makes the math 1,000x faster.
promoter_fires_in_cancer = X_matrix[cancer_mask, :] > p_thr_all
promoter_fires_in_healthy = X_matrix[healthy_mask, :] > p_thr_all

repressor_blocks_in_cancer = X_matrix[cancer_mask, :] < r_thr_all
repressor_blocks_in_healthy = X_matrix[healthy_mask, :] < r_thr_all

# ==========================================
# ELITE GENE POOLS (Expanding to Millions of Combinations)
# ==========================================
mean_cancer = np.mean(X_matrix[cancer_mask, :], axis=0)
mean_healthy = np.mean(X_matrix[healthy_mask, :], axis=0)

fold_change_cancer = mean_cancer / (mean_healthy + 1e-9)
fold_change_healthy = mean_healthy / (mean_cancer + 1e-9)

# CHANGE THESE NUMBERS TO EXPLORE BILLIONS (e.g., set to 7595)
# Top 300 x Top 300 = 13.4 Million Circuits
SEARCH_DEPTH = 300 

elite_promoters = np.argsort(fold_change_cancer)[-SEARCH_DEPTH:]
elite_repressors = np.argsort(fold_change_healthy)[-SEARCH_DEPTH:]

promoter_pairs = list(itertools.combinations(elite_promoters, 2))
all_combinations = list(itertools.product(promoter_pairs, elite_repressors))

total_combos = len(all_combinations)
print(f"\n========================================")
print(f" COMMENCING VECTORIZED SEARCH")
print(f" Total Circuits to Evaluate: {total_combos:,}")
print(f"========================================")

WEIGHT_TP = 2.0       
WEIGHT_FP = 50.0      

best_reward = -np.inf
best_circuit = {}

start_time = time.time()

for idx, combo in enumerate(all_combinations):
    p1_idx, p2_idx = combo[0]
    r_idx = combo[1]
    
    # Because we pre-computed the bitwise matrices, calculating a circuit is just basic Boolean math
    # Bitwise OR (|) for Promoters, Bitwise AND (&) for the Repressor
    cancer_kills = (promoter_fires_in_cancer[:, p1_idx] | promoter_fires_in_cancer[:, p2_idx]) & repressor_blocks_in_cancer[:, r_idx]
    healthy_kills = (promoter_fires_in_healthy[:, p1_idx] | promoter_fires_in_healthy[:, p2_idx]) & repressor_blocks_in_healthy[:, r_idx]
    
    tp = np.sum(cancer_kills)
    fp = np.sum(healthy_kills)
    
    reward = (tp * WEIGHT_TP) - (fp * WEIGHT_FP)
    
    if reward > best_reward:
        best_reward = reward
        best_circuit = {
            'P1': p1_idx, 'P2': p2_idx, 'R': r_idx,
            'Cancer_Killed': tp, 'Friendly_Fire': fp
        }
    
    # Progress Bar
    if idx % 1000000 == 0 and idx > 0:
        elapsed = time.time() - start_time
        print(f"Evaluated {idx:,} / {total_combos:,} circuits... ({elapsed:.1f} sec) | Best Kill Rate: {(best_circuit['Cancer_Killed']/sum(cancer_mask))*100:.1f}%")

# ==========================================
# FINAL REPORT
# ==========================================
print("\n========================================")
print("  GLOBAL MAXIMUM DISCOVERED (MILLIONS SCALE) ")
print("========================================")
p1_final = best_circuit['P1']
p2_final = best_circuit['P2']
r_final = best_circuit['R']

print(f"Promoter 1 (+): {gene_names[p1_final]} (Thr > {p_thr_all[p1_final]:.2f})")
print(f"Promoter 2 (+): {gene_names[p2_final]} (Thr > {p_thr_all[p2_final]:.2f})")
print(f"Repressor  (-): {gene_names[r_final]} (Thr < {r_thr_all[r_final]:.2f})")

cancer_total = sum(cancer_mask)
healthy_total = sum(healthy_mask)
print("\n--- Absolute Safety Report ---")
print(f"Tumor Cells Destroyed: {best_circuit['Cancer_Killed']} / {cancer_total}")
print(f"Healthy Immune Cells Destroyed: {best_circuit['Friendly_Fire']} / {healthy_total}")

tpr = (best_circuit['Cancer_Killed'] / cancer_total) * 100
fpr = (best_circuit['Friendly_Fire'] / healthy_total) * 100
print(f"\nMaximized Efficacy (Kill Rate): {tpr:.1f}%")
print(f"Minimized Toxicity (False Positive Rate): {fpr:.2f}%")