import scanpy as sc
import numpy as np
import itertools
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
# ELITE GENE POOLS
# ==========================================
print("\n3. Calculating Differential Expression to create Elite Gene Pools...")
cancer_expr = X_matrix[y_target == 1]
healthy_expr = X_matrix[y_target == 0]

mean_cancer = np.mean(cancer_expr, axis=0)
mean_healthy = np.mean(healthy_expr, axis=0)

fold_change_cancer = mean_cancer / (mean_healthy + 1e-9)
fold_change_healthy = mean_healthy / (mean_cancer + 1e-9)

# Select the Top 40 Elite Genes for speed and precision
elite_promoters = np.argsort(fold_change_cancer)[-40:]
elite_repressors = np.argsort(fold_change_healthy)[-40:]

# ==========================================
# EXHAUSTIVE BRUTE-FORCE ALGORITHM
# ==========================================
print("\n4. Generating ALL possible Elite Combinations...")
# Create all unique pairs of Promoters, matched with every Repressor
promoter_pairs = list(itertools.combinations(elite_promoters, 2))
all_combinations = list(itertools.product(promoter_pairs, elite_repressors))

total_combos = len(all_combinations)
print(f"Total Circuits to Evaluate: {total_combos}")
print("Commencing Exhaustive Search (This will take ~30-60 seconds)...\n")

WEIGHT_TP = 2.0       
WEIGHT_FP = 50.0      

best_reward = -np.inf
best_circuit = {}

# Pre-calculate arrays for faster math
cancer_mask = (y_target == 1)
healthy_mask = (y_target == 0)

for idx, combo in enumerate(all_combinations):
    p1_idx, p2_idx = combo[0]
    r_idx = combo[1]
    
    # SMART BIOLOGICAL THRESHOLDS:
    # Set Promoter threshold just above the healthy cell baseline (to prevent misfires)
    p1_thr = np.percentile(X_matrix[healthy_mask, p1_idx], 95) + 0.1
    p2_thr = np.percentile(X_matrix[healthy_mask, p2_idx], 95) + 0.1
    
    # Set Repressor threshold just above the cancer baseline (to prevent cancer from blocking it)
    r_thr = np.percentile(X_matrix[cancer_mask, r_idx], 95) + 0.1
    
    # Test the Circuit
    circuit_fired = ((X_matrix[:, p1_idx] > p1_thr) | (X_matrix[:, p2_idx] > p2_thr)) & (X_matrix[:, r_idx] < r_thr)
    
    tp = np.sum(circuit_fired[cancer_mask])
    fp = np.sum(circuit_fired[healthy_mask])
    
    reward = (tp * WEIGHT_TP) - (fp * WEIGHT_FP)
    
    if reward > best_reward:
        best_reward = reward
        best_circuit = {
            'P1': gene_names[p1_idx], 'P1_Thr': p1_thr,
            'P2': gene_names[p2_idx], 'P2_Thr': p2_thr,
            'R': gene_names[r_idx], 'R_Thr': r_thr,
            'Cancer_Killed': tp,
            'Friendly_Fire': fp
        }
    
    # Print Progress Bar
    if idx % 5000 == 0 and idx > 0:
        print(f"Evaluated {idx} / {total_combos} circuits... (Current Best Kill Rate: {(best_circuit['Cancer_Killed']/sum(cancer_mask))*100:.1f}%)")

# ==========================================
# FINAL MATHEMATICAL MAXIMUM REPORT
# ==========================================
print("\n========================================")
print("  GLOBAL MAXIMUM DISCOVERED (EXHAUSTIVE) ")
print("========================================")
print(f"Logic Gate: (PROMOTER_1 OR PROMOTER_2) AND (NOT REPRESSOR)")
print(f"Promoter 1 (+): {best_circuit['P1']} (Thr > {best_circuit['P1_Thr']:.2f})")
print(f"Promoter 2 (+): {best_circuit['P2']} (Thr > {best_circuit['P2_Thr']:.2f})")
print(f"Repressor  (-): {best_circuit['R']} (Thr < {best_circuit['R_Thr']:.2f})")

cancer_total = sum(cancer_mask)
healthy_total = sum(healthy_mask)
print("\n--- Absolute Safety Report ---")
print(f"Tumor Cells Destroyed: {best_circuit['Cancer_Killed']} / {cancer_total}")
print(f"Healthy Immune Cells Destroyed: {best_circuit['Friendly_Fire']} / {healthy_total}")

tpr = (best_circuit['Cancer_Killed'] / cancer_total) * 100
fpr = (best_circuit['Friendly_Fire'] / healthy_total) * 100
print(f"\nMaximized Efficacy (Kill Rate): {tpr:.1f}%")
print(f"Minimized Toxicity (False Positive Rate): {fpr:.2f}%")