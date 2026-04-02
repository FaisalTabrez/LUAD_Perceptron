import scanpy as sc
import numpy as np
import random
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
# NEW: BIOLOGICALLY-INFORMED PRE-FILTERING
# ==========================================
print("\n3. Calculating Differential Expression to create Elite Gene Pools...")
cancer_expr = X_matrix[y_target == 1]
healthy_expr = X_matrix[y_target == 0]

mean_cancer = np.mean(cancer_expr, axis=0)
mean_healthy = np.mean(healthy_expr, axis=0)

# Calculate Fold Changes (Adding 1e-9 to prevent division by zero)
fold_change_cancer = mean_cancer / (mean_healthy + 1e-9)
fold_change_healthy = mean_healthy / (mean_cancer + 1e-9)

# Select the Top 50 Genes specific to Cancer (Promoter Pool)
elite_promoter_indices = np.argsort(fold_change_cancer)[-50:]
# Select the Top 50 Genes specific to Immune/Healthy cells (Repressor Pool)
elite_repressor_indices = np.argsort(fold_change_healthy)[-50:]

print(f"Elite Promoters Found: {len(elite_promoter_indices)}")
print(f"Elite Repressors Found: {len(elite_repressor_indices)}")

# ==========================================
# RL AGENT (Searching only the Elite Pools)
# ==========================================
WEIGHT_TP = 2.0       
WEIGHT_FP = 50.0      
EPISODES = 3000

print("\n4. Initializing v5 RL Agent (Elite OR-AND Logic Gate)...")

def calculate_reward_v5(p1_idx, p2_idx, r_idx, p1_thr, p2_thr, r_thr):
    circuit_fired = ((X_matrix[:, p1_idx] > p1_thr) | (X_matrix[:, p2_idx] > p2_thr)) & (X_matrix[:, r_idx] < r_thr)
    
    tp = np.sum((circuit_fired == True) & (y_target == 1))
    fp = np.sum((circuit_fired == True) & (y_target == 0))
    
    reward = (tp * WEIGHT_TP) - (fp * WEIGHT_FP)
    return reward, tp, fp

best_reward = -999999
best_circuit = {}

for episode in range(EPISODES):
    # Agent ONLY picks from the elite pools now!
    action_p1 = random.choice(elite_promoter_indices)
    action_p2 = random.choice(elite_promoter_indices)
    action_r = random.choice(elite_repressor_indices)
    
    if action_p1 == action_p2: continue 
        
    thr_p1 = random.uniform(0.1, 3.0) 
    thr_p2 = random.uniform(0.1, 3.0) 
    thr_r = random.uniform(0.1, 5.0)  
    
    reward, tp, fp = calculate_reward_v5(action_p1, action_p2, action_r, thr_p1, thr_p2, thr_r)
    
    if reward > best_reward:
        best_reward = reward
        best_circuit = {
            'P1': gene_names[action_p1], 'P1_Thr': thr_p1,
            'P2': gene_names[action_p2], 'P2_Thr': thr_p2,
            'R': gene_names[action_r], 'R_Thr': thr_r,
            'Cancer_Killed': tp,
            'Friendly_Fire': fp
        }
        if episode % 100 == 0 or fp == 0:
            print(f"Ep {episode:04d} | Score: {reward:.1f} | Killed: {tp} | Toxicity: {fp}")

# ==========================================
# FINAL REPORT
# ==========================================
print("\n========================================")
print("  V5 RL AGENT: MAXIMUM EFFICACY BLUEPRINT ")
print("========================================")
print(f"Logic Gate: (PROMOTER_1 OR PROMOTER_2) AND (NOT REPRESSOR)")
print(f"Promoter 1 (+): {best_circuit['P1']} (Thr > {best_circuit['P1_Thr']:.2f})")
print(f"Promoter 2 (+): {best_circuit['P2']} (Thr > {best_circuit['P2_Thr']:.2f})")
print(f"Repressor  (-): {best_circuit['R']} (Thr < {best_circuit['R_Thr']:.2f})")

cancer_total = sum(y_target == 1)
healthy_total = sum(y_target == 0)
print("\n--- Safety Report ---")
print(f"Tumor Cells Destroyed: {best_circuit['Cancer_Killed']} / {cancer_total}")
print(f"Healthy Immune Cells Destroyed: {best_circuit['Friendly_Fire']} / {healthy_total}")

tpr = (best_circuit['Cancer_Killed'] / cancer_total) * 100
fpr = (best_circuit['Friendly_Fire'] / healthy_total) * 100
print(f"\nEfficacy (Kill Rate): {tpr:.1f}%")
print(f"Toxicity (False Positive Rate): {fpr:.2f}%")