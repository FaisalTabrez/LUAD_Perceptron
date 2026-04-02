import scanpy as sc
import numpy as np
import random
from scipy.sparse import issparse

print("1. Loading the Single-Cell Arena...")
# Update filename if necessary!
adata = sc.read_h5ad('LUAD.h5ad') 

adata.obs['Target'] = np.where(adata.obs['author_cell_type_level_1'] == 'Epithelial', 1, 0)

print("2. Generating Training Arena (5,000 cells)...")
sc.pp.subsample(adata, n_obs=5000, random_state=42)
sc.pp.filter_genes(adata, min_cells=500)

X_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
y_target = adata.obs['Target'].values
gene_names = adata.var_names.tolist()

# ==========================================
# V2 RL HYPERPARAMETERS (EXTREME STRICTNESS)
# ==========================================
WEIGHT_TP = 1.0       # Reward for killing 1 Cancer Cell
WEIGHT_FP = 50.0      # FATAL PENALTY for killing 1 Healthy Cell (Up from 10)
WEIGHT_MARGIN = 5.0   # Bonus multiplier for wide biological margins
EPISODES = 3000       # 3x deeper exploration

print(f"\n3. Initializing v2 RL Agent for {EPISODES} Episodes...")
print(f"Strictness Level: MAXIMUM (Friendly Fire Penalty = -{WEIGHT_FP})\n")

def calculate_reward_v2(promoter_idx, repressor_idx, threshold):
    promoter_expr = X_matrix[:, promoter_idx]
    repressor_expr = X_matrix[:, repressor_idx]
    
    circuit_fired = (promoter_expr > threshold) & (repressor_expr < threshold)
    
    tp = np.sum((circuit_fired == True) & (y_target == 1))
    fp = np.sum((circuit_fired == True) & (y_target == 0))
    
    # NEW: Calculate Biological Margin (Distance between Cancer and Healthy)
    # We want the Promoter to be HIGH in cancer, and Repressor HIGH in healthy
    promoter_margin = np.mean(promoter_expr[y_target == 1]) - np.mean(promoter_expr[y_target == 0])
    repressor_margin = np.mean(repressor_expr[y_target == 0]) - np.mean(repressor_expr[y_target == 1])
    
    # If the margin is negative (wrong biological direction), punish it
    margin_bonus = (promoter_margin + repressor_margin) * WEIGHT_MARGIN
    
    # The new, hyper-strict reward function
    reward = (tp * WEIGHT_TP) - (fp * WEIGHT_FP) + margin_bonus
    return reward, tp, fp

best_reward = -999999
best_circuit = {}

for episode in range(EPISODES):
    action_promoter = random.randint(0, len(gene_names) - 1)
    action_repressor = random.randint(0, len(gene_names) - 1)
    if action_promoter == action_repressor: continue
        
    # Widened search space for the threshold
    action_threshold = random.uniform(0.1, 5.0) 
    
    reward, tp, fp = calculate_reward_v2(action_promoter, action_repressor, action_threshold)
    
    if reward > best_reward:
        best_reward = reward
        best_circuit = {
            'Promoter': gene_names[action_promoter],
            'Repressor': gene_names[action_repressor],
            'Threshold': action_threshold,
            'Cancer_Killed': tp,
            'Friendly_Fire': fp
        }
        if episode % 100 == 0 or fp == 0:
            print(f"Ep {episode:04d} | Score: {reward:.1f} | Killed: {tp} | Friendly Fire: {fp} | Thr: {action_threshold:.2f}")

# ==========================================
# V2 MISSION REPORT
# ==========================================
print("\n========================================")
print("  V2 RL AGENT: HYPER-SAFE BLUEPRINT      ")
print("========================================")
print(f"Promoter Sensor (+): {best_circuit['Promoter']}")
print(f"Repressor Sensor (-): {best_circuit['Repressor']}")
print(f"Activation Threshold: {best_circuit['Threshold']:.2f}")
print("--- Safety Report ---")
cancer_total = sum(y_target == 1)
healthy_total = sum(y_target == 0)
print(f"Tumor Cells Destroyed: {best_circuit['Cancer_Killed']} / {cancer_total}")
print(f"Healthy Immune Cells Destroyed: {best_circuit['Friendly_Fire']} / {healthy_total}")

# Calculate True Positive Rate and False Positive Rate
tpr = (best_circuit['Cancer_Killed'] / cancer_total) * 100
fpr = (best_circuit['Friendly_Fire'] / healthy_total) * 100
print(f"\nEfficacy (Kill Rate): {tpr:.1f}%")
print(f"Toxicity (False Positive Rate): {fpr:.2f}%")