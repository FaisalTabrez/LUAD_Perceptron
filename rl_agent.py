import scanpy as sc
import numpy as np
import pandas as pd
import random
from scipy.sparse import issparse

print("1. Loading the Single-Cell Arena...")
adata = sc.read_h5ad('LUAD.h5ad') # UPDATE FILE NAME IF NEEDED

# Set the Targets: 1 = Cancer (Epithelial), 0 = Healthy (TME)
adata.obs['Target'] = np.where(adata.obs['author_cell_type_level_1'] == 'Epithelial', 1, 0)

# Create a fast training arena (5,000 cells) to prevent RAM crashes
print("2. Generating Training Arena (5,000 cells)...")
sc.pp.subsample(adata, n_obs=5000, random_state=42)

# Filter for highly expressed genes to give the RL agent realistic "parts"
sc.pp.filter_genes(adata, min_cells=500)
print(f"Arena Ready: {adata.n_obs} cells and {adata.n_vars} viable mRNA sensors.")

# Convert sparse matrix to dense array for fast math
X_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
y_target = adata.obs['Target'].values
gene_names = adata.var_names.tolist()

# ==========================================
# REINFORCEMENT LEARNING AGENT
# ==========================================
print("\n3. Initializing RL Bio-Compiler Agent...")

def calculate_reward(promoter_idx, repressor_idx, threshold):
    """ The Environment: Tests the circuit and returns the RL Reward """
    # Circuit Logic: IF (Promoter > Threshold) AND (Repressor < Threshold) -> FIRE
    promoter_expr = X_matrix[:, promoter_idx]
    repressor_expr = X_matrix[:, repressor_idx]
    
    # 1 means the cell was killed, 0 means it survived
    circuit_fired = (promoter_expr > threshold) & (repressor_expr < threshold)
    
    # Tally the results
    true_positives = np.sum((circuit_fired == True) & (y_target == 1))   # Cancer killed
    false_positives = np.sum((circuit_fired == True) & (y_target == 0))  # Friendly Fire!
    
    # The Reward Function
    # +1 for killing cancer. -10 for friendly fire (highly discouraged).
    reward = (true_positives * 1.0) - (false_positives * 10.0)
    return reward, true_positives, false_positives

# RL Hyperparameters
episodes = 1000
best_reward = -99999
best_circuit = {}

print("Agent is playing 1,000 episodes of the Game...\n")

for episode in range(episodes):
    # Action: Agent randomly selects 1 Promoter mRNA, 1 Repressor mRNA, and a Threshold
    action_promoter = random.randint(0, len(gene_names) - 1)
    action_repressor = random.randint(0, len(gene_names) - 1)
    
    # Skip if it picks the same gene for both
    if action_promoter == action_repressor: continue
        
    action_threshold = random.uniform(0.5, 3.0) 
    
    # Step: Agent tests the action in the Environment
    reward, tp, fp = calculate_reward(action_promoter, action_repressor, action_threshold)
    
    # Learn: If this circuit got a higher score, save it as the new State-of-the-Art
    if reward > best_reward:
        best_reward = reward
        best_circuit = {
            'Promoter': gene_names[action_promoter],
            'Repressor': gene_names[action_repressor],
            'Threshold': action_threshold,
            'Cancer_Killed': tp,
            'Friendly_Fire': fp
        }
        # Print progress when the agent finds a breakthrough
        if episode % 50 == 0 or reward > 0:
            print(f"Episode {episode} | Breakthrough! New Reward: {reward:.1f} | "
                  f"Killed: {tp} | Friendly Fire: {fp}")

# ==========================================
# MISSION REPORT
# ==========================================
print("\n========================================")
print("     RL AGENT: FINAL CIRCUIT BLUEPRINT    ")
print("========================================")
print(f"Promoter Sensor (+): {best_circuit['Promoter']}")
print(f"Repressor Sensor (-): {best_circuit['Repressor']}")
print(f"Activation Threshold: {best_circuit['Threshold']:.2f}")
print("--- Safety Report ---")
cancer_total = sum(y_target == 1)
healthy_total = sum(y_target == 0)
print(f"Tumor Cells Destroyed: {best_circuit['Cancer_Killed']} / {cancer_total}")
print(f"Healthy Immune Cells Destroyed: {best_circuit['Friendly_Fire']} / {healthy_total} (Friendly Fire)")