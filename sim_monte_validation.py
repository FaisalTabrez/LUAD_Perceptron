import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. BASELINE PARAMETERS & SETTINGS
# ==========================================
# These are our "assumed" ideal biophysical constants
base_params = {
    'alpha': 50.0,  # Max production
    'gamma': 0.1,   # Degradation
    'K_A': 40.0,    # miR-210 threshold
    'K_R': 40.0     # miR-486 threshold
}
n_hill = 2.0
lethal_threshold = 150.0

# Cell expressions from TCGA findings
cancer_miR = {'210': 120.0, '486': 5.0}
healthy_miR = {'210': 10.0,  '486': 150.0}

# Simulation settings
time_hours = np.linspace(0, 48, 500)
num_simulations = 200  # We will test 200 different random biochemical environments
variance = 0.20        # Introduce +/- 20% random error to every constant

# ==========================================
# 2. THE ODE FUNCTION
# ==========================================
def perceptron_circuit(P, t, alpha, gamma, K_A, K_R, miR_210, miR_486):
    H_A = (miR_210**n_hill) / (K_A**n_hill + miR_210**n_hill)
    H_R = (K_R**n_hill) / (K_R**n_hill + miR_486**n_hill)
    return (alpha * H_A * H_R) - (gamma * P)

# ==========================================
# 3. RUN MONTE CARLO SIMULATIONS
# ==========================================
plt.figure(figsize=(11, 7))

cancer_successes = 0
healthy_successes = 0

print(f"Running {num_simulations} simulations with +/- {variance*100}% parameter noise...\n")

for i in range(num_simulations):
    # Randomly perturb every single parameter by up to 20%
    rand_alpha = np.random.uniform(base_params['alpha'] * (1 - variance), base_params['alpha'] * (1 + variance))
    rand_gamma = np.random.uniform(base_params['gamma'] * (1 - variance), base_params['gamma'] * (1 + variance))
    rand_KA    = np.random.uniform(base_params['K_A'] * (1 - variance), base_params['K_A'] * (1 + variance))
    rand_KR    = np.random.uniform(base_params['K_R'] * (1 - variance), base_params['K_R'] * (1 + variance))
    
    # Simulate Cancer Cell with scrambled params
    P_cancer = odeint(perceptron_circuit, 0.0, time_hours, 
                      args=(rand_alpha, rand_gamma, rand_KA, rand_KR, cancer_miR['210'], cancer_miR['486'])).flatten()
    
    # Simulate Healthy Cell with scrambled params
    P_healthy = odeint(perceptron_circuit, 0.0, time_hours, 
                       args=(rand_alpha, rand_gamma, rand_KA, rand_KR, healthy_miR['210'], healthy_miR['486'])).flatten()
    
    # Check if the circuit behaved correctly at the 48-hour mark
    if P_cancer[-1] >= lethal_threshold: cancer_successes += 1
    if P_healthy[-1] < lethal_threshold: healthy_successes += 1
    
    # Plot the random trajectories as semi-transparent lines
    # Only label the first line to avoid 400 legend entries
    plt.plot(time_hours, P_cancer, color='crimson', alpha=0.05, label='Cancer Cell (Noisy)' if i==0 else "")
    plt.plot(time_hours, P_healthy, color='mediumseagreen', alpha=0.05, label='Healthy Cell (Noisy)' if i==0 else "")

# ==========================================
# 4. PLOT FORMATTING & RESULTS
# ==========================================
# Draw the lethal threshold line
plt.axhline(y=lethal_threshold, color='black', linestyle='--', linewidth=2, label='Lethal Threshold (Apoptosis)')

plt.title(f'Monte Carlo Sensitivity Analysis (±{variance*100}% Parameter Variance)', fontsize=15, fontweight='bold')
plt.xlabel('Time (Hours)', fontsize=13)
plt.ylabel('Killer Protein Concentration (nM)', fontsize=13)
plt.ylim(0, 450)
plt.xlim(0, 48)

# Legend formatting
leg = plt.legend(loc='upper left', fontsize=11)
for lh in leg.legend_handles: 
    lh.set_alpha(1) # Make the legend solid, not transparent

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 5. PRINT QUANTITATIVE ROBUSTNESS METRICS
# ==========================================
cancer_robustness = (cancer_successes / num_simulations) * 100
healthy_robustness = (healthy_successes / num_simulations) * 100

print("========================================")
print("     CIRCUIT ROBUSTNESS REPORT          ")
print("========================================")
print(f"Cancer Cell Apoptosis Success Rate: {cancer_robustness:.1f}%")
print(f"Healthy Cell Protection Success Rate: {healthy_robustness:.1f}%")
print("========================================")
print("Conclusion: Even with 20% random error in promoter strength, degradation")
print("rates, and binding thresholds, the logic gate fundamentally holds.")