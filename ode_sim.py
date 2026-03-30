import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 1. BIOPHYSICAL PARAMETERS

alpha = 50.0      # Maximum production rate of Killer Protein (nM per hour)
gamma = 0.1       # Degradation rate of the protein (Proteins decay over time)
n = 2.0           # Hill coefficient (Cooperativity of the genetic parts)
K_A = 40.0        # Dissociation constant (Threshold) for miR-210 sensor
K_R = 40.0        # Dissociation constant (Threshold) for miR-486 sensor

lethal_threshold = 150.0  # conc required to trigger Apoptosis 


# 2. THE ORDINARY DIFFERENTIAL EQUATION (ODE)

def perceptron_circuit(P, t, miR_210, miR_486):
    """
    Calculates the change in Killer Protein concentration over time.
    dP/dt = (Production Rate * Activator_Gate * Repressor_Gate) - Degradation
    """
    # Activating Hill Equation (miR-210)
    H_A = (miR_210**n) / (K_A**n + miR_210**n)
    
    # Repressing Hill Equation (miR-486)
    H_R = (K_R**n) / (K_R**n + miR_486**n)
    
    # The Physics: Synthesis minus Decay
    dP_dt = (alpha * H_A * H_R) - (gamma * P)
    return dP_dt


# 3. VIRTUAL CELL PROFILES

# Typical expression levels (relative units) extracted from TCGA
cancer_cell  = {'miR_210': 120.0, 'miR_486': 5.0}   # Suffocating, lost tumor suppressor
healthy_cell = {'miR_210': 10.0,  'miR_486': 150.0} # Normal oxygen, high tumor suppressor

# Time vector: Simulate from Hour 0 to Hour 48, taking 1000 steps
time_hours = np.linspace(0, 48, 1000)
initial_protein = 0.0  # At hour 0, the circuit hasn't produced anything yet

# ==========================================
# 4. RUN THE BIOCHEMICAL SIMULATION
# ==========================================
# Simulate Cancer Cell
cancer_protein = odeint(perceptron_circuit, initial_protein, time_hours, 
                        args=(cancer_cell['miR_210'], cancer_cell['miR_486']))

# Simulate Healthy Cell
healthy_protein = odeint(perceptron_circuit, initial_protein, time_hours, 
                         args=(healthy_cell['miR_210'], healthy_cell['miR_486']))

# ==========================================
# 5. VISUALIZATION (Publication Ready Graph)
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the protein curves
plt.plot(time_hours, cancer_protein, color='crimson', linewidth=3, label='Lung Cancer Cell')
plt.plot(time_hours, healthy_protein, color='mediumseagreen', linewidth=3, label='Healthy Lung Cell')

# Draw the Apoptosis (Death) Threshold
plt.axhline(y=lethal_threshold, color='black', linestyle='--', linewidth=2, label='Lethal Threshold (Apoptosis)')

# Formatting the graph
plt.title('Biophysical Simulation of the Cellular Perceptron over 48 Hours', fontsize=14, fontweight='bold')
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('Killer Protein Concentration (nM)', fontsize=12)
plt.ylim(0, 400)
plt.xlim(0, 48)
plt.legend(loc='upper left', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()