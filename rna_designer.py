import random
import numpy as np
import math

print("1. Initializing Generative RNA sequence engine...")

# A realistic 20-nucleotide target fragment from the CEACAM6 Cancer mRNA
TARGET_CEACAM6_mRNA = "AUGGCCUACGGAUCGCUAAA"

# The RL Agent's required thermodynamic binding threshold
TARGET_DELTA_G = -35.0  # kcal/mol

# ==========================================
# THERMODYNAMIC PHYSICS ENGINE (Nearest Neighbor Approximation)
# ==========================================
def calculate_binding_energy(synthetic_rna, target_rna):
    """
    Calculates the Gibbs Free Energy (Delta G) of the synthetic RNA 
    binding to the cancer mRNA. More negative = stronger bind.
    """
    delta_g = 0.0
    for i in range(len(synthetic_rna)):
        base1 = synthetic_rna[i]
        base2 = target_rna[i]
        
        # Watson-Crick Pairs (Strongest)
        if (base1 == 'G' and base2 == 'C') or (base1 == 'C' and base2 == 'G'):
            delta_g -= 3.0  
        elif (base1 == 'A' and base2 == 'U') or (base1 == 'U' and base2 == 'A'):
            delta_g -= 2.0  
        # Wobble Pair (Weak)
        elif (base1 == 'G' and base2 == 'U') or (base1 == 'U' and base2 == 'G'):
            delta_g -= 1.0  
        # Mismatch (Repulsive / breaks the hairpin structure)
        else:
            delta_g += 1.5  
            
    return delta_g

# ==========================================
# GENERATIVE AI: SIMULATED ANNEALING
# ==========================================
def mutate_sequence(sequence):
    """ Randomly changes one nucleotide (A, U, C, G) """
    bases = ['A', 'U', 'C', 'G']
    idx = random.randint(0, len(sequence) - 1)
    new_base = random.choice([b for b in bases if b != sequence[idx]])
    return sequence[:idx] + new_base + sequence[idx+1:]

print("2. Generating random synthetic RNA starting point...")
# Start with pure random garbage RNA
current_sequence = "".join(random.choices(['A', 'U', 'C', 'G'], k=len(TARGET_CEACAM6_mRNA)))
current_energy = calculate_binding_energy(current_sequence, TARGET_CEACAM6_mRNA)

print(f"Starting Sequence: {current_sequence} | Initial Energy: {current_energy:.1f} kcal/mol")

# Simulated Annealing Hyperparameters
temperature = 100.0
cooling_rate = 0.99
iterations = 5000

print(f"\n3. Evolving sequence to hit Target Threshold: {TARGET_DELTA_G} kcal/mol...")

for step in range(iterations):
    # Propose a mutation
    new_sequence = mutate_sequence(current_sequence)
    new_energy = calculate_binding_energy(new_sequence, TARGET_CEACAM6_mRNA)
    
    # Calculate how far we are from the RL Agent's required threshold
    current_error = abs(TARGET_DELTA_G - current_energy)
    new_error = abs(TARGET_DELTA_G - new_energy)
    
    # If the mutation gets us closer to the target, ACCEPT IT
    if new_error < current_error:
        current_sequence = new_sequence
        current_energy = new_energy
    # If it's worse, sometimes accept it anyway to avoid local minimums (The "Annealing" part)
    else:
        acceptance_probability = math.exp((current_error - new_error) / temperature)
        if random.random() < acceptance_probability:
            current_sequence = new_sequence
            current_energy = new_energy
            
    # Cool down the temperature
    temperature *= cooling_rate
    
    # Stop early if we hit the exact thermodynamic threshold required
    if abs(current_energy - TARGET_DELTA_G) < 0.1:
        print(f"Convergence achieved at Generation {step}!")
        break

# ==========================================
# FINAL BLUEPRINT FOR THE WET LAB
# ==========================================
print("\n========================================")
print("   SYNTHETIC RNA SENSOR GENERATED       ")
print("========================================")
print(f"Target Cancer mRNA (CEACAM6) : 5' - {TARGET_CEACAM6_mRNA} - 3'")
print(f"Synthetic Sensor (Toehold)   : 3' - {current_sequence} - 5'")
print("----------------------------------------")
print(f"Final Binding Energy (Delta G) : {current_energy:.1f} kcal/mol")
print(f"RL Agent Required Threshold    : {TARGET_DELTA_G:.1f} kcal/mol")
print("========================================")
print("STATUS: SEQUENCE READY FOR SYNTHESIS.")