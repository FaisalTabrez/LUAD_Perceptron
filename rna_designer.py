import random
import math

print("1. Initializing Generative RNA sequence engine for EPCAM...")

TARGET_mRNA = "AUGGCGCCCCCGCAGGUCUC"
TARGET_DELTA_G = -35.0  # kcal/mol

def calculate_binding_energy(synthetic_rna, target_rna):
    delta_g = 0.0
    for i in range(len(synthetic_rna)):
        base1 = synthetic_rna[i]
        base2 = target_rna[i]
        
        if (base1 == 'G' and base2 == 'C') or (base1 == 'C' and base2 == 'G'): delta_g -= 3.0  
        elif (base1 == 'A' and base2 == 'U') or (base1 == 'U' and base2 == 'A'): delta_g -= 2.0  
        elif (base1 == 'G' and base2 == 'U') or (base1 == 'U' and base2 == 'G'): delta_g -= 1.0  
        else: delta_g += 1.5  
    return delta_g

def mutate_sequence(sequence):
    bases =['A', 'U', 'C', 'G']
    idx = random.randint(0, len(sequence) - 1)
    new_base = random.choice([b for b in bases if b != sequence[idx]])
    return sequence[:idx] + new_base + sequence[idx+1:]

current_sequence = "".join(random.choices(['A', 'U', 'C', 'G'], k=len(TARGET_mRNA)))
current_energy = calculate_binding_energy(current_sequence, TARGET_mRNA)

temperature = 100.0
cooling_rate = 0.99
iterations = 5000

print(f"3. Evolving sequence to hit Target Threshold: {TARGET_DELTA_G} kcal/mol...")

for step in range(iterations):
    new_sequence = mutate_sequence(current_sequence)
    new_energy = calculate_binding_energy(new_sequence, TARGET_mRNA)
    
    current_error = abs(TARGET_DELTA_G - current_energy)
    new_error = abs(TARGET_DELTA_G - new_energy)
    
    if new_error < current_error:
        current_sequence = new_sequence
        current_energy = new_energy
    else:
        acceptance_probability = math.exp((current_error - new_error) / temperature)
        if random.random() < acceptance_probability:
            current_sequence = new_sequence
            current_energy = new_energy
            
    temperature *= cooling_rate
    
    if abs(current_energy - TARGET_DELTA_G) < 0.1:
        print(f"Convergence achieved at Generation {step}!")
        break

print("\n========================================")
print("   SYNTHETIC RNA SENSOR GENERATED       ")
print("========================================")
print(f"Target Cancer mRNA (EPCAM) : 5' - {TARGET_mRNA} - 3'")
print(f"Synthetic Sensor (Toehold) : 3' - {current_sequence} - 5'")
print("----------------------------------------")
print(f"Final Binding Energy (Delta G) : {current_energy:.1f} kcal/mol")
print("========================================")