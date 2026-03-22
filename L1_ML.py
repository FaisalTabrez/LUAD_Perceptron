import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

print("1. Loading and cleaning TCGA-LUAD dataset...")
# --- PART 1: DATA LOADING & CLEANING ---
mirna_data = pd.read_csv('TCGA-LUAD.mirna.tsv', sep='\t', index_col=0)
df = mirna_data.T 

# Create the Target Column by reading the TCGA Barcode
def get_label(barcode):
    try:
        sample_code = str(barcode).split('-')[3] 
        if sample_code.startswith('0'):
            return 1  # 1 = Cancer
        elif sample_code.startswith('1'):
            return 0  # 0 = Healthy
        else:
            return np.nan
    except:
        return np.nan

df['Target'] =[get_label(idx) for idx in df.index]
df = df.dropna(subset=['Target'])

# Biological Pre-Filter (Keep abundant miRNAs)
miRNA_columns = df.columns.drop('Target')
mean_expression = df[miRNA_columns].mean()
abundant_miRNAs = mean_expression[mean_expression > 1.0].index
df = df[abundant_miRNAs.tolist() + ['Target']]

print(f"Data ready: {df.shape[0]} patients, {df.shape[1]-1} miRNAs.")
print("2. Hunting for the perfect 3-5 miRNAs...\n")

# --- PART 2: MACHINE LEARNING & CIRCUIT DESIGN ---
X = df.drop(columns=['Target'])
y = df['Target']

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

best_model = None
best_features = []
best_weights =[]
best_bias = 0

# Auto-Tuning Loop for L1 Regularization
for c_val in np.logspace(-3, 1, 200): 
    model = LogisticRegression(penalty='l1', solver='liblinear', C=c_val, 
                               class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    coefs = model.coef_[0]
    surviving_count = np.sum(coefs != 0)
    
    if 3 <= surviving_count <= 5:
        best_model = model
        surviving_indices = np.where(coefs != 0)[0]
        best_features = X.columns[surviving_indices].tolist()
        best_weights = coefs[surviving_indices]
        best_bias = model.intercept_[0]
        break

# --- PART 3: RESULTS OUTPUT ---
if best_model is not None:
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print("========================================")
    print("      CIRCUIT BLUEPRINT DISCOVERED      ")
    print("========================================")
    print(f"Theoretical Accuracy: {acc * 100:.1f}%\n")
    print(f"Test Set Performance:")
    print(f"- Healthy Cells: {tn} survived, {fp} accidentally killed (False Positives)")
    print(f"- Cancer Cells:  {tp} successfully killed, {fn} escaped (False Negatives)\n")
    
    print("--- BIOLOGICAL PARTS LIST ---")
    print(f"Mathematical Bias (Threshold K): {best_bias:.4f}\n")
    
    for feature, weight in zip(best_features, best_weights):
        part_type = "PROMOTER (+) - Triggers Death" if weight > 0 else "REPRESSOR (-) - Protects Cell"
        print(f"Sensor: {feature}")
        print(f"Weight: {weight:.4f}")
        print(f"Biological Part: {part_type}\n")
else:
    print("Could not find a stable circuit with 3-5 inputs. The algorithm couldn't find a clean separation.")