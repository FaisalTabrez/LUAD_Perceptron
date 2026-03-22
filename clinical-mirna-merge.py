import pandas as pd
import numpy as np

# 1. Load ONLY the miRNA data (We don't need the phenotype file!)
mirna_data = pd.read_csv('TCGA-LUAD.mirna.tsv', sep='\t', index_col=0)

# 2. Transpose so Patients are Rows and miRNAs are Columns
df = mirna_data.T 

# 3. Create the Target Column by reading the TCGA Barcode
def get_label(barcode):
    # Splits 'TCGA-44-6147-01A' and grabs the '01A' part
    sample_code = str(barcode).split('-')[3] 
    
    if sample_code.startswith('0'):
        return 1  # 1 = Cancer (Codes 01-09 are tumors)
    elif sample_code.startswith('1'):
        return 0  # 0 = Healthy (Codes 10-19 are normal tissue)
    else:
        return np.nan

df['Target'] =[get_label(idx) for idx in df.index]

# Drop any weird rows that aren't strictly 1 or 0
df = df.dropna(subset=['Target'])


miRNA_columns = df.columns.drop('Target')
mean_expression = df[miRNA_columns].mean()


abundant_miRNAs = mean_expression[mean_expression > 1.0].index
df = df[abundant_miRNAs.tolist() +['Target']]

print("--- Data Processing Complete ---")
print(f"Original miRNAs: {len(miRNA_columns)}")
print(f"Abundant miRNAs kept for Circuit Design: {len(abundant_miRNAs)}")
print("\nFinal Dataset Shape:", df.shape)
print("\nCell Counts:")
print(df['Target'].value_counts().rename(index={1.0: "Cancer Cells (1)", 0.0: "Healthy Cells (0)"}))