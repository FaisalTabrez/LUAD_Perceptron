"""
PHASE 1: ROBUST VALIDATION OF L1 LASSO BIOMARKER DISCOVERY
==========================================================
Addresses peer review feedback: requires proper cross-validation metrics,
not just training set accuracy.

This script:
1. Stratified 5-fold cross-validation (ROC-AUC, sensitivity, specificity, F1)
2. Bootstrap weight stability (1000 resamplings, violin plot)
3. Cross-validated ROC curve with 95% confidence intervals
4. Aggregate confusion matrix across all folds
5. Baseline comparisons (dummy classifier, single-miRNA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
    recall_score, f1_score, auc
)
from sklearn.dummy import DummyClassifier

# ============================================================================
# BIOLOGICAL CONSTANTS (from copilot-instructions.md)
# ============================================================================
K_A = 40.0                          # nM, activation threshold
K_R = 40.0                          # nM, repression threshold
LETHAL_THRESHOLD = 150.0            # nM, killer protein threshold

# Create results directory
RESULTS_DIR = Path('results/phase1_validation')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

df['Target'] = [get_label(idx) for idx in df.index]
df = df.dropna(subset=['Target'])

# Biological Pre-Filter (Keep abundant miRNAs)
miRNA_columns = df.columns.drop('Target')
mean_expression = df[miRNA_columns].mean()
abundant_miRNAs = mean_expression[mean_expression > 1.0].index
df = df[abundant_miRNAs.tolist() + ['Target']]

print(f"Data ready: {df.shape[0]} patients, {df.shape[1]-1} miRNAs.")

X = df.drop(columns=['Target'])
y = df['Target']

n_cancer = np.sum(y == 1)
n_healthy = np.sum(y == 0)
print(f"Class distribution: {n_cancer} cancer (class 1), {n_healthy} healthy (class 0)")
print("="*80)


# ============================================================================
# HELPER FUNCTION: Find stable L1 circuit
# ============================================================================
def find_stable_l1_circuit(X_fold, y_fold):
    """
    Search for L1 model with 3-5 non-zero miRNA weights.
    
    BIOLOGY: We want a circuit with 2-3 miRNA sensors to avoid overfitting
    while maintaining sufficient complexity for specificity.
    """
    best_model = None
    best_features = []
    best_weights = []
    best_bias = 0
    
    for c_val in np.logspace(-3, 1, 200):
        model = LogisticRegression(
            penalty='l1', solver='liblinear', C=c_val,
            class_weight='balanced', random_state=42, max_iter=1000
        )
        model.fit(X_fold, y_fold)
        
        coefs = model.coef_[0]
        surviving_count = np.sum(coefs != 0)
        
        if 3 <= surviving_count <= 5:
            best_model = model
            surviving_indices = np.where(coefs != 0)[0]
            best_features = X_fold.columns[surviving_indices].tolist()
            best_weights = coefs[surviving_indices]
            best_bias = model.intercept_[0]
            break
    
    return best_model, best_features, best_weights, best_bias


# ============================================================================
# PART 2: STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================================
print("\n[2/5] Running Stratified 5-Fold Cross-Validation...")
print("-" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Storage for metrics across folds
fold_results = {
    'fold': [],
    'roc_auc': [],
    'sensitivity': [],      # Recall for cancer (class 1)
    'specificity': [],      # Recall for healthy (class 0)
    'f1_score': [],
    'accuracy': [],
    'tp': [],
    'tn': [],
    'fp': [],
    'fn': [],
}

# Storage for circuit weights across folds
fold_circuits = {
    'fold': [],
    'features': [],
    'weights': [],
    'bias': [],
    'n_features': [],
}

# Storage for cross-validated predictions (for ROC curve)
y_test_all = []
y_pred_proba_all = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold_idx}/5:")
    print(f"  Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")
    
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Find stable circuit
    model, features, weights, bias = find_stable_l1_circuit(X_train_fold, y_train_fold)
    
    if model is None:
        print(f"  ⚠ WARNING: Could not find stable circuit in fold {fold_idx}")
        continue
    
    # Predictions
    y_pred = model.predict(X_test_fold)
    y_pred_proba = model.predict_proba(X_test_fold)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test_fold, y_pred_proba)
    sensitivity = recall_score(y_test_fold, y_pred, pos_label=1)
    specificity = recall_score(y_test_fold, y_pred, pos_label=0)
    f1 = f1_score(y_test_fold, y_pred, pos_label=1)
    acc = accuracy_score(y_test_fold, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()
    
    # Store results
    fold_results['fold'].append(fold_idx)
    fold_results['roc_auc'].append(roc_auc)
    fold_results['sensitivity'].append(sensitivity)
    fold_results['specificity'].append(specificity)
    fold_results['f1_score'].append(f1)
    fold_results['accuracy'].append(acc)
    fold_results['tp'].append(tp)
    fold_results['tn'].append(tn)
    fold_results['fp'].append(fp)
    fold_results['fn'].append(fn)
    
    fold_circuits['fold'].append(fold_idx)
    fold_circuits['features'].append(features)
    fold_circuits['weights'].append(weights)
    fold_circuits['bias'].append(bias)
    fold_circuits['n_features'].append(len(features))
    
    # Concatenate for cross-validated ROC
    y_test_all.extend(y_test_fold)
    y_pred_proba_all.extend(y_pred_proba)
    
    print(f"  ROC-AUC: {roc_auc:.4f} | Sensitivity: {sensitivity:.4f} | "
          f"Specificity: {specificity:.4f} | F1: {f1:.4f}")
    print(f"  Circuit: {len(features)} sensors - {', '.join(features)}")


# Convert to arrays for ROC curve
y_test_all = np.array(y_test_all)
y_pred_proba_all = np.array(y_pred_proba_all)

# Summary statistics
print("\n" + "="*80)
print("CROSS-VALIDATION SUMMARY")
print("="*80)

metrics_summary = pd.DataFrame(fold_results)

print("\nPer-Fold Results:")
print(metrics_summary.to_string(index=False))

print("\n\nAggregate Metrics (Mean ± Std across 5 folds):")
print("-" * 80)
print(f"ROC-AUC:       {np.mean(fold_results['roc_auc']):.4f} ± {np.std(fold_results['roc_auc']):.4f}")
print(f"Sensitivity:   {np.mean(fold_results['sensitivity']):.4f} ± {np.std(fold_results['sensitivity']):.4f}")
print(f"Specificity:   {np.mean(fold_results['specificity']):.4f} ± {np.std(fold_results['specificity']):.4f}")
print(f"F1 Score:      {np.mean(fold_results['f1_score']):.4f} ± {np.std(fold_results['f1_score']):.4f}")
print(f"Accuracy:      {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")

# Aggregate confusion matrix
agg_tp = np.sum(fold_results['tp'])
agg_tn = np.sum(fold_results['tn'])
agg_fp = np.sum(fold_results['fp'])
agg_fn = np.sum(fold_results['fn'])

print("\nAggregate Confusion Matrix (across all folds):")
print("-" * 80)
print(f"TP (Cancer killed):     {agg_tp}")
print(f"TN (Healthy survived):  {agg_tn}")
print(f"FP (Healthy killed):    {agg_fp}")
print(f"FN (Cancer escaped):    {agg_fn}")
print(f"\nTotal Predictions: {agg_tp + agg_tn + agg_fp + agg_fn}")
print(f"Overall Accuracy: {(agg_tp + agg_tn) / (agg_tp + agg_tn + agg_fp + agg_fn):.4f}")


# ============================================================================
# PART 3: COMPARISON BASELINES
# ============================================================================
print("\n" + "="*80)
print("BASELINE COMPARISON ANALYSIS")
print("="*80)

# Baseline 1: Dummy Classifier (always predict majority class)
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_auc_scores = []
dummy_sens_scores = []
dummy_spec_scores = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    
    # For dummy, AUC is meaningless but we'll use accuracy-based metric
    sens = recall_score(y_test, y_pred_dummy, pos_label=1, zero_division=0)
    spec = recall_score(y_test, y_pred_dummy, pos_label=0, zero_division=0)
    
    dummy_sens_scores.append(sens)
    dummy_spec_scores.append(spec)

print("\n[Baseline 1] Dummy Classifier (always majority class):")
print(f"  Sensitivity (mean): {np.mean(dummy_sens_scores):.4f}")
print(f"  Specificity (mean): {np.mean(dummy_spec_scores):.4f}")

# Baseline 2: Single-miRNA classifier (hsa-miR-210 only)
if 'hsa-miR-210' in X.columns:
    mir210_auc_scores = []
    mir210_sens_scores = []
    mir210_spec_scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train_mir210 = X[['hsa-miR-210']].iloc[train_idx]
        X_test_mir210 = X[['hsa-miR-210']].iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        model_mir210 = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        model_mir210.fit(X_train_mir210, y_train)
        
        y_pred_mir210 = model_mir210.predict(X_test_mir210)
        y_pred_proba_mir210 = model_mir210.predict_proba(X_test_mir210)[:, 1]
        
        auc_mir210 = roc_auc_score(y_test, y_pred_proba_mir210)
        sens_mir210 = recall_score(y_test, y_pred_mir210, pos_label=1)
        spec_mir210 = recall_score(y_test, y_pred_mir210, pos_label=0)
        
        mir210_auc_scores.append(auc_mir210)
        mir210_sens_scores.append(sens_mir210)
        mir210_spec_scores.append(spec_mir210)
    
    print("\n[Baseline 2] Single-miRNA Classifier (hsa-miR-210 only):")
    print(f"  ROC-AUC:       {np.mean(mir210_auc_scores):.4f} ± {np.std(mir210_auc_scores):.4f}")
    print(f"  Sensitivity:   {np.mean(mir210_sens_scores):.4f} ± {np.std(mir210_sens_scores):.4f}")
    print(f"  Specificity:   {np.mean(mir210_spec_scores):.4f} ± {np.std(mir210_spec_scores):.4f}")
    
    baseline2_auc = np.mean(mir210_auc_scores)
else:
    print("\n[Baseline 2] hsa-miR-210 not found in dataset")
    baseline2_auc = 0

# L1 Circuit improvement
l1_auc = np.mean(fold_results['roc_auc'])
improvement = l1_auc - baseline2_auc
improvement_pct = (improvement / baseline2_auc * 100) if baseline2_auc > 0 else 0

print(f"\n[L1 Multi-Sensor Circuit]:")
print(f"  ROC-AUC:       {l1_auc:.4f} ± {np.std(fold_results['roc_auc']):.4f}")
print(f"  Improvement over single-miRNA: +{improvement:.4f} ({improvement_pct:.1f}%)")


# ============================================================================
# PART 4: BOOTSTRAP WEIGHT STABILITY (1000 RESAMPLINGS)
# ============================================================================
print("\n" + "="*80)
print("BOOTSTRAP WEIGHT STABILITY ANALYSIS")
print("="*80)
print("\n[3/5] Running 1000 bootstrap resamplings (this may take ~2-3 minutes)...")

N_BOOTSTRAP = 1000
bootstrap_weights = {}  # {miRNA_name: [list of 1000 weights]}
bootstrap_frequencies = {}  # {miRNA_name: frequency of non-zero}

for bootstrap_idx in range(N_BOOTSTRAP):
    # Resample training set with replacement
    resample_indices = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[resample_indices]
    y_boot = y.iloc[resample_indices]
    
    # Fit L1 model
    for c_val in np.logspace(-3, 1, 200):
        model_boot = LogisticRegression(
            penalty='l1', solver='liblinear', C=c_val,
            class_weight='balanced', random_state=bootstrap_idx, max_iter=1000
        )
        model_boot.fit(X_boot, y_boot)
        
        coefs_boot = model_boot.coef_[0]
        if 3 <= np.sum(coefs_boot != 0) <= 5:
            # Record weights
            for mirna_idx, mirna_name in enumerate(X.columns):
                weight = coefs_boot[mirna_idx]
                
                if mirna_name not in bootstrap_weights:
                    bootstrap_weights[mirna_name] = []
                    bootstrap_frequencies[mirna_name] = 0
                
                bootstrap_weights[mirna_name].append(weight)
                
                if weight != 0:
                    bootstrap_frequencies[mirna_name] += 1
            
            break
    
    if (bootstrap_idx + 1) % 200 == 0:
        print(f"  Completed {bootstrap_idx + 1}/{N_BOOTSTRAP} bootstraps...")

print(f"  Completed {N_BOOTSTRAP}/{N_BOOTSTRAP} bootstraps ✓")

# ANALYSIS: Which miRNAs are stable?
print("\nmiRNA Stability (frequency of non-zero weight across 1000 bootstraps):")
print("-" * 80)

stability_results = []
for mirna_name in sorted(bootstrap_frequencies.keys(), 
                        key=lambda x: bootstrap_frequencies[x], reverse=True):
    freq = bootstrap_frequencies[mirna_name]
    pct = (freq / N_BOOTSTRAP) * 100
    
    if freq >= 50:  # Report only miRNAs selected >= 5% of the time
        mean_weight = np.nanmean([w for w in bootstrap_weights[mirna_name] if w != 0]) if freq > 0 else 0
        std_weight = np.nanstd([w for w in bootstrap_weights[mirna_name] if w != 0]) if freq > 0 else 0
        
        stability_results.append({
            'miRNA': mirna_name,
            'Frequency': freq,
            'Percentage': pct,
            'Mean_Weight': mean_weight,
            'Std_Weight': std_weight,
        })
        
        print(f"{mirna_name:20s} | {freq:4d}/{N_BOOTSTRAP} ({pct:5.1f}%) | "
              f"Weight: {mean_weight:+.4f} ± {std_weight:.4f}")

stability_df = pd.DataFrame(stability_results)


# ============================================================================
# PART 5: VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
print("="*80)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- VISUALIZATION 1: ROC Curve (Cross-Validated) ---
print("\n[4/5] Creating ROC curve with 95% confidence intervals...")

fpr, tpr, thresholds = roc_curve(y_test_all, y_pred_proba_all)
roc_auc_cv = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot mean ROC curve
ax.plot(fpr, tpr, color='darkorange', lw=2.5, 
        label=f'Mean ROC-AUC = {roc_auc_cv:.3f}')

# Plot 95% confidence band (computed from fold-wise predictions)
fprs = []
tprs = []
for train_idx, test_idx in skf.split(X, y):
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Refit model for this fold
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    
    model_fold, _, _, _ = find_stable_l1_circuit(X_train_fold, y_train_fold)
    if model_fold is not None:
        y_pred_proba_fold = model_fold.predict_proba(X_test_fold)[:, 1]
        fpr_fold, tpr_fold, _ = roc_curve(y_test_fold, y_pred_proba_fold)
        fprs.append(fpr_fold)
        tprs.append(tpr_fold)

# Interpolate all ROC curves to common FPR scale
mean_fpr = np.linspace(0, 1, 100)
tprs_interp = []
for tpr_fold in tprs:
    fpr_fold = np.linspace(0, len(tpr_fold) - 1, len(tpr_fold)) / (len(tpr_fold) - 1) if len(tpr_fold) > 1 else np.array([0, 1])
    tprs_interp.append(np.interp(mean_fpr, fpr_fold, tpr_fold))

tprs_interp = np.array(tprs_interp)
mean_tpr = np.mean(tprs_interp, axis=0)
std_tpr = np.std(tprs_interp, axis=0)

ci_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1.0)
ci_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0.0)

ax.fill_between(mean_fpr, ci_lower, ci_upper, alpha=0.2, color='darkorange',
                label='95% Confidence Interval')

# Diagonal
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC=0.5)')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
ax.set_title('Cross-Validated ROC Curve: L1 Lasso Cancer Detection', 
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

roc_path = RESULTS_DIR / f"roc_curve_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {roc_path}")
plt.close()


# --- VISUALIZATION 2: Violin Plot (Bootstrap Weight Distributions) ---
print("\n[5/5] Creating bootstrap weight stability violin plots...")

# Prepare data for violin plot
if stability_df is not None and len(stability_df) > 0:
    selected_mirnas = [row['miRNA'] for _, row in stability_df.head(6).iterrows()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create violin plot
    data_by_mirna = [bootstrap_weights[m] for m in selected_mirnas]
    parts = ax.violinplot(data_by_mirna, positions=range(len(selected_mirnas)), 
                           showmeans=True, showmedians=True)
    
    ax.set_xticks(range(len(selected_mirnas)))
    ax.set_xticklabels(selected_mirnas, rotation=45, ha='right')
    ax.set_ylabel('L1 Coefficient Weight', fontsize=11, fontweight='bold')
    ax.set_title('Bootstrap Weight Stability: Distribution of L1 Coefficients\n(1000 bootstrap resamplings)', 
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero Coefficient')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    violin_path = RESULTS_DIR / f"bootstrap_weights_violin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {violin_path}")
    plt.close()


# --- VISUALIZATION 3: Confusion Matrix (Aggregate) ---
fig, ax = plt.subplots(figsize=(8, 7))

cm = np.array([[agg_tn, agg_fp], [agg_fn, agg_tp]])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted Healthy', 'Predicted Cancer'],
            yticklabels=['Actually Healthy', 'Actually Cancer'],
            ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'fontweight': 'bold'})

ax.set_title('Aggregate Confusion Matrix (5-Fold Cross-Validation)', 
             fontsize=12, fontweight='bold', pad=15)
ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

# Add text annotations for rates
sensitivity_rate = agg_tp/(agg_tp+agg_fn) if (agg_tp+agg_fn) > 0 else 0
specificity_rate = agg_tn/(agg_tn+agg_fp) if (agg_tn+agg_fp) > 0 else 0
sensitivity_text = f"Sensitivity: {sensitivity_rate:.3f}"
specificity_text = f"Specificity: {specificity_rate:.3f}"
fig.text(0.5, 0.02, f"{sensitivity_text}  |  {specificity_text}", 
         ha='center', fontsize=10, fontweight='bold')

cm_path = RESULTS_DIR / f"confusion_matrix_aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {cm_path}")
plt.close()


# --- VISUALIZATION 4: Metrics Comparison Plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Metrics by fold
metrics_for_plot = ['roc_auc', 'sensitivity', 'specificity', 'f1_score', 'accuracy']
folds_x = fold_results['fold']

ax = axes[0]
for metric in metrics_for_plot:
    ax.plot(folds_x, fold_results[metric], marker='o', linewidth=2, label=metric.replace('_', ' ').title())

ax.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
ax.set_ylabel('Metric Score', fontsize=11, fontweight='bold')
ax.set_title('Cross-Validation Metrics by Fold', fontsize=11, fontweight='bold')
ax.set_xticks(folds_x)
ax.set_ylim([0, 1.05])
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

# Right: Mean metrics with error bars
metric_names = ['ROC-AUC', 'Sensitivity', 'Specificity', 'F1 Score', 'Accuracy']
means = [np.mean(fold_results[m]) for m in metrics_for_plot]
stds = [np.std(fold_results[m]) for m in metrics_for_plot]

ax = axes[1]
x_pos = np.arange(len(metric_names))
bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')

ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('Mean Metrics with ±1 Std Dev', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')

fig.tight_layout()
metrics_path = RESULTS_DIR / f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {metrics_path}")
plt.close()


# ============================================================================
# FINAL SUMMARY & CIRCUIT BLUEPRINT
# ============================================================================
print("\n" + "="*80)
print("PHASE 1 FINAL SUMMARY: VALIDATED CANCER DETECTION CIRCUIT")
print("="*80)

# Identify the most stable circuit (from last fold)
if fold_circuits['features']:
    print("\nMost Recent Circuit (from final CV fold):")
    print("-" * 80)
    final_features = fold_circuits['features'][-1]
    final_weights = fold_circuits['weights'][-1]
    final_bias = fold_circuits['bias'][-1]
    
    print(f"\nBiological Threshold (Bias): {final_bias:.4f}\n")
    print("--- BIOLOGICAL PARTS LIST ---")
    
    for feature, weight in zip(final_features, final_weights):
        part_type = "PROMOTER (+) - Triggers Death" if weight > 0 else "REPRESSOR (-) - Protects Cell"
        print(f"Sensor: {feature:20s} | Weight: {weight:+.4f} | {part_type}")

# Save results to CSV
print("\n" + "="*80)
print("SAVING DETAILED RESULTS")
print("="*80)

metrics_df = pd.DataFrame(fold_results)
metrics_csv = RESULTS_DIR / f"cross_validation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
metrics_df.to_csv(metrics_csv, index=False)
print(f"✓ Saved metrics: {metrics_csv}")

if stability_df is not None and len(stability_df) > 0:
    stability_csv = RESULTS_DIR / f"bootstrap_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    stability_df.to_csv(stability_csv, index=False)
    print(f"✓ Saved stability: {stability_csv}")

print("\n" + "="*80)
print(f"All deliverables saved to: {RESULTS_DIR}")
print("Phase 1 validation complete. Ready for Phase 2 (Hill ODE).")
print("="*80)
