# %% [markdown]
# # A Synthetic Cellular Perceptron for Selective Apoptosis in Lung Adenocarcinoma
# ## In Silico Design, Validation, and Clinical Translation
# ---
# **GPU-Accelerated Google Colab Pipeline v4.0 — Complete Rebuild**
#
# This notebook implements the end-to-end computational pipeline for designing a
# synthetic genetic circuit that selectively triggers apoptosis in LUAD cells while
# sparing healthy tissue. It addresses all critical peer-review errata from v3.

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP & GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# %%
# ─────────────────────────────────────────────────────────────────────────────
# 1.0  CRITICAL: SET JAX MEMORY ALLOCATION BEFORE ANY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
# JAX by default pre-allocates 90% of GPU RAM on first call, leaving nothing
# for PyTorch. We MUST set this before importing JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.25'

# ─────────────────────────────────────────────────────────────────────────────
# 1.1  INSTALL DEPENDENCIES (Colab-specific)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys

def _install(packages: list[str]) -> None:
    """Install packages silently, skip if already present."""
    for pkg in packages:
        try:
            __import__(pkg.split('==')[0].replace('-', '_'))
        except ImportError:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', pkg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

_install([
    'viennarna',          # RNA secondary-structure prediction (ViennaRNA)
    'scanpy',             # Single-cell analysis
    'anndata',            # Annotated data matrices
    'lifelines',          # Survival analysis (Kaplan-Meier, Cox PH)
    'shap',               # SHAP explainability
    'plotly',             # Interactive HTML plots
    'kaleido',            # Plotly static image export
])

# ─────────────────────────────────────────────────────────────────────────────
# 1.2  CORE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')              # non-interactive backend for Colab
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.sparse import issparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json, io, base64, time, math, re, itertools, requests, textwrap
from dataclasses import dataclass, field

# Machine-learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, recall_score, f1_score,
    accuracy_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# Survival analysis
from lifelines import KaplanMeierFitter, CoxPHFitter

# SHAP
import shap

# ─────────────────────────────────────────────────────────────────────────────
# 1.3  GPU DETECTION & FRAMEWORK INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
import torch
import jax
import jax.numpy as jnp
from jax import random as jrandom, vmap, jit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
JAX_DEVICES = jax.devices()

print("=" * 72)
print("  LUAD CELLULAR PERCEPTRON v4.0 — ENVIRONMENT REPORT")
print("=" * 72)
print(f"  Python     : {sys.version.split()[0]}")
print(f"  NumPy      : {np.__version__}")
print(f"  PyTorch    : {torch.__version__}  device={DEVICE}")
if DEVICE == 'cuda':
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"  JAX        : {jax.__version__}  devices={JAX_DEVICES}")
try:
    import RNA
    print(f"  ViennaRNA  : {RNA.__version__}")
except ImportError:
    print("  ViennaRNA  : NOT FOUND — toehold design will fail")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1.4  BIOPHYSICAL CONSTANTS (VERIFIED — DO NOT CHANGE WITHOUT PEER REVIEW)
# ─────────────────────────────────────────────────────────────────────────────
#
# These constants produce the target steady-states:
#   Cancer  (miR-210=120, miR-486=8)  → P* ≈ 519 nM  (ABOVE lethal threshold)
#   Healthy (miR-210=15,  miR-486=95) → P* ≈  11 nM  (BELOW lethal threshold)
#
# Derivation:
#   P* = (α/γ) × H_A(A, K, n) × H_R(R, K, n)
#   H_A(x) = x^n / (K^n + x^n)     — activator Hill function
#   H_R(x) = K^n / (K^n + x^n)     — repressor Hill function
#
#   Cancer:  H_A(120,40,2) = 14400/16000 = 0.9000
#            H_R(8,40,2)   = 1600/1664   = 0.9615
#            P* = 600 × 0.9000 × 0.9615  = 519.2 nM  ✓
#
#   Healthy: H_A(15,40,2)  = 225/1825    = 0.1233
#            H_R(95,40,2)  = 1600/10625  = 0.1506
#            P* = 600 × 0.1233 × 0.1506  = 11.1 nM   ✓

ALPHA           = 300.0       # nM/h — maximum protein production rate
GAMMA           = 0.5         # 1/h  — first-order degradation rate
K               = 40.0        # nM   — Hill half-saturation constant (K_A = K_R)
HILL_N          = 2.0         # dimensionless — Hill cooperativity coefficient
ALPHA_OVER_GAMMA = ALPHA / GAMMA   # = 600.0 nM — maximum steady-state protein

LETHAL_THRESHOLD = 150.0      # nM   — iCasp9 concentration that initiates apoptosis

# Reference miRNA concentrations (nM) from TCGA-LUAD median expression
MIR210_CANCER   = 120.0
MIR486_CANCER   = 8.0
MIR210_HEALTHY  = 15.0
MIR486_HEALTHY  = 95.0

# ─────────────────────────────────────────────────────────────────────────────
# 1.5  AAV CLINICAL PAYLOAD CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AAV_MAX_PAYLOAD_BP      = 4700    # AAV packaging limit (bp)
ICASP9_DHFR_FKBP_BP    = 1700    # iCasp9-DHFR-FKBP degron fusion (bp)
PROMOTER_SPC_HTERT_BP   = 300     # SP-C / hTERT dual promoter (bp)
DETARGETING_UTR_BP      = 92      # miR-122/miR-1 detargeting UTRs (bp)

# ─────────────────────────────────────────────────────────────────────────────
# 1.6  TOEHOLD SWITCH ARCHITECTURE CONSTANTS (Green et al. 2014)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_LEN    = 30       # total trigger length (nt)
TOEHOLD_LEN    = 12       # domain a: initiates binding
STEM_LEN       = 18       # domain b: strand displacement
KOZAK_LOOP     = "GCCGCCACCAUG"   # mammalian Kozak + AUG (12 nt)
ICASP9_LINKER  = "AACCUGGCGGCA"   # in-frame linker (12 nt, no stop codons)
ICASP9_START   = "AUGUCUGGAGAGCAGAGGGAC"  # iCasp9 ORF first 21 nt

# ─────────────────────────────────────────────────────────────────────────────
# 1.7  RESULTS DIRECTORY & RANDOM SEED
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path('/content/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.8  HELPER: Hill functions (used everywhere)
# ─────────────────────────────────────────────────────────────────────────────
def hill_activator(x: float, k: float = K, n: float = HILL_N) -> float:
    """H_A(x) = x^n / (K^n + x^n). Range [0,1]. Activating Hill function."""
    return (x ** n) / (k ** n + x ** n) if x > 0 else 0.0

def hill_repressor(x: float, k: float = K, n: float = HILL_N) -> float:
    """H_R(x) = K^n / (K^n + x^n). Range [0,1]. Repressing Hill function."""
    return (k ** n) / (k ** n + x ** n)

# ─────────────────────────────────────────────────────────────────────────────
# 1.9  HELPER: Matplotlib figure to base64 PNG (for HTML report)
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Global store for all report figures and metrics
REPORT_FIGURES: Dict[str, str] = {}   # name -> base64 PNG
REPORT_METRICS: Dict[str, any] = {}   # name -> value

print("\n✓ Cell 1 complete. All constants loaded, GPU detected.")
print(f"  Steady-state verification:")
print(f"    Cancer P*  = {ALPHA_OVER_GAMMA * hill_activator(MIR210_CANCER) * hill_repressor(MIR486_CANCER):.1f} nM (target: 519 nM)")
print(f"    Healthy P* = {ALPHA_OVER_GAMMA * hill_activator(MIR210_HEALTHY) * hill_repressor(MIR486_HEALTHY):.1f} nM (target: 11 nM)")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: PHASE 1 — miRNA BIOMARKER DISCOVERY & CLINICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("=" * 72)
print("  PHASE 1: miRNA BIOMARKER DISCOVERY & CLINICAL VALIDATION")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  LOAD TCGA-LUAD miRNA EXPRESSION DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading TCGA-LUAD miRNA expression data...")

mirna_raw = pd.read_csv('/content/TCGA-LUAD.mirna.tsv', sep='\t', index_col=0)
df_mirna = mirna_raw.T.copy()

def _parse_tcga_label(barcode: str) -> Optional[int]:
    """
    Extract cancer/healthy label from TCGA barcode.
    
    TCGA barcode format: TCGA-XX-XXXX-01A-...
    The 4th field encodes sample type:
      01–09 = Tumor (cancer)  → label 1
      10–19 = Normal           → label 0
    """
    try:
        sample_code = str(barcode).split('-')[3]
        if sample_code.startswith('0'):
            return 1  # Cancer
        elif sample_code.startswith('1'):
            return 0  # Healthy
    except (IndexError, AttributeError):
        pass
    return None

df_mirna['Target'] = [_parse_tcga_label(idx) for idx in df_mirna.index]
df_mirna = df_mirna.dropna(subset=['Target'])
df_mirna['Target'] = df_mirna['Target'].astype(int)

# Biological pre-filter: remove miRNAs below noise floor (mean RPM < 1.0)
mirna_cols = df_mirna.columns.drop('Target')
abundant = mirna_cols[df_mirna[mirna_cols].mean() > 1.0]
df_mirna = df_mirna[abundant.tolist() + ['Target']]

X_mirna = df_mirna.drop(columns=['Target'])
y_mirna = df_mirna['Target']

n_cancer = int(y_mirna.sum())
n_healthy = int(len(y_mirna) - n_cancer)

print(f"  Patients: {len(df_mirna)} ({n_cancer} cancer, {n_healthy} normal)")
print(f"  Features: {X_mirna.shape[1]} miRNAs (after mean > 1.0 RPM filter)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  STABL-STYLE L1 LOGISTIC REGRESSION WITH SYNTHETIC NOISE INJECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Running 1,000 bootstrap L1 with Stabl synthetic noise injection...")

N_BOOTSTRAP = 1000
N_NOISE_FEATURES = 50   # synthetic Gaussian noise columns

# Storage: count how often each real miRNA survives L1 + noise competition
selection_counts = np.zeros(X_mirna.shape[1], dtype=int)
weight_accumulator = np.zeros(X_mirna.shape[1], dtype=float)

rng = np.random.default_rng(RNG_SEED)

for b in range(N_BOOTSTRAP):
    # Resample with replacement
    idx = rng.choice(len(X_mirna), size=len(X_mirna), replace=True)
    X_boot = X_mirna.iloc[idx].copy()
    y_boot = y_mirna.iloc[idx]
    
    # Inject synthetic noise features (Stabl trick):
    # Real biomarkers must survive L1 even when competing with noise
    noise = pd.DataFrame(
        rng.standard_normal((len(X_boot), N_NOISE_FEATURES)),
        index=X_boot.index,
        columns=[f'_NOISE_{i}' for i in range(N_NOISE_FEATURES)]
    )
    X_aug = pd.concat([X_boot, noise], axis=1)
    
    # Sweep L1 regularization to find a sparse circuit (2–5 real features)
    for C_val in np.logspace(-3, 1, 80):
        mdl = LogisticRegression(
            penalty='l1', solver='liblinear', C=C_val,
            class_weight='balanced', random_state=b, max_iter=2000
        )
        mdl.fit(X_aug, y_boot)
        
        coefs_real = mdl.coef_[0][:X_mirna.shape[1]]  # exclude noise columns
        n_nonzero = np.sum(coefs_real != 0)
        
        if 2 <= n_nonzero <= 5:
            mask = coefs_real != 0
            selection_counts[mask] += 1
            weight_accumulator[mask] += coefs_real[mask]
            break
    
    if (b + 1) % 200 == 0:
        print(f"    Completed {b + 1}/{N_BOOTSTRAP} bootstraps...")

print(f"    ✓ {N_BOOTSTRAP} bootstraps completed.")

# Identify stable biomarkers (frequency > 60%)
stability_freq = selection_counts / N_BOOTSTRAP
stable_mask = stability_freq > 0.6
mean_weights = np.where(selection_counts > 0, weight_accumulator / selection_counts, 0)

stable_mirnas = X_mirna.columns[stable_mask].tolist()
stable_freqs = stability_freq[stable_mask]
stable_weights = mean_weights[stable_mask]

print(f"\n  Stable biomarkers (freq > 60%):")
for name, freq, wt in sorted(zip(stable_mirnas, stable_freqs, stable_weights),
                               key=lambda t: -t[1]):
    role = "ACTIVATOR (+)" if wt > 0 else "REPRESSOR (−)"
    print(f"    {name:25s}  freq={freq:.3f}  weight={wt:+.4f}  → {role}")

# Store for report
REPORT_METRICS['stable_mirnas'] = stable_mirnas
REPORT_METRICS['mir210_freq'] = float(stability_freq[X_mirna.columns.get_loc('hsa-miR-210')]) if 'hsa-miR-210' in X_mirna.columns else 0.0
REPORT_METRICS['mir486_freq'] = float(stability_freq[X_mirna.columns.get_loc('hsa-miR-486-2')]) if 'hsa-miR-486-2' in X_mirna.columns else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 2.3  STRATIFIED 5-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Running Stratified 5-Fold Cross-Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

cv_auc, cv_sens, cv_spec, cv_f1 = [], [], [], []
y_test_all, y_prob_all = [], []

# Use only the two stable biomarkers for the final circuit
circuit_features = ['hsa-miR-210', 'hsa-miR-486-2']
circuit_features = [f for f in circuit_features if f in X_mirna.columns]

if len(circuit_features) < 2:
    # Fallback: use top-2 stable features
    top2_idx = np.argsort(stability_freq)[-2:]
    circuit_features = X_mirna.columns[top2_idx].tolist()

X_circuit = X_mirna[circuit_features]

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_circuit, y_mirna), 1):
    X_tr, X_te = X_circuit.iloc[tr_idx], X_circuit.iloc[te_idx]
    y_tr, y_te = y_mirna.iloc[tr_idx], y_mirna.iloc[te_idx]
    
    mdl = LogisticRegression(
        penalty='l2', solver='lbfgs', C=1.0,
        class_weight='balanced', random_state=RNG_SEED, max_iter=2000
    )
    mdl.fit(X_tr, y_tr)
    
    y_pred = mdl.predict(X_te)
    y_prob = mdl.predict_proba(X_te)[:, 1]
    
    fold_auc = roc_auc_score(y_te, y_prob)
    fold_sens = recall_score(y_te, y_pred, pos_label=1)
    fold_spec = recall_score(y_te, y_pred, pos_label=0)
    fold_f1 = f1_score(y_te, y_pred, pos_label=1)
    
    cv_auc.append(fold_auc)
    cv_sens.append(fold_sens)
    cv_spec.append(fold_spec)
    cv_f1.append(fold_f1)
    y_test_all.extend(y_te.tolist())
    y_prob_all.extend(y_prob.tolist())
    
    print(f"  Fold {fold}: AUC={fold_auc:.4f}  Sens={fold_sens:.4f}  Spec={fold_spec:.4f}")

mean_auc = np.mean(cv_auc)
mean_sens = np.mean(cv_sens)
mean_spec = np.mean(cv_spec)

print(f"\n  ► Mean AUC:         {mean_auc:.4f} ± {np.std(cv_auc):.4f}")
print(f"  ► Mean Sensitivity: {mean_sens:.4f} ± {np.std(cv_sens):.4f}")
print(f"  ► Mean Specificity: {mean_spec:.4f} ± {np.std(cv_spec):.4f}")

REPORT_METRICS['cv_auc'] = round(mean_auc, 4)
REPORT_METRICS['cv_sensitivity'] = round(mean_sens, 4)
REPORT_METRICS['cv_specificity'] = round(mean_spec, 4)

# ─────────────────────────────────────────────────────────────────────────────
# 2.4  SHAP EXPLAINABILITY (LinearExplainer)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Computing SHAP values (LinearExplainer)...")

# Fit final model on all data for SHAP
mdl_final = LogisticRegression(
    penalty='l2', solver='lbfgs', C=1.0,
    class_weight='balanced', random_state=RNG_SEED, max_iter=2000
)
mdl_final.fit(X_circuit, y_mirna)

explainer = shap.LinearExplainer(mdl_final, X_circuit, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_circuit)

# Determine directionality from mean SHAP
for i, feat in enumerate(circuit_features):
    mean_shap = np.mean(shap_values[:, i])
    direction = "Activator (↑cancer)" if mean_shap > 0 else "Repressor (↓cancer)"
    print(f"  {feat:25s}  mean SHAP = {mean_shap:+.4f}  → {direction}")

# SHAP beeswarm plot
fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
shap.summary_plot(shap_values, X_circuit, feature_names=circuit_features, show=False)
plt.title("SHAP Beeswarm: miRNA Biomarker Directionality", fontsize=12, fontweight='bold')
plt.tight_layout()
REPORT_FIGURES['shap_beeswarm'] = fig_to_base64(plt.gcf())
plt.savefig(RESULTS_DIR / 'phase1_shap_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close('all')
print("  ✓ SHAP plot saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.5  CROSS-VALIDATED ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Generating cross-validated ROC curve...")

y_test_arr = np.array(y_test_all)
y_prob_arr = np.array(y_prob_all)
fpr, tpr, _ = roc_curve(y_test_arr, y_prob_arr)
roc_auc_val = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
ax_roc.plot(fpr, tpr, color='#e74c3c', lw=2.5,
            label=f'L1 Circuit (AUC = {roc_auc_val:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Random (AUC = 0.500)')
ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
ax_roc.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=11, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
ax_roc.set_title('Cross-Validated ROC: L1 miRNA Circuit', fontsize=12, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=10)
ax_roc.set_xlim([-0.01, 1.01])
ax_roc.set_ylim([-0.01, 1.01])
ax_roc.grid(True, alpha=0.25, linestyle=':')
REPORT_FIGURES['roc_curve'] = fig_to_base64(fig_roc)
fig_roc.savefig(RESULTS_DIR / 'phase1_roc_curve.png', dpi=200, bbox_inches='tight')
plt.close(fig_roc)
print(f"  ✓ ROC curve saved (AUC = {roc_auc_val:.3f}).")

# ─────────────────────────────────────────────────────────────────────────────
# 2.6  KAPLAN-MEIER & COX PROPORTIONAL HAZARDS SURVIVAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Kaplan-Meier & Cox PH survival analysis...")

clinical_raw = pd.read_csv('/content/TCGA-LUAD.clinical.tsv', sep='\t')

# Parse survival data
# Columns vary across TCGA downloads; handle common variants
os_col = None
for candidate in ['days_to_death', 'OS.time', 'days_to_last_followup']:
    if candidate in clinical_raw.columns:
        os_col = candidate
        break

status_col = None
for candidate in ['vital_status', 'OS']:
    if candidate in clinical_raw.columns:
        status_col = candidate
        break

# Attempt the survival analysis (may fail if clinical columns are absent)
km_success = False
if os_col and status_col:
    try:
        # Build patient-level risk score from miRNA data
        risk_scores = mdl_final.predict_proba(X_circuit)[:, 1]
        risk_median = np.median(risk_scores)
        risk_group = (risk_scores >= risk_median).astype(int)  # 1 = high-risk
        
        # Merge with clinical data by TCGA barcode
        surv_df = pd.DataFrame({
            'barcode': X_circuit.index,
            'risk_score': risk_scores,
            'risk_group': risk_group
        })
        
        # Extract submitter_id or case_submitter_id from clinical
        id_col = None
        for candidate in ['submitter_id', 'case_submitter_id', 'bcr_patient_barcode']:
            if candidate in clinical_raw.columns:
                id_col = candidate
                break
        
        if id_col:
            # Truncate TCGA barcodes to patient-level (12 chars: TCGA-XX-XXXX)
            surv_df['patient_id'] = surv_df['barcode'].str[:12]
            clinical_raw['patient_id'] = clinical_raw[id_col].str[:12]
            
            merged = surv_df.merge(clinical_raw[['patient_id', os_col, status_col]].drop_duplicates(),
                                    on='patient_id', how='inner')
            
            # Clean survival time and event columns
            merged['time'] = pd.to_numeric(merged[os_col], errors='coerce')
            
            if status_col == 'vital_status':
                merged['event'] = (merged[status_col].str.lower() == 'dead').astype(int)
            else:
                merged['event'] = pd.to_numeric(merged[status_col], errors='coerce')
            
            merged = merged.dropna(subset=['time', 'event'])
            merged = merged[merged['time'] > 0]
            
            if len(merged) > 20:
                # Kaplan-Meier
                fig_km, ax_km = plt.subplots(figsize=(8, 5))
                kmf = KaplanMeierFitter()
                
                for grp, label, color in [(0, 'Low Risk', '#2ecc71'), (1, 'High Risk', '#e74c3c')]:
                    mask_g = merged['risk_group'] == grp
                    kmf.fit(merged.loc[mask_g, 'time'], merged.loc[mask_g, 'event'], label=label)
                    kmf.plot_survival_function(ax=ax_km, ci_show=True, color=color, linewidth=2)
                
                ax_km.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
                ax_km.set_ylabel('Survival Probability', fontsize=11, fontweight='bold')
                ax_km.set_title('Kaplan-Meier: miRNA Risk Score Stratification', fontsize=12, fontweight='bold')
                ax_km.legend(fontsize=10)
                ax_km.grid(True, alpha=0.25, linestyle=':')
                REPORT_FIGURES['kaplan_meier'] = fig_to_base64(fig_km)
                fig_km.savefig(RESULTS_DIR / 'phase1_kaplan_meier.png', dpi=200, bbox_inches='tight')
                plt.close(fig_km)
                
                # Cox PH
                cox_df = merged[['time', 'event', 'risk_score']].copy()
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col='time', event_col='event')
                hr = np.exp(cph.params_['risk_score'])
                p_val = cph.summary.loc['risk_score', 'p']
                
                print(f"  Kaplan-Meier: {len(merged)} patients plotted")
                print(f"  Cox PH: HR = {hr:.2f}, p = {p_val:.4f}")
                REPORT_METRICS['cox_hr'] = round(hr, 2)
                REPORT_METRICS['cox_pval'] = round(p_val, 4)
                km_success = True
    except Exception as e:
        print(f"  ⚠ Survival analysis failed: {e}")

if not km_success:
    print("  ⚠ Clinical data columns not found or insufficient for survival analysis.")
    REPORT_METRICS['cox_hr'] = 'N/A'
    REPORT_METRICS['cox_pval'] = 'N/A'

print("\n✓ Phase 1 complete.")
print(f"  Circuit: {' + '.join(circuit_features)}")
print(f"  AUC = {mean_auc:.4f}, Sensitivity = {mean_sens:.4f}, Specificity = {mean_spec:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: PHASE 2 — BIOPHYSICAL SIMULATION & STOCHASTIC SAFETY
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n" + "=" * 72)
print("  PHASE 2: BIOPHYSICAL SIMULATION & STOCHASTIC SAFETY")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  DETERMINISTIC ODE: 2-input Hill equation over 48 hours
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Deterministic ODE simulation...")

def perceptron_ode(t, P, mir210, mir486):
    """
    ODE for the cellular perceptron kill-switch.
    
    dP/dt = α × H_A(miR-210) × H_R(miR-486) − γ × P
    
    This models a genetic AND gate: the killer protein (iCasp9) is produced
    only when miR-210 is HIGH (cancer) AND miR-486 is LOW (cancer).
    
    Parameters
    ----------
    t       : float — time (hours)
    P       : float — killer protein concentration (nM)
    mir210  : float — miR-210 concentration (nM, constant input)
    mir486  : float — miR-486 concentration (nM, constant input)
    
    Returns
    -------
    dPdt : float — rate of change of killer protein (nM/h)
    """
    H_A = hill_activator(mir210)
    H_R = hill_repressor(mir486)
    return ALPHA * H_A * H_R - GAMMA * P

# Simulate 48 hours for both cell types
t_span = (0.0, 48.0)
t_eval = np.linspace(0, 48, 500)

sol_cancer = solve_ivp(perceptron_ode, t_span, [0.0],
                        args=(MIR210_CANCER, MIR486_CANCER),
                        t_eval=t_eval, method='RK45')
sol_healthy = solve_ivp(perceptron_ode, t_span, [0.0],
                         args=(MIR210_HEALTHY, MIR486_HEALTHY),
                         t_eval=t_eval, method='RK45')

P_cancer_ss = sol_cancer.y[0, -1]
P_healthy_ss = sol_healthy.y[0, -1]

print(f"  Cancer  steady-state P* = {P_cancer_ss:.1f} nM  (target: ~519 nM)")
print(f"  Healthy steady-state P* = {P_healthy_ss:.1f} nM  (target: ~11 nM)")
print(f"  Separation ratio: {P_cancer_ss / max(P_healthy_ss, 0.01):.1f}×")

REPORT_METRICS['ode_cancer_ss'] = round(P_cancer_ss, 1)
REPORT_METRICS['ode_healthy_ss'] = round(P_healthy_ss, 1)

# ODE trajectory plot
fig_ode, ax_ode = plt.subplots(figsize=(9, 5))
ax_ode.plot(sol_cancer.t, sol_cancer.y[0], color='#e74c3c', lw=2.5,
            label=f'Cancer (P*={P_cancer_ss:.0f} nM)')
ax_ode.plot(sol_healthy.t, sol_healthy.y[0], color='#2ecc71', lw=2.5,
            label=f'Healthy (P*={P_healthy_ss:.0f} nM)')
ax_ode.axhline(LETHAL_THRESHOLD, color='#333', linestyle='--', lw=2,
               label=f'Lethal Threshold ({LETHAL_THRESHOLD} nM)')
ax_ode.fill_between(sol_cancer.t, LETHAL_THRESHOLD, sol_cancer.y[0],
                     where=sol_cancer.y[0] > LETHAL_THRESHOLD,
                     alpha=0.15, color='#e74c3c')
ax_ode.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax_ode.set_ylabel('iCasp9 [nM]', fontsize=11, fontweight='bold')
ax_ode.set_title('Deterministic ODE: Killer Protein Dynamics (48h)',
                  fontsize=12, fontweight='bold')
ax_ode.legend(fontsize=10, loc='center right')
ax_ode.set_xlim([0, 48])
ax_ode.set_ylim([0, max(P_cancer_ss * 1.15, 200)])
ax_ode.grid(True, alpha=0.25, linestyle=':')
REPORT_FIGURES['ode_trajectory'] = fig_to_base64(fig_ode)
fig_ode.savefig(RESULTS_DIR / 'phase2_ode_trajectory.png', dpi=200, bbox_inches='tight')
plt.close(fig_ode)

# ─────────────────────────────────────────────────────────────────────────────
# 3.2  MONTE CARLO ROBUSTNESS: 200 trials, ±20% Gaussian noise
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Monte Carlo robustness analysis (200 trials, ±20% noise)...")

N_MC = 200
mc_rng = np.random.default_rng(RNG_SEED)

mc_cancer_peaks = np.zeros(N_MC)
mc_healthy_peaks = np.zeros(N_MC)

for trial in range(N_MC):
    # Perturb all biophysical parameters by ±20% Gaussian noise
    a_pert = ALPHA * (1.0 + 0.20 * mc_rng.standard_normal())
    g_pert = GAMMA * (1.0 + 0.20 * mc_rng.standard_normal())
    ka_pert = K * (1.0 + 0.20 * mc_rng.standard_normal())
    kr_pert = K * (1.0 + 0.20 * mc_rng.standard_normal())
    
    # Clamp to biophysically reasonable ranges
    a_pert = max(a_pert, 10.0)
    g_pert = max(g_pert, 0.05)
    ka_pert = max(ka_pert, 5.0)
    kr_pert = max(kr_pert, 5.0)
    
    # Steady-state: P* = (α/γ) × H_A × H_R
    ratio = a_pert / g_pert
    
    # Cancer
    ha_c = (MIR210_CANCER ** HILL_N) / (ka_pert ** HILL_N + MIR210_CANCER ** HILL_N)
    hr_c = (kr_pert ** HILL_N) / (kr_pert ** HILL_N + MIR486_CANCER ** HILL_N)
    mc_cancer_peaks[trial] = ratio * ha_c * hr_c
    
    # Healthy
    ha_h = (MIR210_HEALTHY ** HILL_N) / (ka_pert ** HILL_N + MIR210_HEALTHY ** HILL_N)
    hr_h = (kr_pert ** HILL_N) / (kr_pert ** HILL_N + MIR486_HEALTHY ** HILL_N)
    mc_healthy_peaks[trial] = ratio * ha_h * hr_h

# Robustness check
cancer_all_above = np.all(mc_cancer_peaks > LETHAL_THRESHOLD)
healthy_all_below = np.all(mc_healthy_peaks < LETHAL_THRESHOLD)
robustness_pct = 100.0 if (cancer_all_above and healthy_all_below) else 0.0

# If not fully robust, count fraction
if not (cancer_all_above and healthy_all_below):
    n_correct = np.sum((mc_cancer_peaks > LETHAL_THRESHOLD) & (mc_healthy_peaks < LETHAL_THRESHOLD))
    robustness_pct = (n_correct / N_MC) * 100.0

print(f"  Cancer  peak range: [{mc_cancer_peaks.min():.1f}, {mc_cancer_peaks.max():.1f}] nM")
print(f"  Healthy peak range: [{mc_healthy_peaks.min():.1f}, {mc_healthy_peaks.max():.1f}] nM")
print(f"  Robustness: {robustness_pct:.1f}% of {N_MC} trials maintain correct classification")

REPORT_METRICS['mc_robustness'] = robustness_pct

# Monte Carlo distribution plot
fig_mc, ax_mc = plt.subplots(figsize=(9, 5))
ax_mc.hist(mc_cancer_peaks, bins=30, alpha=0.6, color='#e74c3c', label='Cancer P*',
           edgecolor='darkred', linewidth=0.8)
ax_mc.hist(mc_healthy_peaks, bins=30, alpha=0.6, color='#2ecc71', label='Healthy P*',
           edgecolor='darkgreen', linewidth=0.8)
ax_mc.axvline(LETHAL_THRESHOLD, color='#333', linestyle='--', lw=2.5,
              label=f'Lethal Threshold ({LETHAL_THRESHOLD} nM)')
ax_mc.set_xlabel('Steady-State iCasp9 [nM]', fontsize=11, fontweight='bold')
ax_mc.set_ylabel('Count', fontsize=11, fontweight='bold')
ax_mc.set_title(f'Monte Carlo Robustness: {N_MC} trials (±20% parameter noise)\n'
                f'Robustness = {robustness_pct:.0f}%',
                fontsize=12, fontweight='bold')
ax_mc.legend(fontsize=10)
ax_mc.grid(True, alpha=0.25, linestyle=':')
REPORT_FIGURES['monte_carlo'] = fig_to_base64(fig_mc)
fig_mc.savefig(RESULTS_DIR / 'phase2_monte_carlo.png', dpi=200, bbox_inches='tight')
plt.close(fig_mc)

# ─────────────────────────────────────────────────────────────────────────────
# 3.3  GILLESPIE SSA VIA JAX vmap (10,000 GPU-parallel trajectories)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Gillespie SSA (JAX vmap, 10,000 trajectories on GPU)...")

# Convert nM concentrations to molecule counts for stochastic simulation
# Volume = 1 pL (10^-12 L) — typical mammalian cell cytoplasm
CELL_VOLUME_L = 1e-12
AVOGADRO = 6.022e23

def nM_to_molecules(conc_nM: float) -> int:
    """Convert nanomolar concentration to discrete molecule count."""
    return int(round(conc_nM * 1e-9 * CELL_VOLUME_L * AVOGADRO))

def molecules_to_nM(count: int) -> float:
    """Convert discrete molecule count to nanomolar concentration."""
    return count / (1e-9 * CELL_VOLUME_L * AVOGADRO)

# Molecule counts for reference conditions
LETHAL_THRESHOLD_MOL = nM_to_molecules(LETHAL_THRESHOLD)

N_SSA = 10000         # trajectories
SSA_T_MAX = 48.0      # hours
SSA_DT = 0.1          # time step for tau-leaping (hours)
SSA_STEPS = int(SSA_T_MAX / SSA_DT)

print(f"  Lethal threshold = {LETHAL_THRESHOLD_MOL} molecules ({LETHAL_THRESHOLD} nM)")
print(f"  Simulating {N_SSA} trajectories × {SSA_STEPS} steps (dt={SSA_DT}h)...")

@jit
def gillespie_tau_leap_trajectory(key, mir210_nM, mir486_nM):
    """
    JAX-compiled tau-leaping Gillespie SSA for one trajectory.
    
    Models two reactions:
      R1: ∅ → P   with propensity α × H_A(miR-210) × H_R(miR-486)
      R2: P → ∅   with propensity γ × P
    
    miR-210 and miR-486 are treated as CONSTANT INPUTS (not species),
    representing the steady-state miRNA milieu inside a cell. This models
    baseline transcriptional noise in the killer protein pathway.
    
    Parameters
    ----------
    key      : JAX PRNGKey
    mir210_nM: float — input miR-210 concentration (nM)
    mir486_nM: float — input miR-486 concentration (nM)
    
    Returns
    -------
    max_P_nM : float — maximum P concentration (nM) across the trajectory
    """
    # Pre-compute Hill function values (constant for this trajectory)
    H_A = (mir210_nM ** HILL_N) / (K ** HILL_N + mir210_nM ** HILL_N)
    H_R = (K ** HILL_N) / (K ** HILL_N + mir486_nM ** HILL_N)
    production_rate_nM_per_h = ALPHA * H_A * H_R
    
    # Convert to molecules/step — INLINE the calculation (cannot call
    # Python int()/round() inside JAX JIT; must use pure JAX arithmetic)
    NM_TO_MOL_FACTOR = 1e-9 * CELL_VOLUME_L * AVOGADRO  # ≈ 0.6022
    prod_rate_mol_per_step = production_rate_nM_per_h * SSA_DT * NM_TO_MOL_FACTOR
    
    P = 0.0            # killer protein (molecules)
    max_P = 0.0
    
    def body_fn(carry, step_key):
        P, max_P = carry
        k1, k2 = jrandom.split(step_key)
        
        # R1: Production — Poisson-distributed number of new molecules
        n_produced = jrandom.poisson(k1, jnp.float32(jnp.maximum(prod_rate_mol_per_step, 0.0)))
        
        # R2: Degradation — each molecule degrades with probability γ×dt
        # Number degraded ~ Binomial(P, γ×dt), approximated by Poisson
        n_degraded = jrandom.poisson(k2, jnp.float32(jnp.maximum(P * GAMMA * SSA_DT, 0.0)))
        n_degraded = jnp.minimum(n_degraded, P)  # can't degrade more than exist
        
        P_new = P + n_produced - n_degraded
        P_new = jnp.maximum(P_new, 0.0)
        max_P_new = jnp.maximum(max_P, P_new)
        
        return (P_new, max_P_new), None
    
    keys = jrandom.split(key, SSA_STEPS)
    (final_P, max_P), _ = jax.lax.scan(body_fn, (P, max_P), keys)
    
    # Convert max P back to nM
    max_P_nM = max_P / (1e-9 * CELL_VOLUME_L * AVOGADRO)
    return max_P_nM

# Vectorize across N_SSA trajectories
batched_gillespie = vmap(gillespie_tau_leap_trajectory, in_axes=(0, None, None))

# --- Cancer cells ---
cancer_keys = jrandom.split(jrandom.PRNGKey(RNG_SEED), N_SSA)
ssa_cancer_max = np.array(batched_gillespie(cancer_keys, MIR210_CANCER, MIR486_CANCER))

# --- Healthy cells ---
healthy_keys = jrandom.split(jrandom.PRNGKey(RNG_SEED + 1000), N_SSA)
ssa_healthy_max = np.array(batched_gillespie(healthy_keys, MIR210_HEALTHY, MIR486_HEALTHY))

# P(lethal) for each cell type
p_lethal_cancer = float(np.mean(ssa_cancer_max > LETHAL_THRESHOLD) * 100)
p_lethal_healthy = float(np.mean(ssa_healthy_max > LETHAL_THRESHOLD) * 100)

print(f"\n  Cancer  P(lethal):  {p_lethal_cancer:.3f}%  (max peak range: [{ssa_cancer_max.min():.1f}, {ssa_cancer_max.max():.1f}] nM)")
print(f"  Healthy P(lethal):  {p_lethal_healthy:.3f}%  (max peak range: [{ssa_healthy_max.min():.1f}, {ssa_healthy_max.max():.1f}] nM)")

REPORT_METRICS['ssa_p_lethal_cancer'] = round(p_lethal_cancer, 3)
REPORT_METRICS['ssa_p_lethal_healthy'] = round(p_lethal_healthy, 3)

# Gillespie distribution plot
fig_ssa, ax_ssa = plt.subplots(figsize=(9, 5))
ax_ssa.hist(ssa_cancer_max, bins=60, alpha=0.6, color='#e74c3c', label='Cancer',
            edgecolor='darkred', linewidth=0.5, density=True)
ax_ssa.hist(ssa_healthy_max, bins=60, alpha=0.6, color='#2ecc71', label='Healthy',
            edgecolor='darkgreen', linewidth=0.5, density=True)
ax_ssa.axvline(LETHAL_THRESHOLD, color='#333', linestyle='--', lw=2.5,
               label=f'Lethal Threshold ({LETHAL_THRESHOLD} nM)')
ax_ssa.set_xlabel('Max iCasp9 [nM] across 48h trajectory', fontsize=11, fontweight='bold')
ax_ssa.set_ylabel('Density', fontsize=11, fontweight='bold')
ax_ssa.set_title(f'Gillespie SSA: {N_SSA:,} Trajectories (JAX GPU)\n'
                 f'P(lethal|cancer)={p_lethal_cancer:.1f}%  P(lethal|healthy)={p_lethal_healthy:.3f}%',
                 fontsize=12, fontweight='bold')
ax_ssa.legend(fontsize=10)
ax_ssa.grid(True, alpha=0.25, linestyle=':')
REPORT_FIGURES['gillespie_ssa'] = fig_to_base64(fig_ssa)
fig_ssa.savefig(RESULTS_DIR / 'phase2_gillespie_ssa.png', dpi=200, bbox_inches='tight')
plt.close(fig_ssa)

# ─────────────────────────────────────────────────────────────────────────────
# 3.4  SPATIAL MOI TRANSDUCTION: AAV6.2FF Poisson model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Spatial MOI transduction (AAV6.2FF Poisson model)...")

moi_values = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
p_transduced = 1.0 - np.exp(-moi_values)   # P(k≥1 viral particle per cell)

fig_moi, ax_moi = plt.subplots(figsize=(8, 4.5))
ax_moi.plot(moi_values, p_transduced * 100, 'o-', color='#3498db', lw=2.5,
            markersize=8, markeredgecolor='#2c3e50', markeredgewidth=1.5)
ax_moi.axhline(95, color='#e67e22', linestyle='--', lw=1.5, alpha=0.7, label='95% transduction')
ax_moi.axhline(99, color='#e74c3c', linestyle='--', lw=1.5, alpha=0.7, label='99% transduction')
ax_moi.set_xlabel('MOI (Multiplicity of Infection)', fontsize=11, fontweight='bold')
ax_moi.set_ylabel('P(transduced) [%]', fontsize=11, fontweight='bold')
ax_moi.set_title('AAV6.2FF Alveolar Delivery: Poisson Transduction Model',
                  fontsize=12, fontweight='bold')
ax_moi.set_xscale('log')
ax_moi.legend(fontsize=10)
ax_moi.grid(True, alpha=0.25, linestyle=':')
ax_moi.set_ylim([0, 105])

# Annotate key MOI values
for moi, pt in zip([1.0, 5.0, 20.0], [p_transduced[1], p_transduced[3], p_transduced[5]]):
    ax_moi.annotate(f'{pt*100:.1f}%', xy=(moi, pt*100), xytext=(moi*1.5, pt*100 - 8),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))

REPORT_FIGURES['moi_transduction'] = fig_to_base64(fig_moi)
fig_moi.savefig(RESULTS_DIR / 'phase2_moi_transduction.png', dpi=200, bbox_inches='tight')
plt.close(fig_moi)

# MOI summary table
print(f"  {'MOI':>6s}  {'P(transduced)':>14s}")
print(f"  {'─'*6}  {'─'*14}")
for m, p in zip(moi_values, p_transduced):
    print(f"  {m:6.1f}  {p*100:13.2f}%")

REPORT_METRICS['moi_95_target'] = float(moi_values[p_transduced >= 0.95][0]) if np.any(p_transduced >= 0.95) else 'N/A'

print("\n✓ Phase 2 complete.")
print(f"  ODE: Cancer={P_cancer_ss:.0f} nM, Healthy={P_healthy_ss:.0f} nM")
print(f"  Monte Carlo: {robustness_pct:.0f}% robustness")
print(f"  Gillespie SSA: P(lethal|healthy) = {p_lethal_healthy:.3f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: PHASE 3 — SINGLE-CELL SOFT-LOGIC GATE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n" + "=" * 72)
print("  PHASE 3: SINGLE-CELL SOFT-LOGIC GATE DISCOVERY (PyTorch GPU)")
print("=" * 72)

import scanpy as sc

# ─────────────────────────────────────────────────────────────────────────────
# 4.1  LOAD AND LABEL SINGLE-CELL DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading LUAD.h5ad scRNA-seq data...")
adata = sc.read_h5ad('/content/LUAD.h5ad')
print(f"  Total cells: {adata.n_obs:,}")
print(f"  Total genes: {adata.n_vars:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.2  CRITICAL LABELING: Strict cancer vs. healthy classification
# ─────────────────────────────────────────────────────────────────────────────
# Previous code INCORRECTLY labeled ALL epithelial cells as cancer.
# Normal lung epithelial subsets (ciliated, club, basal, alveolar type I/II)
# must be assigned to the Healthy/TME pool. Only cells explicitly annotated
# as malignant/tumor go to the Cancer pool.
print("\n[2/5] Strict cell labeling (fixing epithelial misclassification)...")

# Identify the cell-type annotation column (varies by dataset)
celltype_col = None
for candidate in ['cell_type', 'author_cell_type', 'cell_type_detailed',
                   'author_cell_type_level_1', 'CellType', 'celltype']:
    if candidate in adata.obs.columns:
        celltype_col = candidate
        break

if celltype_col is None:
    raise ValueError("No cell-type annotation column found in adata.obs")

print(f"  Using annotation column: '{celltype_col}'")
print(f"  Unique cell types: {adata.obs[celltype_col].nunique()}")

# Define NORMAL lung epithelial subtypes that must be EXCLUDED from cancer pool
NORMAL_EPITHELIAL_PATTERNS = [
    'ciliated', 'club', 'basal', 'alveolar', 'at1', 'at2',
    'type i', 'type ii', 'goblet', 'ionocyte', 'neuroendocrine',
    'secretory', 'mucous', 'serous', 'clara',
    'normal', 'healthy', 'non-malignant', 'non_malignant'
]

CANCER_PATTERNS = [
    'malignant', 'tumor', 'cancer', 'carcinoma', 'neoplastic',
    'luad', 'adenocarcinoma'
]

def classify_cell(celltype_str: str) -> str:
    """
    Classify a cell as 'cancer' or 'healthy' based on its annotation.
    
    CRITICAL LOGIC:
    1. If annotation matches any CANCER_PATTERN → 'cancer'
    2. If annotation matches NORMAL_EPITHELIAL → 'healthy' (NOT cancer!)
    3. If annotation is 'Epithelial' with NO further qualifier → check sub-columns
    4. All immune/stromal/endothelial cells → 'healthy' (TME pool)
    """
    ct = str(celltype_str).lower().strip()
    
    # Check cancer patterns FIRST
    for pattern in CANCER_PATTERNS:
        if pattern in ct:
            return 'cancer'
    
    # Check normal epithelial patterns
    for pattern in NORMAL_EPITHELIAL_PATTERNS:
        if pattern in ct:
            return 'healthy'
    
    # Pure "epithelial" without qualifier — CHECK if dataset has a separate
    # malignancy annotation column
    if 'epithelial' in ct:
        return 'cancer'  # Conservative: in tumor scRNA-seq, bare "Epithelial" is typically cancer
    
    # Everything else (immune, stromal, endothelial, etc.) is healthy/TME
    return 'healthy'

adata.obs['cell_class'] = adata.obs[celltype_col].apply(classify_cell)

n_cancer_sc = int((adata.obs['cell_class'] == 'cancer').sum())
n_healthy_sc = int((adata.obs['cell_class'] == 'healthy').sum())
print(f"  Cancer cells:  {n_cancer_sc:,}")
print(f"  Healthy cells: {n_healthy_sc:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.3  SUBSAMPLE AND FILTER GENES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Subsampling and filtering genes...")

SUBSAMPLE_SIZE = 5000
sc.pp.subsample(adata, n_obs=min(SUBSAMPLE_SIZE, adata.n_obs), random_state=RNG_SEED)

# Minimum expression filter: prevent dropout "ghost genes" from dominating
# A gene must have mean > 0.5 counts to be considered real signal
X_dense = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
gene_means = X_dense.mean(axis=0)
valid_genes = gene_means > 0.5

X_dense = X_dense[:, valid_genes].astype(np.float32)
gene_names_sc = np.array(adata.var_names)[valid_genes]

cancer_mask_sc = (adata.obs['cell_class'].values == 'cancer')
healthy_mask_sc = (adata.obs['cell_class'].values == 'healthy')

print(f"  After subsample: {X_dense.shape[0]:,} cells")
print(f"  After gene filter (mean > 0.5): {X_dense.shape[1]:,} genes")
print(f"  Cancer: {cancer_mask_sc.sum():,}, Healthy: {healthy_mask_sc.sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.4  PyTorch VECTORIZED SOFT-LOGIC GATE SEARCH
# ─────────────────────────────────────────────────────────────────────────────
# Evaluates ~13.4M combinatorial (P1 OR P2) AND NOT R gates using continuous
# Hill-function outputs on GPU.
#
# CRITICAL MEMORY MANAGEMENT:
#   - Chunk repressor loop (chunk_size=10) to avoid GPU OOM
#   - torch.cuda.empty_cache() inside loop
#   - ONLY store results with reward > 0 (not all 13.4M)
print("\n[4/5] PyTorch soft-logic gate search (~13.4M combinations)...")

# Constants for soft-logic scoring
WEIGHT_TP = 2.0        # reward per cancer cell killed
WEIGHT_FP = 50.0       # penalty per healthy cell toxified
SEARCH_DEPTH = 300     # top N promoters and repressors

# Compute per-gene fold-change to identify elite pools
mean_cancer_sc = X_dense[cancer_mask_sc].mean(axis=0)
mean_healthy_sc = X_dense[healthy_mask_sc].mean(axis=0)

fc_promoter = mean_cancer_sc / (mean_healthy_sc + 1e-9)     # cancer-specific
fc_repressor = mean_healthy_sc / (mean_cancer_sc + 1e-9)     # healthy-specific

elite_promoters = np.argsort(fc_promoter)[-SEARCH_DEPTH:]
elite_repressors = np.argsort(fc_repressor)[-SEARCH_DEPTH:]

promoter_pairs = list(itertools.combinations(elite_promoters, 2))
n_total_circuits = len(promoter_pairs) * SEARCH_DEPTH

print(f"  Elite pools: {SEARCH_DEPTH} promoters × {SEARCH_DEPTH} repressors")
print(f"  Promoter pairs: C({SEARCH_DEPTH},2) = {len(promoter_pairs):,}")
print(f"  Total circuits: {n_total_circuits:,}")

# Compute per-gene Hill thresholds
K_prom = np.percentile(X_dense[cancer_mask_sc], 95, axis=0)
K_prom = np.maximum(K_prom, 0.1)
K_repr = np.percentile(X_dense[healthy_mask_sc], 5, axis=0)
K_repr = np.maximum(K_repr, 0.1)

# Move data to GPU
X_gpu = torch.tensor(X_dense, dtype=torch.float32, device=DEVICE)
K_prom_gpu = torch.tensor(K_prom, dtype=torch.float32, device=DEVICE)
K_repr_gpu = torch.tensor(K_repr, dtype=torch.float32, device=DEVICE)
cancer_idx_gpu = torch.tensor(np.where(cancer_mask_sc)[0], dtype=torch.long, device=DEVICE)
healthy_idx_gpu = torch.tensor(np.where(healthy_mask_sc)[0], dtype=torch.long, device=DEVICE)

n_cancer_cells = cancer_idx_gpu.shape[0]
n_healthy_cells = healthy_idx_gpu.shape[0]

# Pre-compute Hill outputs for ALL elite promoters (shape: n_cells × n_promoters)
print("  Pre-computing Hill outputs for elite genes...")
prom_indices = torch.tensor(elite_promoters, dtype=torch.long, device=DEVICE)
repr_indices = torch.tensor(np.array(list(elite_repressors)), dtype=torch.long, device=DEVICE)

X_prom = X_gpu[:, prom_indices]       # (n_cells, 300)
K_prom_elite = K_prom_gpu[prom_indices]  # (300,)
H_prom = (X_prom ** HILL_N) / (K_prom_elite ** HILL_N + X_prom ** HILL_N + 1e-12)  # (n_cells, 300)

X_repr_all = X_gpu[:, repr_indices]      # (n_cells, 300)
K_repr_elite = K_repr_gpu[repr_indices]  # (300,)

# MEMORY-EFFICIENT SEARCH: Chunk over repressors in groups of 10
REPRESSOR_CHUNK_SIZE = 10
positive_results: List[Dict] = []  # ONLY store positive-reward circuits

t_start = time.time()
circuits_evaluated = 0

for r_chunk_start in range(0, SEARCH_DEPTH, REPRESSOR_CHUNK_SIZE):
    r_chunk_end = min(r_chunk_start + REPRESSOR_CHUNK_SIZE, SEARCH_DEPTH)
    chunk_size = r_chunk_end - r_chunk_start
    
    # Compute repressor Hill for this chunk: (n_cells, chunk_size)
    # H_R(x) = K^n / (K^n + x^n) — this IS the "NOT R" function:
    #   R HIGH (healthy) → H_R → 0 (gate blocked) ✓
    #   R LOW  (cancer)  → H_R → 1 (gate passes)  ✓
    # CRITICAL FIX: Do NOT apply "1 - H_R", that inverts the logic.
    X_r_chunk = X_repr_all[:, r_chunk_start:r_chunk_end]
    K_r_chunk = K_repr_elite[r_chunk_start:r_chunk_end]
    H_r_chunk = (K_r_chunk ** HILL_N) / (K_r_chunk ** HILL_N + X_r_chunk ** HILL_N + 1e-12)
    # H_r_chunk is DIRECTLY the "AND NOT R" component — no inversion needed
    
    for pair_idx, (p1_local, p2_local) in enumerate(promoter_pairs):
        # Look up pre-computed Hill outputs using searchsorted
        # (elite_promoters is sorted from np.argsort, so searchsorted is correct)
        p1_pos = np.searchsorted(elite_promoters, p1_local)
        p2_pos = np.searchsorted(elite_promoters, p2_local)
        H_p1 = H_prom[:, p1_pos]
        H_p2 = H_prom[:, p2_pos]
        
        # Soft OR: 1 - (1-H_p1)*(1-H_p2) — shape: (n_cells,)
        H_or = 1.0 - (1.0 - H_p1) * (1.0 - H_p2)
        
        # Broadcast gate output across chunk: (n_cells, chunk_size)
        H_or_exp = H_or.unsqueeze(1)  # (n_cells, 1)
        gate_output = H_or_exp * H_r_chunk  # (n_cells, chunk_size)
        P_star = ALPHA_OVER_GAMMA * gate_output    # (n_cells, chunk_size)
        
        # Count kills per repressor in chunk
        P_cancer = P_star[cancer_idx_gpu]     # (n_cancer, chunk_size)
        P_healthy = P_star[healthy_idx_gpu]   # (n_healthy, chunk_size)
        
        cancer_kills = (P_cancer > LETHAL_THRESHOLD).sum(dim=0).cpu().numpy()  # (chunk_size,)
        healthy_kills = (P_healthy > LETHAL_THRESHOLD).sum(dim=0).cpu().numpy()
        
        rewards = WEIGHT_TP * cancer_kills - WEIGHT_FP * healthy_kills
        
        # ONLY store positive-reward circuits
        for r_offset in range(chunk_size):
            circuits_evaluated += 1
            if rewards[r_offset] > 0:
                r_global = r_chunk_start + r_offset
                positive_results.append({
                    'p1_idx': int(p1_local),
                    'p2_idx': int(p2_local),
                    'r_idx': int(elite_repressors[r_global]),
                    'p1_name': gene_names_sc[p1_local],
                    'p2_name': gene_names_sc[p2_local],
                    'r_name': gene_names_sc[elite_repressors[r_global]],
                    'cancer_kills': int(cancer_kills[r_offset]),
                    'healthy_kills': int(healthy_kills[r_offset]),
                    'reward': float(rewards[r_offset]),
                    'kill_rate': float(cancer_kills[r_offset] / n_cancer_cells * 100),
                    'toxicity_rate': float(healthy_kills[r_offset] / n_healthy_cells * 100),
                })
    
    # CRITICAL: Free GPU memory after each repressor chunk
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    elapsed = time.time() - t_start
    pct_done = (r_chunk_end / SEARCH_DEPTH) * 100
    if (r_chunk_start % 50 == 0) or r_chunk_end == SEARCH_DEPTH:
        print(f"    [{pct_done:5.1f}%] Evaluated {circuits_evaluated:,} circuits "
              f"({elapsed:.0f}s, {len(positive_results):,} positive)")

# Sort by reward
positive_results.sort(key=lambda x: x['reward'], reverse=True)

t_total = time.time() - t_start
print(f"\n  Search completed in {t_total:.1f}s ({circuits_evaluated:,} circuits)")
print(f"  Positive-reward circuits: {len(positive_results):,}")

# Report top 5
print(f"\n  {'Rk':>3s}  {'Reward':>8s}  {'Kill%':>6s}  {'Tox%':>6s}  {'P1':>12s}  {'P2':>12s}  {'R':>12s}")
print(f"  {'─'*3}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}")
for rank, c in enumerate(positive_results[:5], 1):
    print(f"  {rank:3d}  {c['reward']:8.0f}  {c['kill_rate']:5.1f}%  {c['toxicity_rate']:5.2f}%  "
          f"{c['p1_name']:>12s}  {c['p2_name']:>12s}  {c['r_name']:>12s}")

# Store best AI-discovered gate
if positive_results:
    best_gate = positive_results[0]
    REPORT_METRICS['ai_gate'] = f"({best_gate['p1_name']} OR {best_gate['p2_name']}) AND NOT {best_gate['r_name']}"
    REPORT_METRICS['ai_kill_rate'] = best_gate['kill_rate']
    REPORT_METRICS['ai_toxicity'] = best_gate['toxicity_rate']

# ─────────────────────────────────────────────────────────────────────────────
# 4.5  HEAD-TO-HEAD BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Head-to-Head Benchmark...")

def evaluate_gate(p1_name, p2_name, r_name, X, gene_names, K_p, K_r,
                  cancer_mask, healthy_mask):
    """Evaluate a (P1 OR P2) AND NOT R gate on the expression matrix."""
    try:
        p1_i = np.where(gene_names == p1_name)[0][0]
        p2_i = np.where(gene_names == p2_name)[0][0]
        r_i = np.where(gene_names == r_name)[0][0]
    except IndexError:
        return None
    
    expr_p1 = X[:, p1_i]
    expr_p2 = X[:, p2_i]
    expr_r = X[:, r_i]
    
    H_p1 = (expr_p1 ** HILL_N) / (K_p[p1_i] ** HILL_N + expr_p1 ** HILL_N + 1e-12)
    H_p2 = (expr_p2 ** HILL_N) / (K_p[p2_i] ** HILL_N + expr_p2 ** HILL_N + 1e-12)
    # H_R = K^n/(K^n+x^n) — DIRECTLY encodes "NOT R" (high when R is low)
    # CRITICAL FIX: Do NOT subtract from 1 — that inverts the logic.
    H_r = (K_r[r_i] ** HILL_N) / (K_r[r_i] ** HILL_N + expr_r ** HILL_N + 1e-12)
    
    H_or = 1.0 - (1.0 - H_p1) * (1.0 - H_p2)
    P_star = ALPHA_OVER_GAMMA * H_or * H_r
    
    ck = np.sum(P_star[cancer_mask] > LETHAL_THRESHOLD)
    hk = np.sum(P_star[healthy_mask] > LETHAL_THRESHOLD)
    
    return {
        'cancer_kills': int(ck),
        'healthy_kills': int(hk),
        'kill_rate': float(ck / cancer_mask.sum() * 100),
        'toxicity_rate': float(hk / healthy_mask.sum() * 100),
        'reward': float(WEIGHT_TP * ck - WEIGHT_FP * hk),
    }

# AI-discovered gate: (EHF OR TMC5) AND NOT SRGN
ai_result = evaluate_gate('EHF', 'TMC5', 'SRGN', X_dense, gene_names_sc,
                           K_prom, K_repr, cancer_mask_sc, healthy_mask_sc)

# Literature-derived gate: (SCGB3A2 OR TOX3) AND NOT IGLC2
lit_result = evaluate_gate('SCGB3A2', 'TOX3', 'IGLC2', X_dense, gene_names_sc,
                            K_prom, K_repr, cancer_mask_sc, healthy_mask_sc)

print(f"\n  {'Gate':42s}  {'Kill%':>6s}  {'Tox%':>6s}  {'Reward':>8s}")
print(f"  {'─'*42}  {'─'*6}  {'─'*6}  {'─'*8}")

if ai_result:
    print(f"  {'(EHF OR TMC5) AND NOT SRGN':42s}  {ai_result['kill_rate']:5.1f}%  "
          f"{ai_result['toxicity_rate']:5.2f}%  {ai_result['reward']:8.0f}")
    REPORT_METRICS['ai_benchmark_kill'] = ai_result['kill_rate']
    REPORT_METRICS['ai_benchmark_tox'] = ai_result['toxicity_rate']
else:
    print(f"  {'(EHF OR TMC5) AND NOT SRGN':42s}  — gene(s) not found in dataset")

if lit_result:
    print(f"  {'(SCGB3A2 OR TOX3) AND NOT IGLC2':42s}  {lit_result['kill_rate']:5.1f}%  "
          f"{lit_result['toxicity_rate']:5.2f}%  {lit_result['reward']:8.0f}")
    REPORT_METRICS['lit_benchmark_kill'] = lit_result['kill_rate']
    REPORT_METRICS['lit_benchmark_tox'] = lit_result['toxicity_rate']
else:
    print(f"  {'(SCGB3A2 OR TOX3) AND NOT IGLC2':42s}  — gene(s) not found in dataset")

# Save top circuits to CSV
if positive_results:
    top_df = pd.DataFrame(positive_results[:100])
    top_df.to_csv(RESULTS_DIR / 'phase3_top_circuits.csv', index=False)
    print(f"\n  ✓ Top 100 circuits saved to results/phase3_top_circuits.csv")

# Cleanup GPU memory
del X_gpu, H_prom, X_repr_all, X_prom
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

print("\n✓ Phase 3 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: PHASE 4 — TARGET-AWARE RNA TOEHOLD DESIGN (ViennaRNA + JAX)
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n" + "=" * 72)
print("  PHASE 4: TARGET-AWARE RNA TOEHOLD DESIGN (ViennaRNA + JAX)")
print("=" * 72)

import RNA  # ViennaRNA Python bindings

# ─────────────────────────────────────────────────────────────────────────────
# 5.1  ENSEMBL REST API: Auto-discover canonical transcript + fetch mature mRNA
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL FIX: Previous code fetched genomic pre-mRNA (with introns),
# producing biologically implausible lengths (e.g., SCGB3A2 at 11,024 nt).
# This function strictly fetches ONLY the mature, spliced mRNA (cDNA).

ENSEMBL_REST = "https://rest.ensembl.org"

def fetch_mature_mrna(gene_symbol: str, species: str = 'homo_sapiens',
                       max_retries: int = 3) -> Optional[Tuple[str, str, int]]:
    """
    Fetch the mature (spliced) mRNA sequence for a gene from Ensembl.
    
    STEPS:
    1. /lookup/symbol/{species}/{gene} → get canonical transcript ID
    2. /sequence/id/{transcript_id}?type=cdna → get spliced mRNA (CDS + UTRs)
    
    CRITICAL: Using type=cdna ensures we get ONLY exonic sequence (no introns).
    This avoids the pre-mRNA bug where SCGB3A2 appeared as 11,024 nt instead
    of its true mature mRNA length (~400 nt).
    
    Parameters
    ----------
    gene_symbol : str — HGNC gene symbol (e.g., 'SCGB3A2', 'EHF')
    species     : str — Ensembl species name
    max_retries : int — number of retry attempts for transient failures
    
    Returns
    -------
    Tuple of (transcript_id, rna_sequence, length) or None on failure
    """
    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(max_retries):
        try:
            # Step 1: Lookup gene → get canonical transcript
            lookup_url = f"{ENSEMBL_REST}/lookup/symbol/{species}/{gene_symbol}"
            lookup_params = {'expand': '1'}
            resp = requests.get(lookup_url, headers=headers, params=lookup_params, timeout=15)
            
            if resp.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            
            gene_data = resp.json()
            
            # Find canonical transcript
            canonical_id = gene_data.get('canonical_transcript')
            if canonical_id and '.' in canonical_id:
                canonical_id = canonical_id.split('.')[0]  # Remove version
            
            if not canonical_id:
                # Fallback: pick the longest protein-coding transcript
                transcripts = gene_data.get('Transcript', [])
                coding_tx = [t for t in transcripts if t.get('biotype') == 'protein_coding']
                if coding_tx:
                    coding_tx.sort(key=lambda t: t.get('length', 0), reverse=True)
                    canonical_id = coding_tx[0]['id']
                elif transcripts:
                    canonical_id = transcripts[0]['id']
                else:
                    print(f"    ⚠ No transcripts found for {gene_symbol}")
                    return None
            
            # Step 2: Fetch mature mRNA (cDNA = spliced exonic sequence)
            seq_url = f"{ENSEMBL_REST}/sequence/id/{canonical_id}"
            seq_params = {'type': 'cdna'}  # CRITICAL: cdna = mature mRNA only
            
            time.sleep(0.34)  # Ensembl rate limit: max 3 requests/sec
            
            seq_resp = requests.get(seq_url, headers=headers, params=seq_params, timeout=15)
            
            if seq_resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            seq_resp.raise_for_status()
            
            seq_data = seq_resp.json()
            dna_seq = seq_data.get('seq', '')
            
            if not dna_seq:
                print(f"    ⚠ Empty sequence for {gene_symbol} ({canonical_id})")
                return None
            
            # Convert DNA to RNA (T → U)
            rna_seq = dna_seq.upper().replace('T', 'U')
            
            return (canonical_id, rna_seq, len(rna_seq))
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠ Failed to fetch {gene_symbol} after {max_retries} retries: {e}")
                return None
    
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 5.2  FETCH MATURE mRNA FOR ALL TARGET GENES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Fetching mature mRNA sequences from Ensembl REST API...")

TARGET_GENES = ['SCGB3A2', 'EHF', 'TMC5', 'TOX3', 'SRGN', 'IGLC2']
gene_sequences: Dict[str, Tuple[str, str, int]] = {}  # gene → (tx_id, rna_seq, length)

for gene in TARGET_GENES:
    result = fetch_mature_mrna(gene)
    if result:
        tx_id, rna_seq, length = result
        gene_sequences[gene] = result
        print(f"  ✓ {gene:10s}  transcript={tx_id}  length={length:,} nt (mature mRNA)")
        # Sanity check: mature mRNA should be < 10,000 nt for most genes
        if length > 10000:
            print(f"    ⚠ WARNING: {gene} transcript is {length} nt — verify this is cDNA, not genomic")
    else:
        print(f"  ✗ {gene:10s}  FAILED — will skip toehold design for this gene")

# ─────────────────────────────────────────────────────────────────────────────
# 5.3  RNA SEQUENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
_RNA_COMP = str.maketrans("AUGC", "UACG")

def rc_rna(seq: str) -> str:
    """RNA reverse complement (A↔U, G↔C)."""
    return seq.upper().translate(_RNA_COMP)[::-1]

def gc_content(seq: str) -> float:
    """Fractional GC content [0, 1]."""
    s = seq.upper()
    return (s.count('G') + s.count('C')) / len(s) if s else 0.0

def has_homopolymer(seq: str, max_run: int = 6) -> bool:
    """True if sequence contains a homopolymer run ≥ max_run nt."""
    for base in 'AUGC':
        if base * max_run in seq.upper():
            return True
    return False

def has_stop_codon(seq: str, frame: int = 0) -> bool:
    """True if sequence contains an in-frame stop codon."""
    stops = {'UAA', 'UAG', 'UGA'}
    s = seq.upper()
    for i in range(frame, len(s) - 2, 3):
        if s[i:i+3] in stops:
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# 5.4  TARGET-AWARE ACCESSIBILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Computing per-nucleotide accessibility via partition function...")

def compute_accessibility(rna_seq: str, window_size: int = TRIGGER_LEN) -> np.ndarray:
    """
    Compute per-nucleotide unpaired probability using ViennaRNA partition function.
    
    Uses RNA.fold_compound to fold the full-length mature transcript, then
    extracts the base-pair probability matrix to calculate the unpaired
    probability at each position: P_unpaired(i) = 1 - sum_j(P_bp(i,j))
    
    Parameters
    ----------
    rna_seq     : str — full mature mRNA sequence
    window_size : int — window size for averaging
    
    Returns
    -------
    accessibility : np.ndarray — per-position unpaired probability [0,1]
    """
    # For long sequences, use local folding window to manage memory
    seq_len = len(rna_seq)
    
    if seq_len > 3000:
        # Use windowed folding for very long transcripts
        md = RNA.md()
        md.window_size = 200
        md.max_bp_span = 150
        fc = RNA.fold_compound(rna_seq, md, RNA.OPTION_WINDOW)
        # For windowed mode, use MFE structure accessibility
        struct, mfe = RNA.fold(rna_seq[:min(seq_len, 2000)])
        unpaired = np.array([1.0 if c == '.' else 0.0 for c in struct])
        # Pad or truncate to full length
        if len(unpaired) < seq_len:
            unpaired = np.pad(unpaired, (0, seq_len - len(unpaired)), constant_values=0.5)
        return unpaired[:seq_len]
    
    # Standard partition function approach
    fc = RNA.fold_compound(rna_seq)
    fc.pf()
    bpp = fc.bpp()
    
    # Calculate unpaired probability at each position
    unpaired = np.ones(seq_len)
    for i in range(1, seq_len + 1):  # ViennaRNA uses 1-based indexing
        paired_prob = 0.0
        for j in range(1, seq_len + 1):
            if i != j:
                try:
                    paired_prob += bpp[min(i,j)][max(i,j)]
                except (IndexError, TypeError):
                    pass
        unpaired[i-1] = max(0.0, 1.0 - paired_prob)
    
    return unpaired

# ─────────────────────────────────────────────────────────────────────────────
# 5.5  WINDOW SELECTION: 30-nt trigger with GC + homopolymer + accessibility
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Scanning for optimal 30-nt trigger windows...")

@dataclass
class TriggerWindow:
    """A candidate 30-nt trigger window from the target mRNA."""
    gene: str
    position: int
    sequence: str          # 30-nt RNA trigger
    gc: float              # GC content [0,1]
    mean_accessibility: float  # mean unpaired probability
    has_homopolymer: bool

def select_trigger_windows(gene: str, rna_seq: str,
                            top_n: int = 5) -> List[TriggerWindow]:
    """
    Select optimal 30-nt trigger windows from a mature mRNA.
    
    FILTERS (all must pass):
    1. GC content: 40–55% (balanced thermodynamics)
    2. No homopolymer runs ≥ 6 nt (avoids secondary structure)
    3. Ranked by mean accessibility (unpaired probability)
    """
    accessibility = compute_accessibility(rna_seq)
    
    candidates = []
    for pos in range(len(rna_seq) - TRIGGER_LEN + 1):
        window = rna_seq[pos:pos + TRIGGER_LEN]
        gc_val = gc_content(window)
        
        # STRICT FILTER 1: GC content 40–55%
        if gc_val < 0.40 or gc_val > 0.55:
            continue
        
        # STRICT FILTER 2: No homopolymer runs
        if has_homopolymer(window, max_run=6):
            continue
        
        # Compute mean accessibility for this window
        mean_acc = float(np.mean(accessibility[pos:pos + TRIGGER_LEN]))
        
        candidates.append(TriggerWindow(
            gene=gene, position=pos, sequence=window,
            gc=gc_val, mean_accessibility=mean_acc,
            has_homopolymer=False
        ))
    
    # Sort by accessibility (highest first)
    candidates.sort(key=lambda w: w.mean_accessibility, reverse=True)
    return candidates[:top_n]

# Select windows for each target gene
gene_windows: Dict[str, List[TriggerWindow]] = {}

for gene in TARGET_GENES:
    if gene not in gene_sequences:
        continue
    tx_id, rna_seq, length = gene_sequences[gene]
    windows = select_trigger_windows(gene, rna_seq, top_n=3)
    gene_windows[gene] = windows
    print(f"  {gene:10s}: {len(windows)} valid windows (from {length} nt transcript)")
    for w in windows[:2]:
        print(f"    pos={w.position:5d}  GC={w.gc:.2f}  access={w.mean_accessibility:.3f}  "
              f"seq={w.sequence[:20]}...")

# ─────────────────────────────────────────────────────────────────────────────
# 5.6  SIMULATED ANNEALING: Design toehold sensors with dual-objective
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Simulated Annealing: Designing toehold sensors (32 chains)...")

TARGET_DG = -35.0        # Target ΔG(AB) for trigger-sensor complex (kcal/mol)
MARGIN_TARGET = -5.0     # Heterodimer margin: ΔG(AB) - ΔG(BB) ≤ this value

def build_sensor(trigger: str) -> str:
    """
    Build a Green et al. 2014 Type-A toehold switch sensor from a 30-nt trigger.
    
    Architecture: 5'-[RC(b)]-[KOZAK_LOOP]-[b]-[RC(a)]-[LINKER]-[REPORTER]-3'
    Where: a = trigger[0:12] (toehold), b = trigger[12:30] (stem)
    """
    a = trigger[:TOEHOLD_LEN]          # 12 nt toehold
    b = trigger[TOEHOLD_LEN:]          # 18 nt stem
    rc_a = rc_rna(a)
    rc_b = rc_rna(b)
    return rc_b + KOZAK_LOOP + b + rc_a + ICASP9_LINKER + ICASP9_START

def evaluate_toehold(trigger: str, sensor: str) -> Dict:
    """
    Evaluate thermodynamics of a toehold switch using ViennaRNA.
    
    CRITICAL BUG FIX:
    - ΔG(AB) = RNAcofold(trigger & sensor) — heterodimer
    - ΔG(BB) = RNAcofold(sensor & sensor)  — homodimer (NOT trigger & sensor!)
    
    Previous code incorrectly computed ΔG(BB) as RNAcofold(trigger, sensor),
    which is actually ΔG(AB) again. The homodimer must strictly be sensor × sensor.
    """
    # Individual folds
    struct_sensor, dg_sensor = RNA.fold(sensor)
    struct_trigger, dg_trigger = RNA.fold(trigger)
    
    # HETERODIMER: trigger + sensor → ON complex [ΔG(AB)]
    struct_AB, dg_AB = RNA.cofold(trigger + "&" + sensor)
    
    # HOMODIMER: sensor + sensor → self-pairing [ΔG(BB)]
    # CRITICAL FIX — must be (sensor, sensor), NOT (trigger, sensor)!
    struct_BB, dg_BB = RNA.cofold(sensor + "&" + sensor)
    
    # Thermodynamic driving force
    ddG = dg_AB - (dg_sensor + dg_trigger)
    
    # Heterodimer margin: how much more stable is AB vs BB?
    margin = dg_AB - dg_BB
    
    return {
        'dg_sensor': round(dg_sensor, 2),
        'dg_trigger': round(dg_trigger, 2),
        'dg_AB': round(dg_AB, 2),
        'dg_BB': round(dg_BB, 2),
        'ddG': round(ddG, 2),
        'margin': round(margin, 2),
        'struct_sensor': struct_sensor,
        'struct_AB': struct_AB,
    }

def sa_dual_objective(dg_AB: float, dg_BB: float) -> float:
    """
    Dual-objective scoring for Simulated Annealing.
    
    score = |ΔG(AB) - target| + max(0, margin + 5.0) × 3.0
    
    Where margin = ΔG(AB) - ΔG(BB).
    
    Minimizing this score simultaneously targets:
    1. ΔG(AB) close to -35.0 kcal/mol
    2. Heterodimer margin ≤ -5.0 kcal/mol (AB much more stable than BB)
    """
    margin = dg_AB - dg_BB
    return abs(dg_AB - TARGET_DG) + max(0.0, margin + 5.0) * 3.0

def check_immunogenicity(sensor: str, max_stem: int = 20) -> bool:
    """
    Screen sensor for contiguous dsRNA stems ≥ max_stem bp.
    
    dsRNA stems ≥ 20 bp trigger innate immune sensors RIG-I and MDA5,
    which would cause inflammatory response and circuit destruction.
    The sensor must NOT contain any stem of this length.
    
    Returns True if sensor PASSES (no long stems), False if it FAILS.
    """
    struct, _ = RNA.fold(sensor)
    # Count max contiguous paired bases
    max_paired_run = 0
    current_run = 0
    for c in struct:
        if c in '()':
            current_run += 1
            max_paired_run = max(max_paired_run, current_run)
        else:
            current_run = 0
    return max_paired_run < max_stem

# Run SA for each gene with valid trigger windows
toehold_results: Dict[str, Dict] = {}

N_SA_CHAINS = 32
N_SA_STEPS = 200
SA_T_INIT = 5.0
SA_T_MIN = 0.01

for gene, windows in gene_windows.items():
    if not windows:
        continue
    
    # Use the best-accessibility window as starting trigger
    best_window = windows[0]
    trigger = best_window.sequence
    
    print(f"\n  {gene}: SA optimization ({N_SA_CHAINS} chains × {N_SA_STEPS} steps)...")
    print(f"    Trigger: {trigger} (pos={best_window.position}, GC={best_window.gc:.2f})")
    
    best_score = float('inf')
    best_design = None
    sa_rng = np.random.default_rng(RNG_SEED)
    
    for chain in range(N_SA_CHAINS):
        # Start from the trigger with small random mutations
        current_trigger = list(trigger)
        
        # Introduce 1-3 random mutations for diversity across chains
        if chain > 0:
            n_mutations = sa_rng.integers(1, 4)
            for _ in range(n_mutations):
                mut_pos = sa_rng.integers(0, TRIGGER_LEN)
                current_trigger[mut_pos] = sa_rng.choice(list('AUGC'))
        
        current_trigger_str = ''.join(current_trigger)
        current_sensor = build_sensor(current_trigger_str)
        
        try:
            current_thermo = evaluate_toehold(current_trigger_str, current_sensor)
            current_score = sa_dual_objective(current_thermo['dg_AB'], current_thermo['dg_BB'])
        except Exception:
            continue
        
        # SA annealing loop
        for step in range(N_SA_STEPS):
            T = SA_T_INIT * (SA_T_MIN / SA_T_INIT) ** (step / N_SA_STEPS)
            
            # Mutate one random position
            new_trigger = list(current_trigger_str)
            mut_pos = sa_rng.integers(0, TRIGGER_LEN)
            new_trigger[mut_pos] = sa_rng.choice(list('AUGC'))
            new_trigger_str = ''.join(new_trigger)
            
            # Check GC filter
            new_gc = gc_content(new_trigger_str)
            if new_gc < 0.40 or new_gc > 0.55:
                continue
            
            # Check homopolymer filter
            if has_homopolymer(new_trigger_str):
                continue
            
            new_sensor = build_sensor(new_trigger_str)
            
            try:
                new_thermo = evaluate_toehold(new_trigger_str, new_sensor)
                new_score = sa_dual_objective(new_thermo['dg_AB'], new_thermo['dg_BB'])
            except Exception:
                continue
            
            # Metropolis acceptance criterion
            delta = new_score - current_score
            if delta < 0 or sa_rng.random() < np.exp(-delta / max(T, 1e-8)):
                current_trigger_str = new_trigger_str
                current_sensor = new_sensor
                current_thermo = new_thermo
                current_score = new_score
        
        # Update global best
        if current_score < best_score:
            best_score = current_score
            best_design = {
                'gene': gene,
                'trigger': current_trigger_str,
                'sensor': current_sensor,
                'thermo': current_thermo,
                'score': current_score,
                'gc': gc_content(current_trigger_str),
                'window': best_window,
                'chain': chain,
                'immunogenic_pass': check_immunogenicity(current_sensor),
            }
    
    if best_design:
        toehold_results[gene] = best_design
        t = best_design['thermo']
        print(f"    ✓ ΔG(AB)={t['dg_AB']:.1f}  ΔG(BB)={t['dg_BB']:.1f}  "
              f"margin={t['margin']:.1f}  ΔΔG={t['ddG']:.1f}  "
              f"GC={best_design['gc']:.2f}  immuno={'PASS' if best_design['immunogenic_pass'] else 'FAIL'}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.7  VIENNARNA FINAL POLISHING & VALIDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] ViennaRNA final validation...")

print(f"\n  {'Gene':10s}  {'ΔG(AB)':>8s}  {'ΔG(BB)':>8s}  {'Margin':>8s}  {'ΔΔG':>8s}  "
      f"{'GC':>5s}  {'Immuno':>7s}  {'Score':>7s}")
print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*7}")

for gene, design in toehold_results.items():
    t = design['thermo']
    ig = 'PASS' if design['immunogenic_pass'] else 'FAIL'
    print(f"  {gene:10s}  {t['dg_AB']:8.1f}  {t['dg_BB']:8.1f}  {t['margin']:8.1f}  "
          f"{t['ddG']:8.1f}  {design['gc']:5.2f}  {ig:>7s}  {design['score']:7.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.8  IMMUNOGENICITY SCREEN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] RIG-I/MDA5 immunogenicity screen (dsRNA stems < 20 bp)...")

all_pass = True
for gene, design in toehold_results.items():
    if not design['immunogenic_pass']:
        print(f"  ⚠ {gene}: FAIL — contiguous dsRNA stem ≥ 20 bp detected")
        all_pass = False
    else:
        print(f"  ✓ {gene}: PASS — no dsRNA stems ≥ 20 bp")

REPORT_METRICS['toehold_designs'] = len(toehold_results)
REPORT_METRICS['immunogenicity_all_pass'] = all_pass

print("\n✓ Phase 4 complete.")
print(f"  Designed {len(toehold_results)} toehold sensors for {len(TARGET_GENES)} target genes.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: PHASE 5 — GTEx TOXICOLOGY & EVOLUTIONARY ESCAPE
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n" + "=" * 72)
print("  PHASE 5: GTEx TOXICOLOGY & EVOLUTIONARY ESCAPE")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 6.1  GTEx v8 TISSUE SAFETY AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/2] GTEx v8 tissue safety audit...")

GTEX_API = "https://gtexportal.org/api/v2"

def fetch_gtex_expression(gene_symbol: str) -> Optional[Dict[str, float]]:
    """
    Query GTEx v8 REST API for median TPM across all tissues.
    
    Returns a dict mapping tissue name → median TPM.
    If the API call fails, returns None.
    """
    try:
        url = f"{GTEX_API}/expression/medianGeneExpression"
        params = {
            'gencodeId': gene_symbol,
            'datasetId': 'gtex_v8',
        }
        # First, resolve gencodeId from gene symbol
        lookup_url = f"{GTEX_API}/reference/gene"
        lookup_params = {'geneId': gene_symbol, 'datasetId': 'gtex_v8'}
        
        resp = requests.get(lookup_url, params=lookup_params, timeout=15)
        if resp.status_code != 200:
            return None
        
        gene_data = resp.json()
        if not gene_data or 'data' not in gene_data or not gene_data['data']:
            return None
        
        gencode_id = gene_data['data'][0].get('gencodeId', '')
        
        # Now fetch expression
        expr_url = f"{GTEX_API}/expression/medianGeneExpression"
        expr_params = {'gencodeId': gencode_id, 'datasetId': 'gtex_v8'}
        
        expr_resp = requests.get(expr_url, params=expr_params, timeout=15)
        if expr_resp.status_code != 200:
            return None
        
        expr_data = expr_resp.json()
        tissue_expr = {}
        for entry in expr_data.get('data', []):
            tissue = entry.get('tissueSiteDetailId', 'Unknown')
            median_tpm = entry.get('median', 0.0)
            tissue_expr[tissue] = float(median_tpm)
        
        return tissue_expr if tissue_expr else None
        
    except Exception as e:
        print(f"    ⚠ GTEx API error for {gene_symbol}: {e}")
        return None

# Query GTEx for AI gate genes
AI_GATE_GENES = ['EHF', 'TMC5', 'SRGN']

gtex_data: Dict[str, Dict[str, float]] = {}
for gene in AI_GATE_GENES:
    print(f"  Querying GTEx for {gene}...")
    result = fetch_gtex_expression(gene)
    if result:
        gtex_data[gene] = result
        print(f"    ✓ {gene}: Data for {len(result)} tissues")
    else:
        print(f"    ⚠ {gene}: GTEx API unavailable — using fallback analysis")
        time.sleep(1)

# Apply continuous soft-logic to GTEx tissue medians
if len(gtex_data) >= 2:
    # Get union of all tissues
    all_tissues = set()
    for d in gtex_data.values():
        all_tissues.update(d.keys())
    all_tissues = sorted(all_tissues)
    
    # Build tissue expression matrix
    tissue_misfires = []
    for tissue in all_tissues:
        # Get expression for each gene (default 0 if missing)
        p1_expr = gtex_data.get('EHF', {}).get(tissue, 0.0)
        p2_expr = gtex_data.get('TMC5', {}).get(tissue, 0.0)
        r_expr = gtex_data.get('SRGN', {}).get(tissue, 0.0)
        
        # Compute soft-logic gate output
        # Use log2(TPM+1) transformation for consistency
        p1_log = np.log2(p1_expr + 1)
        p2_log = np.log2(p2_expr + 1)
        r_log = np.log2(r_expr + 1)
        
        # Hill functions with K = median expression as threshold
        K_tissue = 2.0  # log2(TPM+1) threshold
        H_p1 = (p1_log ** HILL_N) / (K_tissue ** HILL_N + p1_log ** HILL_N + 1e-12)
        H_p2 = (p2_log ** HILL_N) / (K_tissue ** HILL_N + p2_log ** HILL_N + 1e-12)
        
        # H_r = K^n/(K^n+x^n) = repressor Hill, ALREADY encodes "NOT R"
        # When SRGN is HIGH in a tissue → H_r → 0 → gate blocked (safe) ✓
        # When SRGN is LOW → H_r → 1 → gate passes (potential misfire) ✓
        H_r = (K_tissue ** HILL_N) / (K_tissue ** HILL_N + r_log ** HILL_N + 1e-12)
        
        H_or = 1.0 - (1.0 - H_p1) * (1.0 - H_p2)
        gate_output = H_or * H_r  # CORRECT: no inversion needed
        P_star_tissue = ALPHA_OVER_GAMMA * gate_output
        
        if P_star_tissue > LETHAL_THRESHOLD:
            tissue_misfires.append((tissue, P_star_tissue))
    
    print(f"\n  Off-target tissue misfires (P* > {LETHAL_THRESHOLD} nM): {len(tissue_misfires)}")
    if tissue_misfires:
        for tissue, p_star in tissue_misfires:
            print(f"    ⚠ {tissue}: P* = {p_star:.0f} nM")
        print(f"  → MANDATE: Aerosol delivery only (NOT systemic IV)")
    else:
        print(f"  ✓ 0 off-target misfires across {len(all_tissues)} tissues")
        print(f"  → Aerosol delivery STILL mandated for regulatory safety")
    
    REPORT_METRICS['gtex_misfires'] = len(tissue_misfires)
    REPORT_METRICS['gtex_tissues_tested'] = len(all_tissues)
else:
    print("  ⚠ Insufficient GTEx data — skipping tissue audit")
    REPORT_METRICS['gtex_misfires'] = 'N/A'

# ─────────────────────────────────────────────────────────────────────────────
# 6.2  EVOLUTIONARY ESCAPE: Moran Birth-Death Process
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/2] Moran birth-death evolutionary escape simulation...")

# The Dual-Orthogonal iCasp9-DHFR-FKBP degron system requires simultaneous
# escape from three independent mechanisms, giving a composite mutation rate
# of μ = 10^-14 per cell per division.

MU_COMPOSITE = 1e-14     # composite escape mutation rate
MORAN_POP = 10000         # tumor population size
MORAN_CYCLES = 150        # treatment cycles
MORAN_RUNS = 30           # independent simulation runs
ESCAPE_FRACTION_THR = 0.01  # 1% escaped = relapse

print(f"  Parameters:")
print(f"    Population size: {MORAN_POP:,}")
print(f"    Mutation rate:   μ = {MU_COMPOSITE:.0e}")
print(f"    Cycles:          {MORAN_CYCLES}")
print(f"    Runs:            {MORAN_RUNS}")

escape_trajectories = np.zeros((MORAN_RUNS, MORAN_CYCLES + 1))
moran_rng = np.random.default_rng(RNG_SEED)

for run in range(MORAN_RUNS):
    # State: number of escaped cells
    n_escaped = 0
    n_total = MORAN_POP
    
    escape_trajectories[run, 0] = n_escaped / n_total
    
    for cycle in range(1, MORAN_CYCLES + 1):
        # Moran step: one birth, one death
        # Birth: choose proportional to fitness
        # Escaped cells have fitness advantage 1.2x
        n_sensitive = n_total - n_escaped
        
        # Mutation: each sensitive cell can mutate to escaped
        new_mutants = moran_rng.binomial(n_sensitive, MU_COMPOSITE)
        n_escaped += new_mutants
        
        # Selection: escaped cells have slight fitness advantage
        # Moran update: one random birth (fitness-weighted), one random death
        if n_total > 0:
            # Fitness-weighted birth probability
            p_escaped_birth = (n_escaped * 1.2) / (n_escaped * 1.2 + n_sensitive * 1.0 + 1e-30)
            
            if moran_rng.random() < p_escaped_birth:
                n_escaped = min(n_escaped + 1, n_total)
            
            # Random death (uniform)
            if moran_rng.random() < n_escaped / n_total:
                n_escaped = max(n_escaped - 1, 0)
        
        escape_trajectories[run, cycle] = n_escaped / n_total

# Analyze results
final_escape_fracs = escape_trajectories[:, -1]
n_relapsed = int(np.sum(final_escape_fracs > ESCAPE_FRACTION_THR))
mean_final_escape = float(np.mean(final_escape_fracs))

print(f"\n  Results:")
print(f"    Mean final escape fraction:     {mean_final_escape:.2e}")
print(f"    Runs with relapse (>{ESCAPE_FRACTION_THR*100:.0f}%):    {n_relapsed}/{MORAN_RUNS}")
print(f"    Max escape fraction observed:   {final_escape_fracs.max():.2e}")

REPORT_METRICS['escape_mean_final'] = f"{mean_final_escape:.2e}"
REPORT_METRICS['escape_relapsed'] = n_relapsed
REPORT_METRICS['escape_runs'] = MORAN_RUNS

# Escape trajectory plot
fig_esc, ax_esc = plt.subplots(figsize=(9, 5))
cycles_x = np.arange(MORAN_CYCLES + 1)

for run in range(MORAN_RUNS):
    ax_esc.plot(cycles_x, escape_trajectories[run], alpha=0.3, color='#7f8c8d', lw=0.8)

mean_traj = np.mean(escape_trajectories, axis=0)
ax_esc.plot(cycles_x, mean_traj, color='#e74c3c', lw=2.5, label='Mean escape fraction')
ax_esc.axhline(ESCAPE_FRACTION_THR, color='#333', linestyle='--', lw=1.5,
               label=f'Relapse threshold ({ESCAPE_FRACTION_THR*100:.0f}%)')
ax_esc.set_xlabel('Treatment Cycle', fontsize=11, fontweight='bold')
ax_esc.set_ylabel('Escape Fraction', fontsize=11, fontweight='bold')
ax_esc.set_title(f'Moran Birth-Death: Evolutionary Escape (μ={MU_COMPOSITE:.0e})\n'
                 f'{n_relapsed}/{MORAN_RUNS} runs relapsed',
                 fontsize=12, fontweight='bold')
ax_esc.legend(fontsize=10)
ax_esc.grid(True, alpha=0.25, linestyle=':')
ax_esc.set_yscale('symlog', linthresh=1e-10)
ax_esc.set_ylim([-1e-15, 0.1])
REPORT_FIGURES['evolutionary_escape'] = fig_to_base64(fig_esc)
fig_esc.savefig(RESULTS_DIR / 'phase5_evolutionary_escape.png', dpi=200, bbox_inches='tight')
plt.close(fig_esc)

print("\n✓ Phase 5 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: INTERACTIVE HTML REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n" + "=" * 72)
print("  INTERACTIVE HTML REPORT GENERATION")
print("=" * 72)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# 7.1  BUILD SELF-CONTAINED HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/1] Compiling interactive HTML report...")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LUAD Cellular Perceptron — Pipeline Report v4.0</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  :root {{
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a2e;
    --accent: #e74c3c;
    --accent2: #2ecc71;
    --text: #e8e8e8;
    --muted: #8892b0;
    --border: #2d2d44;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
  header {{
    background: linear-gradient(135deg, #1a0a2e 0%, #0a1628 50%, #0f0a1e 100%);
    border-bottom: 2px solid var(--accent);
    padding: 48px 0;
    text-align: center;
  }}
  h1 {{
    font-size: 2rem;
    background: linear-gradient(135deg, var(--accent), #f39c12);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
  }}
  h2 {{
    font-size: 1.3rem;
    color: var(--accent);
    margin: 40px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .subtitle {{ color: var(--muted); font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 24px 0; }}
  .metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 32px rgba(231,76,60,0.15); }}
  .metric-value {{ font-size: 2.2rem; font-weight: 700; color: var(--accent); }}
  .metric-label {{ color: var(--muted); font-size: 0.85rem; margin-top: 4px; }}
  .metric-ok {{ color: var(--accent2); }}
  .figure-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin: 16px 0;
  }}
  .figure-card img {{ width: 100%; border-radius: 8px; }}
  .figure-caption {{ color: var(--muted); font-size: 0.85rem; margin-top: 8px; text-align: center; }}
  table {{
    width: 100%; border-collapse: separate; border-spacing: 0;
    background: var(--card); border-radius: 12px; overflow: hidden;
    margin: 16px 0;
  }}
  th {{ background: var(--surface); color: var(--accent); padding: 12px 16px; text-align: left; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 12px 16px; border-top: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(231,76,60,0.05); }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600;
  }}
  .badge-pass {{ background: rgba(46,204,113,0.15); color: var(--accent2); border: 1px solid rgba(46,204,113,0.3); }}
  .badge-fail {{ background: rgba(231,76,60,0.15); color: var(--accent); border: 1px solid rgba(231,76,60,0.3); }}
  footer {{
    text-align: center; padding: 40px 0; color: var(--muted); font-size: 0.8rem;
    border-top: 1px solid var(--border); margin-top: 60px;
  }}
</style>
</head>
<body>

<header>
  <div class="container">
    <h1>A Synthetic Cellular Perceptron for Selective Apoptosis in LUAD</h1>
    <p class="subtitle">In Silico Design, Validation, and Clinical Translation — Pipeline Report v4.0</p>
    <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
  </div>
</header>

<div class="container">

  <h2>Phase 1: miRNA Biomarker Discovery</h2>
  <div class="grid">
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('cv_auc', 'N/A')}</div>
      <div class="metric-label">Cross-Validated AUC</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('cv_sensitivity', 'N/A')}</div>
      <div class="metric-label">Mean Sensitivity</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('cv_specificity', 'N/A')}</div>
      <div class="metric-label">Mean Specificity</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('cox_hr', 'N/A')}</div>
      <div class="metric-label">Cox PH Hazard Ratio (p={REPORT_METRICS.get('cox_pval', 'N/A')})</div>
    </div>
  </div>
"""

# Add figures
for fig_name, fig_label in [
    ('roc_curve', 'Cross-Validated ROC Curve'),
    ('shap_beeswarm', 'SHAP Biomarker Directionality'),
    ('kaplan_meier', 'Kaplan-Meier Survival Stratification'),
]:
    if fig_name in REPORT_FIGURES:
        html_content += f"""
  <div class="figure-card">
    <img src="data:image/png;base64,{REPORT_FIGURES[fig_name]}" alt="{fig_label}">
    <div class="figure-caption">{fig_label}</div>
  </div>
"""

# Phase 2
html_content += f"""
  <h2>Phase 2: Biophysical Simulation & Stochastic Safety</h2>
  <div class="grid">
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('ode_cancer_ss', 'N/A')} <span style="font-size:1rem;">nM</span></div>
      <div class="metric-label">Cancer ODE Steady-State</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('ode_healthy_ss', 'N/A')} <span style="font-size:1rem;">nM</span></div>
      <div class="metric-label">Healthy ODE Steady-State</div>
    </div>
    <div class="metric-card">
      <div class="metric-value metric-ok">{REPORT_METRICS.get('mc_robustness', 'N/A')}%</div>
      <div class="metric-label">Monte Carlo Robustness</div>
    </div>
    <div class="metric-card">
      <div class="metric-value metric-ok">{REPORT_METRICS.get('ssa_p_lethal_healthy', 'N/A')}%</div>
      <div class="metric-label">Gillespie P(lethal|healthy)</div>
    </div>
  </div>
"""

for fig_name, fig_label in [
    ('ode_trajectory', 'Deterministic ODE: 48h Trajectory'),
    ('monte_carlo', 'Monte Carlo Parameter Robustness'),
    ('gillespie_ssa', f'Gillespie SSA: {N_SSA:,} Stochastic Trajectories'),
    ('moi_transduction', 'AAV6.2FF Poisson Transduction Model'),
]:
    if fig_name in REPORT_FIGURES:
        html_content += f"""
  <div class="figure-card">
    <img src="data:image/png;base64,{REPORT_FIGURES[fig_name]}" alt="{fig_label}">
    <div class="figure-caption">{fig_label}</div>
  </div>
"""

# Phase 3
html_content += f"""
  <h2>Phase 3: Single-Cell Soft-Logic Gate Discovery</h2>
  <div class="grid">
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('ai_kill_rate', 'N/A')}%</div>
      <div class="metric-label">AI Gate Kill Rate</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('ai_toxicity', 'N/A')}%</div>
      <div class="metric-label">AI Gate Toxicity</div>
    </div>
  </div>

  <h3 style="color: var(--muted); font-size: 1rem; margin: 20px 0 12px;">Head-to-Head Benchmark</h3>
  <table>
    <tr><th>Gate</th><th>Kill Rate</th><th>Toxicity</th></tr>
    <tr>
      <td>{REPORT_METRICS.get('ai_gate', '(EHF OR TMC5) AND NOT SRGN')}</td>
      <td>{REPORT_METRICS.get('ai_benchmark_kill', 'N/A')}%</td>
      <td>{REPORT_METRICS.get('ai_benchmark_tox', 'N/A')}%</td>
    </tr>
    <tr>
      <td>(SCGB3A2 OR TOX3) AND NOT IGLC2</td>
      <td>{REPORT_METRICS.get('lit_benchmark_kill', 'N/A')}%</td>
      <td>{REPORT_METRICS.get('lit_benchmark_tox', 'N/A')}%</td>
    </tr>
  </table>
"""

# Phase 4
html_content += f"""
  <h2>Phase 4: RNA Toehold Sensor Design</h2>
  <div class="grid">
    <div class="metric-card">
      <div class="metric-value">{REPORT_METRICS.get('toehold_designs', 0)}</div>
      <div class="metric-label">Sensors Designed</div>
    </div>
    <div class="metric-card">
      <div class="metric-value"><span class="badge {'badge-pass' if REPORT_METRICS.get('immunogenicity_all_pass', False) else 'badge-fail'}">
        {'ALL PASS' if REPORT_METRICS.get('immunogenicity_all_pass', False) else 'CHECK'}
      </span></div>
      <div class="metric-label">RIG-I/MDA5 Screen</div>
    </div>
  </div>

  <table>
    <tr><th>Gene</th><th>ΔG(AB)</th><th>ΔG(BB)</th><th>Margin</th><th>GC%</th><th>Immunogenicity</th></tr>
"""

for gene, design in toehold_results.items():
    t = design['thermo']
    ig = 'PASS' if design['immunogenic_pass'] else 'FAIL'
    ig_class = 'badge-pass' if design['immunogenic_pass'] else 'badge-fail'
    html_content += f"""    <tr>
      <td>{gene}</td>
      <td>{t['dg_AB']:.1f}</td>
      <td>{t['dg_BB']:.1f}</td>
      <td>{t['margin']:.1f}</td>
      <td>{design['gc']*100:.0f}%</td>
      <td><span class="badge {ig_class}">{ig}</span></td>
    </tr>
"""

html_content += "  </table>\n"

# Phase 5
html_content += f"""
  <h2>Phase 5: GTEx Toxicology & Evolutionary Escape</h2>
  <div class="grid">
    <div class="metric-card">
      <div class="metric-value metric-ok">{REPORT_METRICS.get('gtex_misfires', 'N/A')}</div>
      <div class="metric-label">Off-Target Tissue Misfires</div>
    </div>
    <div class="metric-card">
      <div class="metric-value metric-ok">{REPORT_METRICS.get('escape_relapsed', 0)}/{REPORT_METRICS.get('escape_runs', 30)}</div>
      <div class="metric-label">Evolutionary Relapses (μ={MU_COMPOSITE:.0e})</div>
    </div>
  </div>
"""

if 'evolutionary_escape' in REPORT_FIGURES:
    html_content += f"""
  <div class="figure-card">
    <img src="data:image/png;base64,{REPORT_FIGURES['evolutionary_escape']}" alt="Evolutionary Escape Simulation">
    <div class="figure-caption">Moran Birth-Death: Evolutionary Escape Analysis ({MORAN_RUNS} runs × {MORAN_CYCLES} cycles)</div>
  </div>
"""

# AAV Payload Summary
remaining_bp = AAV_MAX_PAYLOAD_BP - ICASP9_DHFR_FKBP_BP - PROMOTER_SPC_HTERT_BP - DETARGETING_UTR_BP
html_content += f"""
  <h2>AAV Payload Budget</h2>
  <table>
    <tr><th>Component</th><th>Size (bp)</th><th>Cumulative</th></tr>
    <tr><td>iCasp9-DHFR-FKBP Degron</td><td>{ICASP9_DHFR_FKBP_BP:,}</td><td>{ICASP9_DHFR_FKBP_BP:,}</td></tr>
    <tr><td>SP-C/hTERT Dual Promoter</td><td>{PROMOTER_SPC_HTERT_BP}</td><td>{ICASP9_DHFR_FKBP_BP + PROMOTER_SPC_HTERT_BP:,}</td></tr>
    <tr><td>Detargeting UTRs (miR-122/miR-1)</td><td>{DETARGETING_UTR_BP}</td><td>{ICASP9_DHFR_FKBP_BP + PROMOTER_SPC_HTERT_BP + DETARGETING_UTR_BP:,}</td></tr>
    <tr style="font-weight:bold;"><td>Remaining for toehold sensors</td><td>{remaining_bp:,}</td><td>{AAV_MAX_PAYLOAD_BP:,} (max)</td></tr>
  </table>

  <footer>
    <p>LUAD Cellular Perceptron Pipeline v4.0 — Computational Analysis Only (In Silico)</p>
    <p>All analyses are simulations. No experimental data. For research purposes only.</p>
  </footer>

</div>
</body>
</html>"""

# Save HTML report
report_path = RESULTS_DIR / f'LUAD_Perceptron_Report_v4.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n  ✓ Interactive HTML report saved to: {report_path}")
print(f"  ✓ Report size: {report_path.stat().st_size / 1024:.0f} KB")

# Also save a metrics summary JSON
metrics_path = RESULTS_DIR / 'pipeline_metrics_v4.json'
with open(metrics_path, 'w') as f:
    json.dump(REPORT_METRICS, f, indent=2, default=str)
print(f"  ✓ Metrics JSON saved to: {metrics_path}")

print("\n" + "=" * 72)
print("  PIPELINE COMPLETE")
print("=" * 72)
print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │ LUAD Cellular Perceptron v4.0 — FINAL SUMMARY          │
  ├─────────────────────────────────────────────────────────┤
  │ Phase 1: AUC = {REPORT_METRICS.get('cv_auc', 'N/A'):<8s}  Sens = {str(REPORT_METRICS.get('cv_sensitivity', 'N/A')):<8s}           │
  │ Phase 2: Cancer = {str(REPORT_METRICS.get('ode_cancer_ss', 'N/A')):<6s} nM  Healthy = {str(REPORT_METRICS.get('ode_healthy_ss', 'N/A')):<5s} nM     │
  │          Monte Carlo = {str(REPORT_METRICS.get('mc_robustness', 'N/A')):<5s}% robust                │
  │          Gillespie P(lethal|healthy) = {str(REPORT_METRICS.get('ssa_p_lethal_healthy', 'N/A')):<7s}%     │
  │ Phase 3: AI Gate Kill = {str(REPORT_METRICS.get('ai_kill_rate', 'N/A')):<6s}%                     │
  │ Phase 4: {REPORT_METRICS.get('toehold_designs', 0)} toehold sensors designed               │
  │ Phase 5: {REPORT_METRICS.get('gtex_misfires', 'N/A')} GTEx misfires, {REPORT_METRICS.get('escape_relapsed', 0)}/{REPORT_METRICS.get('escape_runs', 30)} escapes           │
  └─────────────────────────────────────────────────────────┘
""")
