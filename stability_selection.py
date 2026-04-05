"""
Stability Selection for miRNA Biomarker Discovery
===================================================

BIOLOGICAL JUSTIFICATION:
L1 regularization (Lasso) in L1_ML.py suffers from collinearity: multiple
hypoxia-driven miRNAs (miR-210, miR-486-2) are co-regulated by HIF-1α, so Lasso
arbitrarily selects one and drops the rest depending on random train/test splits.

Stability Selection (Meinshausen & Bühlmann, 2010) addresses this by:
1. Subsampling 500 times with random 80% draws
2. Running L1-regularized logistic regression on each subsample
3. Tracking how often each miRNA gets selected (non-zero coefficient)
4. Keeping only stable features (selection_frequency > threshold)

This is critical for AAV payload design: the 4.7 kb AAV packaging limit
means we MUST identify the minimal set of redundant miRNAs necessary to
achieve target selectivity. Stability selection ensures the final set is
robust across different patient subsets, reducing risk of off-target activity
in unseen test populations.

MATHEMATICS:
For each subsample k ∈ {1,...,500}:
  1. Draw random sample I_k ⊂ {1,...,n} with |I_k| = 0.8n
  2. Fit logistic regression: min_β L(β; X_I_k, y_I_k) + λ||β||_1
  3. Record selected features S_k = {j : β_j ≠ 0}
  
Stability frequency: π̂_j = (1/500) Σ_{k=1}^{500} I(j ∈ S_k)

Features with π̂_j > τ (threshold τ=0.6) are stable biomarkers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Set
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================
TCGA_MIRNA_FILE = 'TCGA-LUAD.mirna.tsv'
LOG2_RPM_THRESHOLD = 1.0  # Biological pre-filter cutoff
N_BOOTSTRAP_SAMPLES = 500  # Stability selection iterations
SUBSAMPLE_FRACTION = 0.8   # Draw 80% of samples per iteration
STABILITY_THRESHOLD = 0.6  # Min selection frequency to be "stable"
MIN_PLOT_FREQUENCY = 0.3   # Don't plot miRNAs below this threshold
RANDOM_SEED = 42
C_REGULARIZATION = 1.0    # L1 regularization strength (sklearn LogisticRegression)

# Reference findings from Phase 1 (using lowercase 'mir' to match TCGA data format)
REFERENCE_MIRNAS = {'hsa-mir-210', 'hsa-mir-486-1', 'hsa-mir-486-2'}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_label_from_barcode(barcode: str) -> np.int64:
    """
    Extract tissue type from TCGA barcode's sample type field.
    
    BIOLOGICAL CONTEXT:
    TCGA barcodes encode sample origin: sample codes 01-09 are primary tumours,
    10-19 are normal tissue. LUAD specifically uses:
    - 01A = primary tumour (adenocarcinoma cell)
    - 11A = normal adjacent tissue (healthy squamous/alveolar cell)
    This binary classification forms our ground truth for the perceptron.
    
    MATHEMATICS:
    Barcode format: TCGA-LUAD-##-####-##[sample-code]-##[vial-code]-##[portion-code]
    Extract position [3] → sample_code: 01=tumor (y=1), 11=normal (y=0)
    
    Args:
        barcode: TCGA sample barcode string
        
    Returns:
        1 if tumour sample, 0 if normal, np.nan if parsing fails
    """
    try:
        sample_code = str(barcode).split('-')[3]
        if sample_code.startswith('0'):
            return np.int64(1)  # Tumour → positive class
        elif sample_code.startswith('1'):
            return np.int64(0)  # Normal → negative class
        else:
            return np.nan
    except:
        return np.nan


def load_and_prefilter_mirnas(
    filepath: str,
    log2_rpm_threshold: float = LOG2_RPM_THRESHOLD
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load TCGA miRNA data and apply biological pre-filtering.
    
    BIOLOGICAL CONTEXT:
    miRNA abundance varies ~4 orders of magnitude; the lowest-abundance miRNAs
    are often sequencing noise or developmentally inactive. We filter to
    mean log2(RPM) > 1.0 (≈ 2 RPM baseline) to keep biologically relevant
    miRNAs with signal above noise floor. This matches standard lncRNA/miRNA
    QC pipelines in cancer genomics.
    
    MATHEMATICS:
    For each miRNA j:
      mean_expr_j = (1/n) Σ_i x_{ij}
    Retain S = {j : mean_expr_j > threshold}
    This reduces feature space from ~2000 known miRNAs to ~300-400 expressed
    in LUAD, improving numerical stability and reducing multiple testing burden.
    
    Args:
        filepath: Path to TCGA-LUAD.mirna.tsv (rows=samples, cols=miRNAs)
        log2_rpm_threshold: Minimum mean log2(RPM) to retain a miRNA
        
    Returns:
        df: DataFrame with rows=samples, cols=filtered miRNAs + 'Target' label
        y: Series of binary labels (1=tumour, 0=normal)
    """
    # Load data (TCGA format: rows=miRNAs, cols=samples)
    mirna_raw = pd.read_csv(filepath, sep='\t', index_col=0)
    df = mirna_raw.T  # Transpose: rows=samples, cols=miRNAs
    
    # Extract labels from TCGA barcodes
    df['Target'] = [get_label_from_barcode(idx) for idx in df.index]
    df = df.dropna(subset=['Target'])
    
    # Biological pre-filter
    mirna_columns = df.columns.drop('Target')
    mean_expression = df[mirna_columns].mean()
    abundant_mirnas = mean_expression[mean_expression > log2_rpm_threshold].index.tolist()
    
    df_filtered = df[abundant_mirnas + ['Target']].copy()
    y = df_filtered['Target']
    
    print(f"[PreFilter] Loaded {df.shape[0]} samples")
    print(f"[PreFilter] Retained {len(abundant_mirnas)}/{len(mirna_columns)} miRNAs "
          f"(mean log2-RPM > {log2_rpm_threshold})")
    print(f"[PreFilter] Class balance: {(y==1).sum()} tumour, {(y==0).sum()} normal\n")
    
    return df_filtered, y


def run_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = N_BOOTSTRAP_SAMPLES,
    subsample_fraction: float = SUBSAMPLE_FRACTION,
    c_regularization: float = C_REGULARIZATION,
    random_seed: int = RANDOM_SEED
) -> Dict[str, float]:
    """
    Run stability selection: repeated L1 logistic regression on random subsamples.
    
    BIOLOGICAL CONTEXT:
    Hypoxia-driven circuits are often redundant: if HIF-1α activates both
    miR-210 and miR-486 for angiogenesis suppression, Lasso will arbitrarily
    keep one and drop the other on different train/test splits. Stability
    selection identifies BOTH by asking: "across 500 random patient cohorts,
    which miRNAs are consistently selected?" The answer reveals the true
    functional redundancy crucial for robust AAV design.
    
    MATHEMATICS:
    For each iteration k ∈ {1,...,n_iterations}:
      1. Draw subsample I_k with replacement: |I_k| = ceil(subsample_fraction × n)
      2. Fit L1 logistic regression on (X_I_k, y_I_k) with balanced class weighting
         min_β L(β; X_I_k, y_I_k) + C_reg^(-1) ||β||_1
      3. Record indicator S_k = {j : |β_j| > 1e-8}
      
    Stability score: π̂_j = (1/n_iterations) Σ_k I(j ∈ S_k)
    
    The balanced class_weight='balanced' ensures equal importance for rare
    normal tissue biopsy vs. abundant TCGA tumour samples (typical imbalance 3:1).
    
    Args:
        X: Predictor matrix (rows=samples, cols=miRNAs), already pre-filtered
        y: Binary response vector (1=tumour, 0=normal)
        n_iterations: Number of bootstrap replicates
        subsample_fraction: Fraction of samples to draw each iteration
        c_regularization: Inverse regularization strength (higher C = less penalty)
        random_seed: RNG seed for reproducibility
        
    Returns:
        selection_frequency: Dict mapping miRNA name → selection frequency [0,1]
    """
    rng = np.random.default_rng(random_seed)
    n_samples = X.shape[0]
    subsample_size = int(np.ceil(subsample_fraction * n_samples))
    
    # Initialize selection counter
    selection_count = {col: 0 for col in X.columns}
    
    print(f"[StabilitySelection] Starting {n_iterations} bootstrap replicates...")
    print(f"[StabilitySelection] Subsample size: {subsample_size} / {n_samples} samples\n")
    
    for iteration in range(n_iterations):
        # Draw random subsample indices (without replacement)
        subsample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X.iloc[subsample_idx, :]
        y_sub = y.iloc[subsample_idx]
        
        # Fit L1 logistic regression with balanced class weights
        # class_weight='balanced' automatically adjusts class weights inversely
        # proportional to class frequency, mitigating imbalance in TCGA dataset
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=c_regularization,
            class_weight='balanced',
            random_state=random_seed + iteration,
            max_iter=1000,
            verbose=0
        )
        model.fit(X_sub, y_sub)
        
        # Record non-zero coefficients
        coefs = model.coef_[0]
        selected_mirnas = np.where(np.abs(coefs) > 1e-8)[0]
        
        for idx in selected_mirnas:
            mirna_name = X.columns[idx]
            selection_count[mirna_name] += 1
        
        if (iteration + 1) % 50 == 0:
            mean_selected = np.mean([selection_count[m] for m in selection_count])
            print(f"[StabilitySelection] Iteration {iteration+1:3d}/500 | "
                  f"Mean features per run: {mean_selected:.1f}")
    
    # Normalize to frequencies
    selection_frequency = {
        mirna: count / n_iterations
        for mirna, count in selection_count.items()
    }
    
    print(f"\n[StabilitySelection] Complete. Detected {sum(1 for f in selection_frequency.values() if f > 0)} "
          f"features selected >= 1 time.\n")
    
    return selection_frequency


def compute_stable_features_weights(
    X: pd.DataFrame,
    y: pd.Series,
    selection_frequency: Dict[str, float],
    stability_threshold: float = STABILITY_THRESHOLD,
    c_regularization: float = C_REGULARIZATION,
    random_seed: int = RANDOM_SEED
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit final L1 model on FULL dataset using only stable features.
    
    BIOLOGICAL CONTEXT:
    After identifying stable features via subsampling, we refit on the full
    dataset to get final mean weights. These weights represent the "average
    circuit strength" across the entire LUAD population, controlling for
    HIF-1α-mediated redundancy.
    
    MATHEMATICS:
    Let S_stable = {j : π̂_j > threshold}
    Refit: min_β L(β; X_S_stable, y) + C_reg^(-1) ||β||_1
    Return: μ_j = β_j for j ∈ S_stable
    
    Args:
        X: Full predictor matrix
        y: Full response vector
        selection_frequency: Dict from stability selection
        stability_threshold: Minimum frequency to be "stable"
        c_regularization: Regularization strength
        random_seed: RNG seed
        
    Returns:
        stable_features: Dict {miRNA → frequency}
        mean_weights: Dict {miRNA → coefficient}
    """
    # Filter to stable features
    stable_features = {
        mirna: freq
        for mirna, freq in selection_frequency.items()
        if freq > stability_threshold
    }
    
    if len(stable_features) == 0:
        print("[ComputeWeights] WARNING: No features survive stability threshold!")
        return {}, {}
    
    stable_mirna_list = list(stable_features.keys())
    X_stable = X[stable_mirna_list]
    
    # Fit final model on full data, stable features only
    model_final = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=c_regularization,
        class_weight='balanced',
        random_state=random_seed,
        max_iter=1000
    )
    model_final.fit(X_stable, y)
    
    # Extract coefficients (mean weights)
    coefs_final = model_final.coef_[0]
    mean_weights = {
        mirna: float(coef)
        for mirna, coef in zip(stable_mirna_list, coefs_final)
    }
    
    return stable_features, mean_weights


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Orchestrate full stability selection pipeline and generate report & figure."""
    
    print("=" * 75)
    print("STABILITY SELECTION FOR LUAD miRNA BIOMARKERS")
    print("=" * 75)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Load and pre-filter
    print("STEP 1: Load and pre-filter data")
    print("-" * 75)
    df_filtered, y = load_and_prefilter_mirnas(TCGA_MIRNA_FILE)
    X = df_filtered.drop(columns=['Target'])
    
    # Step 2: Run stability selection
    print("STEP 2: Run stability selection (500 iterations)")
    print("-" * 75)
    selection_frequency = run_stability_selection(
        X, y,
        n_iterations=N_BOOTSTRAP_SAMPLES,
        subsample_fraction=SUBSAMPLE_FRACTION,
        c_regularization=C_REGULARIZATION,
        random_seed=RANDOM_SEED
    )
    
    # Step 3: Compute stable features and weights
    print("STEP 3: Compute mean weights on full data (stable features only)")
    print("-" * 75)
    stable_features, mean_weights = compute_stable_features_weights(
        X, y,
        selection_frequency,
        stability_threshold=STABILITY_THRESHOLD,
        c_regularization=C_REGULARIZATION,
        random_seed=RANDOM_SEED
    )
    
    # Step 4: Results summary
    print("STEP 4: Stability Selection Results")
    print("-" * 75)
    
    print(f"\nSTABLE FEATURES (frequency > {STABILITY_THRESHOLD}):")
    print(f"Total stable miRNAs: {len(stable_features)}\n")
    
    for mirna in sorted(stable_features.keys(), 
                        key=lambda m: stable_features[m], reverse=True):
        freq = stable_features[mirna]
        weight = mean_weights[mirna]
        part_type = "PROMOTER (+)" if weight > 0 else "REPRESSOR (-)"
        in_reference = " ✓ REFERENCE" if mirna in REFERENCE_MIRNAS else ""
        print(f"  {mirna:20s}  Freq={freq:.3f}  Weight={weight:+.4f}  {part_type}{in_reference}")
    
    print(f"\nREFERENCE miRNAs from Phase 1:")
    for mirna in REFERENCE_MIRNAS:
        if mirna in selection_frequency:
            freq = selection_frequency[mirna]
            status = "✓ STABLE" if mirna in stable_features else "✗ UNSTABLE"
            print(f"  {mirna:20s}  Freq={freq:.3f}  [{status}]")
        else:
            print(f"  {mirna:20s}  NOT SELECTED IN ANY ITERATION")
    
    # Step 5: Generate visualization
    print("\nSTEP 5: Generating visualization...")
    print("-" * 75)
    
    # Prepare data for plotting (exclude very rare features)
    plot_data = {
        k: v for k, v in selection_frequency.items()
        if v >= MIN_PLOT_FREQUENCY
    }
    
    if not plot_data:
        print("WARNING: No features pass plotting threshold!")
        return
    
    # Sort by frequency
    mirnas_sorted = sorted(plot_data.keys(), key=lambda m: plot_data[m])
    frequencies_sorted = [plot_data[m] for m in mirnas_sorted]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color code: reference vs non-reference
    colors = [
        '#e74c3c' if mirna in REFERENCE_MIRNAS else '#3498db'
        for mirna in mirnas_sorted
    ]
    
    # Horizontal bar chart
    bars = ax.barh(mirnas_sorted, frequencies_sorted, color=colors, alpha=0.8, edgecolor='black', linewidth=1.0)
    
    # Vertical line at stability threshold
    ax.axvline(x=STABILITY_THRESHOLD, color='red', linestyle='--', linewidth=2.5, 
               label=f'Stability Threshold (τ={STABILITY_THRESHOLD})', zorder=10)
    
    # Styling
    ax.set_xlabel('Selection Frequency (out of 500 iterations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('miRNA', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    # Title and legend
    ax.set_title(
        'Stability Selection: miRNA Biomarker Robustness\n'
        'L1 Logistic Regression on 500 Random 80% Subsamples',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Reference (Phase 1)'),
        Patch(facecolor='#3498db', alpha=0.8, edgecolor='black', label='Candidate'),
        plt.Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label=f'Stability τ={STABILITY_THRESHOLD}')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right', framealpha=0.95)
    
    # Add text annotations
    textbox = f"""Cohort: {X.shape[0]} patients, {X.shape[1]} miRNAs
Filter: log2(RPM) > {LOG2_RPM_THRESHOLD}
Stable set (tau>{STABILITY_THRESHOLD}): {len(stable_features)} miRNAs
Plot threshold: freq >= {MIN_PLOT_FREQUENCY}
    """
    ax.text(0.02, 0.98, textbox, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figpath = f'results/stability_selection_{timestamp}.png'
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figpath}\n")
    
    # Step 6: Save numerical results
    print("STEP 6: Saving results to CSV...")
    print("-" * 75)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'miRNA': list(selection_frequency.keys()),
        'Selection_Frequency': list(selection_frequency.values()),
        'Is_Stable': [f > STABILITY_THRESHOLD for f in selection_frequency.values()],
        'Is_Reference': [m in REFERENCE_MIRNAS for m in selection_frequency.keys()],
        'Mean_Weight': [mean_weights.get(m, 0.0) for m in selection_frequency.keys()]
    }).sort_values('Selection_Frequency', ascending=False)
    
    csvpath = f'results/stability_selection_results_{timestamp}.csv'
    results_df.to_csv(csvpath, index=False)
    print(f"Results saved to: {csvpath}\n")
    
    print("=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == '__main__':
    main()
