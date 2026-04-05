# LUAD Perceptron — Copilot Project Context

## Project Identity
Computational pipeline for in silico design of a cellular perceptron targeting
Lung Adenocarcinoma (LUAD). Five-phase pipeline, all analyses are in silico only.
Author is a bachelor's student with no wet-lab access.

## Tech Stack
Python 3.11 | numpy | pandas | scipy | scikit-learn | scanpy | anndata
ViennaRNA (RNA) | gillespy2 (stochastic) | matplotlib | seaborn

## Pipeline Phases (current state)
- Phase 1: L1 Lasso on TCGA-LUAD miRNA → hsa-miR-210 (promoter), hsa-miR-486-2 (repressor)
- Phase 2: Hill-equation ODE, steady-state P* = (α/γ)×H_A×H_R, Monte Carlo ±20%
- Phase 3: 13.4M exhaustive Boolean OR-AND search on CellxGene scRNA-seq (117,266 cells)
           Best gate: EPCAM OR CXCL17 AND NOT SRGN → 86.1% kill, 0.14% immune toxicity
- Phase 4: Toehold switch, Type-A Green 2014, Kozak GCCGCCACCAUG, validated ViennaRNA
           Trigger pos 5: 5'-ACCUGCUCUGAGCGAGUGAGAACCUACUGG-3'
           ΔG_switch=-31.60, ΔG_trigger=0.00, ΔG_duplex=-50.90, ΔΔG=-19.30 kcal/mol
- Phase 5: GTEx safety → mRNA circuit fires in 13/24 tissues (pan-epithelial EPCAM)
           Mandate: aerosol inhalation delivery only, NOT IV
- Phase 6 (IN PROGRESS): Stability Selection to address collinearity
  Status: ✓ COMPLETED - stability_selection.py
  Key result: Both mir-210 & mir-486-1 survive at 100% frequency! mir-486-2 drops to 20.8%
  Stable set size: 19 miRNAs (mean log2-RPM > 1.0, freq > 0.6)
  Run: python stability_selection.py
  Output: results/stability_selection_[timestamp].png, results/stability_selection_results_[timestamp].csv

## Active Development Priorities (from peer review)
1. ✓ Replace L1 Lasso with Stability Selection (Stabl framework) — DONE
2. Replace deterministic ODE with Gillespie SSA (gillespy2)
3. Replace Boolean gate scoring with continuous Hill transfer function
4. Add full-transcript EPCAM mRNA accessibility (NUPACK/Toehold-VISTA approach)
5. Implement Moran process evolutionary escape simulation
6. Model metabolic burden / retroactivity (ribosome pool competition)

## Constants (never change without asking)
TRIGGER_LEN=30, TOEHOLD_LEN=12, STEM_LEN=18
K_A=40.0 nM, K_R=40.0 nM, n=2.0, alpha=50.0, gamma=0.1
LETHAL_THRESHOLD=150.0 nM
EPCAM_THR=1.10, CXCL17_THR=1.10, SRGN_THR=2.10 (TPM)

## Coding Standards
- Type-annotated Python (typing module)
- Docstrings on every function with biological justification, not just code description
- Never use magic numbers inline — define named constants at top of file
- All random ops use numpy.random.default_rng(42) for reproducibility
- Save all outputs to /results/ with timestamped filenames

## Phase 6: Stability Selection Script (stability_selection.py)

### Purpose
Addresses collinearity in L1 Lasso by identifying robust biomarkers across 500 bootstrap iterations.
HIF-1α-driven miRNAs (mir-210, mir-486) often co-regulated: Lasso drops one per split.
Stability selection reveals TRUE dependencies, essential for AAV payload design (4.7 kb limit).

### Usage
python stability_selection.py

### Key Parameters (constants at top of script)
- N_BOOTSTRAP_SAMPLES = 500     # Robust sample size for stability
- SUBSAMPLE_FRACTION = 0.8      # 80% subsamples per iteration
- LOG2_RPM_THRESHOLD = 1.0      # Biological pre-filter (noise floor)
- STABILITY_THRESHOLD = 0.6     # Min frequency to be "stable"
- MIN_PLOT_FREQUENCY = 0.3      # Hide very rare features in plot
- C_REGULARIZATION = 1.0        # L1 strength (inverse regularization)

### Output Files (results/ directory)
1. stability_selection_[timestamp].png  
   - Horizontal bar chart: miRNA names (y-axis) vs. selection frequency (x-axis)
   - Red dashed line at τ=0.6 threshold
   - Reference miRNAs (Phase 1) colored RED; candidates colored BLUE
   - Only displays freq ≥ 0.3 (ignores noise)

2. stability_selection_results_[timestamp].csv
   - Columns: miRNA, Selection_Frequency, Is_Stable, Is_Reference, Mean_Weight
   - Sorted by descending frequency
   - Mean_Weight=0 for unstable miRNAs (refitted on stable set only)

### Biological Interpretation
**Promoter vs. Repressor**:
- Positive weights → Promoters (increase tumour signal → KILL)
- Negative weights → Repressors (decrease tumour signal → PROTECT)

**Stable Indicator (freq > 0.6)**:
- Suggests this miRNA is reliably selected across 500 random patient cohorts
- Implies biological relevance: not an artifact of this particular split
- Safe to include in AAV circuit (won't disappear on unseen test data)

**Circuit Design Guidance**:
- Use BOTH mir-210 AND mir-135b (both 100% stable) for robust HIF-1α sensing
- mir-486-1 is the stable repressor, NOT mir-486-2 (which dropped to 20.8%)
- Add 1-2 backup repressors from 80-90% stable set for toxicity insurance