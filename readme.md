# LUAD_Perceptron
Vibe Coded attempt to understand bio-stats and circuits
Python scripts for in silico design and validation of miRNA/gene-driven logic-gate "cellular perceptrons" for LUAD (lung adenocarcinoma).

This repository currently contains standalone analysis scripts (not a packaged module) that cover:

- TCGA miRNA preprocessing and labeling
- Sparse logistic-regression feature selection
- Reinforcement-learning / exhaustive search for OR-AND genetic logic gates
- RNA sensor sequence design with simulated annealing
- ODE and Monte Carlo validation of circuit dynamics

## Reported Results (LUAD Case Study)

### 1. Global-Optimal Logic Gate (13.4M Exhaustive Search)

After a vectorized Boolean exhaustive search across 13.4 million combinatorial logic gates (top 300 overexpressed and top 300 underexpressed genes), the pipeline identified the following 3-input OR-AND gate as the global optimum for the LUAD tumor microenvironment:

Logical blueprint:

$$
\mathrm{IF}\ (\mathrm{Promoter}_1 > 1.10)\ \mathrm{OR}\ (\mathrm{Promoter}_2 > 1.10)\ \mathrm{AND}\ (\mathrm{Repressor} < 2.10)\ \Rightarrow\ \mathrm{TRIGGER\ APOPTOSIS}
$$

- Promoter 1 (+): ENSG00000119888 (EPCAM)
- Promoter 2 (+): ENSG00000189377 (CXCL17)
- Repressor (-): ENSG00000122862 (SRGN)

### 2. Quantitative Performance Metrics

Evaluation arena: 5,000 single cells (1,551 malignant epithelial + 3,449 healthy/immune) with a strict toxicity penalty.

- Tumor cells destroyed (true positives): 1,336 / 1,551
- Immune cells accidentally destroyed (false positives): 5 / 3,449
- Maximized efficacy (kill rate / sensitivity): 86.1%
- Minimized toxicity (false positive rate): 0.14%

### 3. Biological Significance of Selected Genes

- EPCAM: canonical epithelial tumor biomarker; broad primary tumor coverage.
- CXCL17: associated with LUAD angiogenesis in mucosal tissues; backup detection pathway for cells reducing EPCAM during EMT-like transitions.
- SRGN: hematopoietic/immune-lineage-associated proteoglycan; used as a biological safety lock to reduce off-target immune depletion.

### 4. Thermodynamic and Biophysical Validation

The mathematical gate was mapped to physical sequence-level design using simulated annealing and ViennaRNA (RNAcofold).

- Example designed RNA toehold sensor: 5'-UUUUGUGAUCCGUGGUUUAU-3'
- Heuristic target energy: $\Delta G = -35.0$ kcal/mol
- Predicted heterodimer binding energy: $\Delta G = -18.48$ kcal/mol (Turner 2004, $37^\circ$C)
- Predicted structure: `..((((.(((((((((....&....)))))))))))))...`

Dot-plot screenshots used for structural inspection:

- A Monomer (`AUGGCCUACGGAUCGCUAAA`)

![A monomer dot plot](Screenshot%202026-04-02%20163424.png)

- B Monomer (`UUUUGUGAUCCGUGGUUUAU`)

![B monomer dot plot](Screenshot%202026-04-02%20163445.png)

- AB Heterodimer

![AB heterodimer dot plot](Screenshot%202026-04-02%20163504.png)

- BB Homodimer

![BB homodimer dot plot](Screenshot%202026-04-02%20163519.png)

- AA Homodimer

![AA homodimer dot plot](Screenshot%202026-04-02%20163534.png)

Ensemble free-energy summary from dot-plot analyses:

| Complex | Ensemble free energy |
| --- | --- |
| AB Heterodimer | -20.486630 |
| AA Homodimer | -8.735091 |
| BB Homodimer | -9.317922 |
| A Monomer | -1.735774 |
| B Monomer | -0.275602 |

Interpretation of free energies: because the AB heterodimer has the most negative ensemble free energy, it is thermodynamically favored over AA and BB self-dimers, supporting selective target-sensor pairing.

Interpretation: the predicted secondary structure supports intentional wobble pairing and a functional 4-nt toehold region, helping prevent constitutive ribosome blockage.

### 5. Kinetic Robustness (Monte Carlo ODE Validation)

ODE simulations (Hill model with $n = 2.0$) were run over 48 hours with parameter perturbation:

- Noise model: plus/minus 20% multidimensional variance in $\alpha$, $\gamma$, and $K$
- Total stochastic trials: 200
- Robustness score: 100%

Across all noisy simulations, cancer trajectories crossed the lethal apoptosis threshold (>150 nM), while healthy-cell trajectories remained below the threshold.

## Repository Contents

### Input Data Files

- `TCGA-LUAD.mirna.tsv`: miRNA expression matrix used by preprocessing + ML scripts.
- `TCGA-LUAD.clinical.tsv`: clinical file present in repo (currently not used by the main scripts).
- `LUAD.h5ad`: single-cell AnnData used by Scanpy/RL scripts.

### Data Preparation and Classical ML

- `find_signi_mirna.py`: loads `TCGA-LUAD.mirna.tsv`, derives `Target` label from TCGA barcode, filters low-abundance miRNAs, and prints dataset stats.
- `clinical-mirna-merge.py`: same current preprocessing flow as `find_signi_mirna.py` (name suggests merge, but script behavior is preprocessing/labeling).
- `L1_ML.py`: trains L1-regularized logistic regression, searches for a 3-5 feature sparse solution, and prints selected markers, weights, and confusion matrix metrics.

### Single-Cell Inspection and Environment Setup

- `sc_check.py`: quick inspection of key AnnData annotations (`author_cell_type_level_1`, `author_cell_type_level_2`, `disease`).
- `sc_train.py`: helper script to inspect `adata.obs` metadata and build a binary RL target using a configurable cell-type column.

### RL / Exhaustive Logic-Gate Search

- `rl_agent.py`: baseline 2-input gate search with strict false-positive penalty.
- `rl_agent_v2.py`: 3-input OR-AND gate RL search over random genes/thresholds.
- `rl_afgent_v3.py`: RL search constrained to elite differential-expression pools.
- `rl_agebt_v4.py`: exhaustive brute-force over elite pools with percentile-derived thresholds.
- `rl_agent_v5.py`: vectorized large-scale exhaustive search over promoter pairs and repressors (includes progress timing).

Note: filenames `rl_afgent_v3.py` and `rl_agebt_v4.py` appear to contain typos but are referenced here exactly as they exist.

### Biophysical and Robustness Simulation

- `rna_designer.py`: simulated annealing RNA sequence generator to match a target binding energy ($\Delta G$) against a target mRNA fragment.
- `ode_sim.py`: Hill-function ODE simulation for killer-protein dynamics in cancer vs healthy cell profiles.
- `sim_monte_validation.py`: Monte Carlo sensitivity analysis (parameter perturbation) of the ODE model with robustness summary metrics.

## Environment Setup

Use Python 3.9+.

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib scikit-learn scanpy anndata
```

## How to Run

Run scripts from the repository root so relative file paths resolve correctly.

Examples:

```bash
python find_signi_mirna.py
python L1_ML.py
python sc_check.py
python rl_agent_v5.py
python rna_designer.py
python ode_sim.py
python sim_monte_validation.py
```

## Suggested Workflow

1. Verify data and annotations with `sc_check.py` and (if needed) `sc_train.py`.
2. Run `find_signi_mirna.py` or `clinical-mirna-merge.py` to inspect TCGA labeling and filtering.
3. Run `L1_ML.py` for sparse marker discovery.
4. Explore logic-gate search variants from `rl_agent.py` through `rl_agent_v5.py`.
5. Validate biophysical plausibility with `rna_designer.py`, `ode_sim.py`, and `sim_monte_validation.py`.

## Notes

- Scripts currently print results to console and/or plots; they do not persist outputs to files by default.
- Most scripts use fixed seeds/parameters inside the code. Edit constants in each script to tune behavior.
- `TCGA-LUAD.clinical.tsv` is present but not integrated into the current primary pipeline scripts.
