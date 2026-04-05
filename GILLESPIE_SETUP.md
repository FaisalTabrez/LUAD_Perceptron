# Phase 7: GillesPy2 Stochastic Simulation — Setup & Execution

## Installation

### 1. Install GillesPy2
```bash
pip install gillespy2
```

**Note on GillesPy2**: The package is called `gillespy2` on PyPI, imported as `import gillespy2 as gillespy` or `from gillespy2 import Model, Parameter, Species, Reaction`.

If you encounter issues:
- **Latest version (1.0+)**: `pip install --upgrade gillespy2`
- **Requires**: numpy, scipy (for ODE fallback), pandas

### 2. Install Supporting Packages (if not already present)
```bash
pip install numpy scipy pandas matplotlib seaborn
```

## Execution

### Quick Run
```bash
python gillespie_sim.py
```

Expected runtime: **2–5 minutes** (10,000 trajectories × 200 timepoints each)

### Output Files
The script saves to `results/` directory:
- `gillespie_sim_[timestamp].png` — 3-panel publication figure
- `gillespie_sim_results_[timestamp].csv` — Safety metrics table

## What the Script Does (Biological Workflow)

### Step 1: Initialize GillesPy2 Models
- **Cancer cell**:  miR-210=600 mol (~800 nM), miR-486=30 mol (~50 nM)
- **Healthy cell**: miR-210=60 mol (~100 nM),  miR-486=450 mol (~750 nM)
  (Note: exact conversion uses Avogadro's number + volume = 1e-15 L)

### Step 2: Define Reactions
1. **Production** (zero-order in reactants, Hill-regulated):
   ```
   ∅ → KillerProtein
   propensity = alpha × H_A(miR210) × H_R(miR486)
   ```
   - H_A (activator Hill) = miR210^n / (K_A^n + miR210^n)
   - H_R (repressor Hill) = K_R^n / (K_R^n + miR486^n)

2. **Degradation** (first-order):
   ```
   KillerProtein → ∅
   propensity = gamma × [KillerProtein]
   ```

### Step 3: Run τ-Leaping Solver
- 10,000 independent stochastic trajectories per cell type
- 48 hours simulated time
- 200 timepoints for smooth plotting
- Outputs: molecule counts at each timepoint

### Step 4: Compute Safety Metrics
**False-Positive Rate** (most critical):
- For each healthy trajectory: find max(KillerProtein) over entire 48h
- Convert to nM using molecule_count_to_nM()
- Count how many exceed LETHAL_THRESHOLD = 150 nM
- FP_rate = (count exceeding) / (10,000 total)

**Clinical Acceptance Criteria**:
- FP_rate < 0.1% → EXCELLENT (margin for real-world noise)
- FP_rate < 1.0% → GOOD (standard for diagnostics)
- FP_rate > 5% → UNSAFE (redesign required)

### Step 5: Generate Plots
- **Panel A**: Stochastic trajectories (shows variability)
- **Panel B**: Histogram of max(KillerProtein) (shows separation)
- **Panel C**: Sensitivity curve (robustness to threshold choice)

## Expected Results (Ballpark)

If the circuit is well-tuned (K_A = K_R = 40 nM, alpha = 50, gamma = 0.1):

| Cell Type | Mean max(KP) [nM] | Std [nM] | Notes |
|-----------|-------------------|----------|-------|
| Cancer    | 180–250           | 20–40    | Should reliably exceed 150 nM |
| Healthy   | 20–50             | 10–20    | Should rarely exceed 150 nM |
| **FP Rate** | **< 1%** | - | Safe for clinical use |

## Troubleshooting

### Q: "ModuleNotFoundError: No module named 'gillespie2'"
**A**: Install with: `pip install gillespy2 --upgrade`

### Q: "ImportError: cannot import name 'Model' from 'gillespy2'"
**A**: Check GillesPy2 version. Update to latest: `pip install gillespy2 --upgrade`

### Q: Solver crashes or hangs
**A**: Try switching solver (edit line ~400 in gillespie_sim.py):
```python
# Change from:
result = gillespy.SSACSolver(model).run(...)

# Try:
result = gillespy.TauLeapingSolver(model).run(...)  # Faster, approximate
result = gillespy.ODESolver(model).run(...)         # Deterministic fallback
```

### Q: "results directory not found"
**A**: Create it manually:
```bash
mkdir results
```
or adjust output path in gillespie_sim.py (line ~490)

## Paper Citation Notes

This simulation implements the stochastic extension of Phase 2's Hill-equation ODE model.
If publishing, cite:
- **Gillespie algorithm**: Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. J. Phys. Chem.
- **GillesPy2 framework**: Hellander et al. (2019). Flexible and efficient stochastic simulation of reaction networks using GillesPy2.
- **Hill kinetics**: Alon, U. (2007). An Introduction to Networks in Molecular Biology. MIT Press. (Chapters 2–3 on logic gates)

## Next Steps (Phase 8+)

After validating false-positive rate:
1. **Phase 8**: Metabolic burden model (ribosome pool depletion)
2. **Phase 9**: Evolutionary escape dynamics (Moran process)
3. **Phase 10**: Full-transcript EPCAM accessibility (NUPACK/Toehold-VISTA)

---
*Author: Bachelor's student, in silico only*
*Date: April 2026*
*Status: Phase 7 development, peer-review cycle*
