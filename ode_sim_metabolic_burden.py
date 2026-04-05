"""
================================================================================
PHASE 2+ EXTENSION: METABOLIC BURDEN MODELING
================================================================================

Extended ODE simulation incorporating synthetic biology metabolic burden.

Problem: Introducing EPCAM sensor + CXCL17 sensor + SRGN repressor + Caspase-9 
actuator forces the host cell to share ribosomes between synthetic and endogenous 
genes. Ribosomal resource contention can crash the perceptron's thresholds.

Solution: Model ribosomal availability (R) as a dynamic state variable that 
represents the fraction of ribosomes unoccupied by synthetic transcripts.

Extended ODE System:
  dP/dt = alpha * R * H_A * H_R - gamma * P  (burden-dependent production)
  dR/dt = mu * (1 - R) - burden_per_transcript * mRNA_total  (ribosomal recovery)

Scenarios:
  1. Low burden: 1 synthetic transcript (just EPCAM sensor)
  2. Medium burden: 3 transcripts (EPCAM + CXCL17 + Caspase-9 ORF)
  3. High burden: 4 transcripts (full circuit including SRGN repressor)

Expected Finding: Circuit efficacy degrades with increasing metabolic burden.
At some threshold, cancer cells fail to reach lethal protein concentration.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime
from typing import Dict, Tuple, List


# ==========================================
# 1. BIOPHYSICAL PARAMETERS
# ==========================================

# Hill function thresholds (from Phase 1 biomarker validation)
alpha_max = 50.0      # Maximum production rate of Killer Protein (nM per hour)
gamma = 0.1           # Degradation rate of the protein (hr^-1)
n = 2.0               # Hill coefficient (Cooperativity)
K_A = 40.0            # Dissociation constant for miR-210 sensor activator
K_R = 40.0            # Dissociation constant for miR-486 sensor repressor

lethal_threshold = 150.0  # Protein concentration required for apoptosis (nM)

# Ribosomal dynamics parameters
mu = 0.5              # Ribosome recovery rate (hr^-1) [literature: 0.1-1.0]
burden_per_transcript = 0.15  # Ribosomal occupancy per transcript [normalized units]
                               # Each mRNA molecule ties up ribosomes during translation.
                               # Typical E. coli: 8-12 ribosomes per mRNA.
                               # Normalized: assume ~0.15 units per transcript in mammalian cell


# ==========================================
# 2. SCENARIO DEFINITIONS (TRANSCRIPT BURDENS)
# ==========================================

SCENARIOS = {
    'low_burden': {
        'description': '1 synthetic transcript (EPCAM sensor only)',
        'n_transcripts': 1,
        'transcripts': ['EPCAM sensor'],
        'color': 'green',
        'linestyle': '-'
    },
    'medium_burden': {
        'description': '3 transcripts (EPCAM + CXCL17 sensors + Caspase-9)',
        'n_transcripts': 3,
        'transcripts': ['EPCAM sensor', 'CXCL17 sensor', 'Caspase-9 ORF'],
        'color': 'orange',
        'linestyle': '--'
    },
    'high_burden': {
        'description': '4 transcripts (full circuit: EPCAM + CXCL17 + SRGN repressor + Caspase-9)',
        'n_transcripts': 4,
        'transcripts': ['EPCAM sensor', 'CXCL17 sensor', 'SRGN repressor', 'Caspase-9 ORF'],
        'color': 'red',
        'linestyle': ':'
    }
}


# ==========================================
# 3. ODE SYSTEM: PROTEIN + RIBOSOMAL DYNAMICS
# ==========================================

def perceptron_circuit_with_burden(state, t, miR_210, miR_486, n_transcripts):
    """
    Extended ODE system incorporating metabolic burden.
    
    State vector:
      state[0] = P (Killer Protein concentration, nM)
      state[1] = R (Available ribosomes, normalized 0-1)
    
    Parameters:
      miR_210: Activating miRNA concentration (cancer: high, healthy: low)
      miR_486: Repressor miRNA concentration (cancer: low, healthy: high)
      n_transcripts: Number of synthetic transcripts competing for ribosomes
    
    Returns:
      [dP/dt, dR/dt]
    """
    P, R = state
    
    # Activating Hill function (miR-210 promotes killing)
    H_A = (miR_210**n) / (K_A**n + miR_210**n)
    
    # Repressing Hill function (miR-486 prevents killing)
    H_R = (K_R**n) / (K_R**n + miR_486**n)
    
    # Synthetic protein production scales with ribosomal availability
    dP_dt = (alpha_max * R * H_A * H_R) - (gamma * P)
    
    # Ribosomal dynamics:
    # Recovery term: ribosomes freed at rate mu
    # Burden term: ribosomes tied up by n_transcripts (assume each makes ~1 mRNA/cell)
    mRNA_total = n_transcripts  # Steady-state mRNA count proportional to transcripts
    dR_dt = mu * (1.0 - R) - burden_per_transcript * mRNA_total * R
    
    return [dP_dt, dR_dt]


# ==========================================
# 4. VIRTUAL CELL PROFILES
# ==========================================

# Expression levels from TCGA LUAD (Phase 1 analysis)
cancer_cell = {'miR_210': 120.0, 'miR_486': 5.0}
healthy_cell = {'miR_210': 10.0, 'miR_486': 150.0}

# Time simulation (48 hours)
time_hours = np.linspace(0, 48, 1000)
initial_state = [0.0, 1.0]  # Initial: [P=0 nM, R=1.0 (fully available ribos)]


# ==========================================
# 5. RUN SIMULATIONS FOR ALL SCENARIOS
# ==========================================

def run_scenario_simulations():
    """
    Run ODE simulations for all burden scenarios (cancer + healthy cells).
    
    Returns:
      Dict mapping scenario names to simulation results
    """
    results = {}
    
    for scenario_name, scenario_config in SCENARIOS.items():
        n_transcripts = scenario_config['n_transcripts']
        
        # Simulate cancer cell with this burden
        cancer_trajectory = odeint(
            perceptron_circuit_with_burden,
            initial_state,
            time_hours,
            args=(cancer_cell['miR_210'], cancer_cell['miR_486'], n_transcripts)
        )
        
        # Simulate healthy cell with this burden
        healthy_trajectory = odeint(
            perceptron_circuit_with_burden,
            initial_state,
            time_hours,
            args=(healthy_cell['miR_210'], healthy_cell['miR_486'], n_transcripts)
        )
        
        results[scenario_name] = {
            'config': scenario_config,
            'cancer': cancer_trajectory,
            'healthy': healthy_trajectory,
            'time': time_hours
        }
    
    return results


def analyze_circuit_performance(results: Dict) -> Dict:
    """
    Analyze circuit performance metrics for each scenario.
    
    Returns:
      Dict with performance metrics (peak protein, crosses threshold, etc.)
    """
    metrics = {}
    
    for scenario_name, data in results.items():
        cancer_P = data['cancer'][:, 0]  # Protein concentration trajectory
        cancer_R = data['cancer'][:, 1]  # Ribosomal availability
        healthy_P = data['healthy'][:, 0]
        healthy_R = data['healthy'][:, 1]
        
        # Peak protein concentrations
        cancer_peak = np.max(cancer_P)
        healthy_peak = np.max(healthy_P)
        
        # Does circuit kill cancer? (cancer protein > threshold)
        cancer_kills = cancer_peak > lethal_threshold
        cancer_time_to_kill = None
        if cancer_kills:
            crossing_idx = np.where(cancer_P > lethal_threshold)[0]
            if len(crossing_idx) > 0:
                cancer_time_to_kill = time_hours[crossing_idx[0]]
        
        # Does circuit harm healthy? (healthy protein > threshold)
        healthy_toxicity = healthy_peak > lethal_threshold
        
        # Ribosomal stress (minimum R during simulation)
        cancer_r_min = np.min(cancer_R)
        healthy_r_min = np.min(healthy_R)
        
        # Efficacy (cancer kill rate) vs Specificity (healthy protection)
        # Assume Hill-based damage: damage proportional to protein above threshold
        cancer_damage = np.mean(np.maximum(cancer_P - lethal_threshold, 0)) / lethal_threshold
        healthy_damage = np.mean(np.maximum(healthy_P - lethal_threshold, 0)) / lethal_threshold
        
        metrics[scenario_name] = {
            'cancer_peak_protein': cancer_peak,
            'healthy_peak_protein': healthy_peak,
            'cancer_kills': cancer_kills,
            'cancer_time_to_kill': cancer_time_to_kill,
            'healthy_toxicity': healthy_toxicity,
            'cancer_min_ribosomes': cancer_r_min,
            'healthy_min_ribosomes': healthy_r_min,
            'cancer_damage': cancer_damage,
            'healthy_damage': healthy_damage,
            'efficacy_rank': 0  # Will be filled in ranking step
        }
    
    return metrics


# ==========================================
# 6. CIRCUIT FOOTPRINT CALCULATION
# ==========================================

def calculate_circuit_footprint() -> Dict:
    """
    Calculate total base-pair footprint of the synthetic circuit.
    
    Components:
      - EPCAM toehold switch: miRNA recognition + riboswitch structure
      - CXCL17 toehold switch: miRNA recognition + riboswitch structure
      - SRGN repressor construct: includes promoter + ORF
      - Caspase-9 ORF: killing actuator protein
      - Promoters + terminators: regulatory elements
      - Polycistronic linkers: ribosomal binding sites (RBS)
    
    Returns:
      Dict with component footprints and total
    """
    footprint = {
        'EPCAM_toehold_switch': 93,      # nt (literature: 93-100 nt typical)
        'CXCL17_toehold_switch': 93,     # nt (estimate, similar to EPCAM)
        'SRGN_repressor_construct': 500, # nt (estimate: promoter ~50 + ORF ~450)
        'Caspase9_ORF': 1281,            # nt (human CASP9 CDS)
        'promoters_terminators': 600,    # nt (3 promoters @ ~150 ea + 2 SV40 polyA @ ~75)
        'RBS_linkers_spacers': 150,      # nt (polycistronic RBS ~10 nt × 15 insertions)
    }
    
    total_nt = sum(footprint.values())
    total_kb = total_nt / 1000.0
    
    footprint['TOTAL_NUCLEOTIDES'] = total_nt
    footprint['TOTAL_KILOBASES'] = total_kb
    footprint['AAV_CAPACITY_KB'] = 4.7
    footprint['AAV_HEADROOM_KB'] = 4.7 - total_kb
    footprint['AAV_COMPLIANT'] = total_kb <= 4.7
    
    return footprint


# ==========================================
# 7. VISUALIZATION
# ==========================================

def plot_burden_comparison(results: Dict, metrics: Dict):
    """
    Create publication-quality figure comparing all burden scenarios.
    
    Plots both cancer and healthy cell trajectories for each scenario,
    with lethal threshold highlighted.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        'Circuit Performance Under Metabolic Burden: Ribosomal Limitations Reduce Efficacy',
        fontsize=14,
        fontweight='bold'
    )
    
    scenario_list = list(SCENARIOS.keys())
    
    # Protein concentration trajectories (top row)
    # Ribosomal availability trajectories (bottom row)
    for idx, scenario_name in enumerate(scenario_list):
        data = results[scenario_name]
        metric = metrics[scenario_name]
        
        # Top row: Protein concentration
        ax_protein = axes[0, idx]
        cancer_P = data['cancer'][:, 0]
        healthy_P = data['healthy'][:, 0]
        
        ax_protein.plot(data['time'], cancer_P, 
                       color='crimson', linewidth=2.5, label='Cancer cell')
        ax_protein.plot(data['time'], healthy_P, 
                       color='mediumseagreen', linewidth=2.5, label='Healthy cell')
        ax_protein.axhline(y=lethal_threshold, color='black', 
                          linestyle='--', linewidth=2, label='Lethal threshold')
        
        # Fill threshold region
        ax_protein.fill_between(data['time'], lethal_threshold, 400, 
                               alpha=0.15, color='red', label='Kill zone')
        
        # Annotations
        kills_success = "✓ KILLS" if metric['cancer_kills'] else "✗ FAILS"
        toxicity_status = "⚠ TOXIC" if metric['healthy_toxicity'] else "✓ SAFE"
        
        title_text = f"{data['config']['description']}\n{kills_success} | {toxicity_status}"
        ax_protein.set_title(title_text, fontsize=11, fontweight='bold')
        ax_protein.set_xlabel('Time (hours)', fontsize=10)
        ax_protein.set_ylabel('Protein Concentration (nM)', fontsize=10)
        ax_protein.set_ylim([0, 400])
        ax_protein.grid(alpha=0.3)
        if idx == 0:
            ax_protein.legend(loc='upper left', fontsize=9)
        
        # Bottom row: Ribosomal availability
        ax_ribos = axes[1, idx]
        cancer_R = data['cancer'][:, 1]
        healthy_R = data['healthy'][:, 1]
        
        ax_ribos.plot(data['time'], cancer_R, 
                     color='crimson', linewidth=2.5, label='Cancer cell')
        ax_ribos.plot(data['time'], healthy_R, 
                     color='mediumseagreen', linewidth=2.5, label='Healthy cell')
        
        # Strain indicator: R < 0.5 = severe ribosomal stress
        ax_ribos.fill_between(data['time'], 0, 0.5, alpha=0.15, 
                             color='orange', label='Ribosomal stress')
        
        ax_ribos.set_title(f'Ribosomal Occupancy | Burden: {data["config"]["n_transcripts"]} transcripts',
                          fontsize=11, fontweight='bold')
        ax_ribos.set_xlabel('Time (hours)', fontsize=10)
        ax_ribos.set_ylabel('Available Ribosomes (normalized)', fontsize=10)
        ax_ribos.set_ylim([0, 1.1])
        ax_ribos.grid(alpha=0.3)
        if idx == 0:
            ax_ribos.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_burden_efficacy_tradeoff(metrics: Dict):
    """
    Create efficacy vs. burden comparison figure.
    
    Shows how circuit performance degrades with increased metabolic load.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    scenario_list = list(SCENARIOS.keys())
    n_transcripts = [SCENARIOS[s]['n_transcripts'] for s in scenario_list]
    
    cancer_peaks = [metrics[s]['cancer_peak_protein'] for s in scenario_list]
    healthy_peaks = [metrics[s]['healthy_peak_protein'] for s in scenario_list]
    cancer_min_r = [metrics[s]['cancer_min_ribosomes'] for s in scenario_list]
    
    # Panel 1: Peak protein vs. burden
    ax1 = axes[0]
    ax1.plot(n_transcripts, cancer_peaks, 'o-', color='crimson', 
            linewidth=2.5, markersize=10, label='Cancer peak protein')
    ax1.plot(n_transcripts, healthy_peaks, 's-', color='mediumseagreen', 
            linewidth=2.5, markersize=10, label='Healthy peak protein')
    ax1.axhline(y=lethal_threshold, color='black', linestyle='--', 
               linewidth=2, label='Lethal threshold')
    ax1.set_xlabel('Number of Synthetic Transcripts', fontsize=11)
    ax1.set_ylabel('Peak Protein Concentration (nM)', fontsize=11)
    ax1.set_xticks(n_transcripts)
    ax1.set_ylim([0, 350])
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_title('Circuit Efficacy Degradation', fontsize=12, fontweight='bold')
    
    # Panel 2: Minimum ribosomal availability vs. burden
    ax2 = axes[1]
    ax2.plot(n_transcripts, cancer_min_r, 'o-', color='purple', 
            linewidth=2.5, markersize=10, label='Cancer cell')
    ax2.fill_between(n_transcripts, 0, 0.5, alpha=0.2, color='orange')
    ax2.text(2, 0.25, 'RIBOSOMAL STRESS ZONE', fontsize=9, 
            ha='center', style='italic', color='orange', fontweight='bold')
    ax2.set_xlabel('Number of Synthetic Transcripts', fontsize=11)
    ax2.set_ylabel('Minimum Available Ribosomes', fontsize=11)
    ax2.set_xticks(n_transcripts)
    ax2.set_ylim([0, 1.1])
    ax2.grid(alpha=0.3)
    ax2.set_title('Ribosomal Burden', fontsize=12, fontweight='bold')
    
    # Panel 3: Efficacy score
    efficacy_scores = [metrics[s]['cancer_damage'] for s in scenario_list]
    toxicity_scores = [metrics[s]['healthy_damage'] for s in scenario_list]
    
    ax3 = axes[2]
    x_pos = np.arange(len(scenario_list))
    width = 0.35
    
    ax3.bar(x_pos - width/2, efficacy_scores, width, label='Cancer damage', 
           color='crimson', alpha=0.8)
    ax3.bar(x_pos + width/2, toxicity_scores, width, label='Healthy damage', 
           color='mediumseagreen', alpha=0.8)
    
    ax3.set_xlabel('Burden Scenario', fontsize=11)
    ax3.set_ylabel('Normalized Damage Score', fontsize=11)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Low\n(1 TX)', 'Medium\n(3 TX)', 'High\n(4 TX)'], fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_title('Efficacy vs. Toxicity Trade-off', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ==========================================
# 8. MAIN EXECUTION & REPORTING
# ==========================================

def main():
    """
    Execute full metabolic burden analysis pipeline.
    """
    print("\n" + "="*80)
    print("PHASE 2+ EXTENSION: METABOLIC BURDEN ANALYSIS")
    print("="*80)
    print("\n📋 Problem: Ribosomal resource contention with synthetic circuit")
    print("   Introducing EPCAM + CXCL17 + SRGN + Caspase-9 consumes ribosomes")
    print("   → Reduces endogenous protein synthesis capacity")
    print("   → May crash perceptron thresholds\n")
    
    # Run simulations
    print("🔬 Running ODE simulations for all burden scenarios...")
    results = run_scenario_simulations()
    print("   ✓ Simulations complete (48-hour trajectories)")
    
    # Analyze performance
    print("\n📊 Analyzing circuit performance metrics...")
    metrics = analyze_circuit_performance(results)
    print("   ✓ Metrics computed for all scenarios\n")
    
    # Print results table
    print("-" * 80)
    print("RESULTS SUMMARY")
    print("-" * 80)
    print(f"{'Scenario':<20} {'Transcripts':<12} {'Cancer Peak':<15} {'Kills?':<10} {'Time-to-Kill':<15}")
    print("-" * 80)
    
    for scenario_name in SCENARIOS.keys():
        metric = metrics[scenario_name]
        n_tx = SCENARIOS[scenario_name]['n_transcripts']
        peaks = metric['cancer_peak_protein']
        kills = "✓ YES" if metric['cancer_kills'] else "✗ NO"
        time2kill = f"{metric['cancer_time_to_kill']:.2f} hr" if metric['cancer_time_to_kill'] else "N/A"
        
        print(f"{scenario_name:<20} {n_tx:<12} {peaks:>8.1f} nM {kills:<10} {time2kill:<15}")
    
    print("\n" + "-" * 80)
    print("RIBOSOMAL STRESS ANALYSIS")
    print("-" * 80)
    print(f"{'Scenario':<20} {'Cancer Min R':<15} {'Healthy Min R':<15} {'Stress Level':<15}")
    print("-" * 80)
    
    for scenario_name in SCENARIOS.keys():
        metric = metrics[scenario_name]
        cancer_r = metric['cancer_min_ribosomes']
        healthy_r = metric['healthy_min_ribosomes']
        stress = "🔴 SEVERE" if cancer_r < 0.3 else "🟡 MODERATE" if cancer_r < 0.6 else "🟢 MILD"
        
        print(f"{scenario_name:<20} {cancer_r:>6.3f} (R) {healthy_r:>7.3f} (R) {stress:<15}")
    
    print("\n" + "-" * 80)
    print("CIRCUIT FOOTPRINT ANALYSIS")
    print("-" * 80)
    
    footprint = calculate_circuit_footprint()
    
    print("\nComponent breakdown:")
    print(f"  EPCAM toehold switch:        {footprint['EPCAM_toehold_switch']:>5} nt")
    print(f"  CXCL17 toehold switch:       {footprint['CXCL17_toehold_switch']:>5} nt")
    print(f"  SRGN repressor construct:    {footprint['SRGN_repressor_construct']:>5} nt")
    print(f"  Caspase-9 ORF:               {footprint['Caspase9_ORF']:>5} nt")
    print(f"  Promoters + terminators:     {footprint['promoters_terminators']:>5} nt")
    print(f"  RBS + linkers + spacers:     {footprint['RBS_linkers_spacers']:>5} nt")
    print(f"  {'-'*45}")
    print(f"  TOTAL:                       {footprint['TOTAL_NUCLEOTIDES']:>5} nt ({footprint['TOTAL_KILOBASES']:.2f} kb)")
    
    print(f"\nAAV Compatibility:")
    print(f"  AAV packaging capacity:      {footprint['AAV_CAPACITY_KB']:.2f} kb")
    print(f"  Circuit footprint:           {footprint['TOTAL_KILOBASES']:.2f} kb")
    print(f"  Headroom:                    {footprint['AAV_HEADROOM_KB']:.2f} kb")
    
    if footprint['AAV_COMPLIANT']:
        print(f"  ✓ AAV COMPLIANT (fits within 4.7 kb limit)")
    else:
        overage = footprint['TOTAL_KILOBASES'] - footprint['AAV_CAPACITY_KB']
        print(f"  ✗ EXCEEDS AAV LIMIT by {overage:.2f} kb")
        print(f"     → Requires split AAV or alternative delivery")
    
    print("\n" + "-" * 80)
    print("KEY FINDINGS")
    print("-" * 80)
    
    # Find maximum tolerable burden
    max_burden = 0
    for scenario_name in SCENARIOS.keys():
        if metrics[scenario_name]['cancer_kills']:
            n_tx = SCENARIOS[scenario_name]['n_transcripts']
            max_burden = max(max_burden, n_tx)
    
    print(f"\n✨ Maximum tolerable synthetic transcript load: {max_burden} genes")
    print(f"   (before circuit fails to reach {lethal_threshold} nM threshold)")
    
    if max_burden == 1:
        print("   ⚠️  WARNING: Only single-sensor circuits viable with this burden model")
        print("   → Multi-sensor redundancy not achievable without metabolic engineering")
    elif max_burden == 3:
        print("   ✓ Dual-sensor system (EPCAM + CXCL17) plus Caspase-9 is viable")
        print("   → SRGN repressor addition (4th transcript) crosses threshold")
    elif max_burden >= 4:
        print("   ✓ Full circuit (4 transcripts) remains viable")
        print("   → May require ribosome engineering or alternative expression systems")
    
    # Plot and save
    print("\n📈 Generating visualizations...")
    
    fig1 = plot_burden_comparison(results, metrics)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig1_path = f'results/metabolic_burden_trajectories_{timestamp}.png'
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fig1_path}")
    
    fig2 = plot_burden_efficacy_tradeoff(metrics)
    fig2_path = f'results/metabolic_burden_efficacy_tradeoff_{timestamp}.png'
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fig2_path}")
    
    # Save metrics to CSV
    import csv
    csv_path = f'results/metabolic_burden_metrics_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Scenario', 'Transcripts', 'Cancer_Peak_nM', 'Healthy_Peak_nM',
            'Circuit_Kills', 'Time_to_Kill_hr', 'Cancer_Min_Ribosomes',
            'Healthy_Min_Ribosomes', 'Cancer_Damage_Score', 'Healthy_Damage_Score'
        ])
        
        for scenario_name in SCENARIOS.keys():
            m = metrics[scenario_name]
            writer.writerow([
                scenario_name,
                SCENARIOS[scenario_name]['n_transcripts'],
                f"{m['cancer_peak_protein']:.2f}",
                f"{m['healthy_peak_protein']:.2f}",
                "Yes" if m['cancer_kills'] else "No",
                f"{m['cancer_time_to_kill']:.2f}" if m['cancer_time_to_kill'] else "N/A",
                f"{m['cancer_min_ribosomes']:.3f}",
                f"{m['healthy_min_ribosomes']:.3f}",
                f"{m['cancer_damage']:.3f}",
                f"{m['healthy_damage']:.3f}"
            ])
    print(f"   ✓ Saved: {csv_path}")
    
    # Save footprint to CSV
    footprint_csv = f'results/circuit_footprint_{timestamp}.csv'
    with open(footprint_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Component', 'Nucleotides'])
        for key, value in footprint.items():
            if not isinstance(value, bool):
                writer.writerow([key, value])
    print(f"   ✓ Saved: {footprint_csv}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results, metrics, footprint


if __name__ == '__main__':
    results, metrics, footprint = main()
    plt.show()
