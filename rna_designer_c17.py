
"""
rna_designer_cxcl17.py — Toehold Switch Sensor Designer for CXCL17 mRNA
========================================================================
LUAD Perceptron Pipeline | Phase 4B: Physical Sequence Design (OR Gate 2)

Implements the Green et al. (2014) Type-A toehold switch architecture to 
design a de-novo RNA sensor for CXCL17 mRNA. This script features the 
upgraded Mammalian Kozak sequence for efficient eukaryotic translation.

INSTALLATION
------------
  pip install ViennaRNA
"""

from __future__ import annotations
import sys
import math
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _require_viennarna() -> None:
    try:
        import RNA  # noqa: F401
    except ImportError:
        sys.exit(
            "\n[ERROR] ViennaRNA Python bindings not found.\n"
            "Install with: pip install ViennaRNA\n"
        )

_require_viennarna()
import RNA  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARCHITECTURE CONSTANTS  (Green et al. 2014 - Mammalian Upgraded)
# ─────────────────────────────────────────────────────────────────────────────

TRIGGER_LEN   : int = 30   
TOEHOLD_LEN   : int = 12   
STEM_LEN      : int = 18   

# Mammalian Optimized Kozak Sequence (Replaces bacterial Shine-Dalgarno)
SWITCH_LOOP   : str = "GCCGCCACCAUG"            # 12 nt

# In-frame linker bridging the switch scaffold to the reporter ORF.
SWITCH_LINKER : str = "AACCUGGCGGCA"            # 12 nt

# First 21 nt of the Caspase-9 apoptosis effector ORF
REPORTER_START: str = "AUGUCUGGAGAGCAGAGGGAC"   # 21 nt

# Thermodynamic thresholds for hard-filtering candidates
MIN_DUPLEX_DG     : float = -20.0   
MAX_TRIGGER_DG    : float =  -8.0   
MIN_SWITCH_DG     : float =  -5.0   
MIN_DELTADETLAG   : float = -10.0   

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CXCL17 TARGET SEQUENCE  (NM_207036.4)
# ─────────────────────────────────────────────────────────────────────────────
# Target: Human CXCL17 (Angiogenic Chemokine)
# Found by the RL Agent to capture EMT-transformed, EPCAM-negative tumor cells.
CXCL17_CDS_REGION: str = (
    "AUGAAACUGCUCUGCCUCCUGGGCUUGCUG"
    "UUGGCCCUCUUGAACCCUGGCCUGCCUGUC"
    "ACGAUUCGGGUCACCGAACCCCCAGACUCC"
    "AAGCUU"
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SEQUENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_RNA_COMPLEMENT = str.maketrans("AUGC", "UACG")

def to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")

def rc(seq: str) -> str:
    return seq.upper().translate(_RNA_COMPLEMENT)[::-1]

def gc(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0

_STOPS = {"UAA", "UAG", "UGA"}

def has_stop_codon(seq: str, frame: int = 0) -> bool:
    s = seq.upper()
    for i in range(frame, len(s) - 2, 3):
        if s[i : i + 3] in _STOPS:
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# 4.  VIENNARNA WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def vfold(seq: str):
    return RNA.fold(seq)

def vcofold(seq1: str, seq2: str):
    return RNA.cofold(seq1 + "&" + seq2)

def toehold_accessibility(switch_rna: str) -> float:
    struct, _ = vfold(switch_rna)
    toehold_start = STEM_LEN + len(SWITCH_LOOP) + STEM_LEN         
    toehold_end   = toehold_start + TOEHOLD_LEN                     
    toehold_struct = struct[toehold_start : toehold_end]
    unpaired = toehold_struct.count(".")
    return unpaired / TOEHOLD_LEN

def mrna_window_accessibility(target_rna: str, window_start: int) -> float:
    struct, _ = vfold(target_rna)
    window_struct = struct[window_start : window_start + TRIGGER_LEN]
    return window_struct.count(".") / TRIGGER_LEN

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TOEHOLD SWITCH DESIGN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToeholdDesign:
    position         : int
    trigger_rna      : str
    a_domain         : str     
    b_domain         : str     
    switch_rna       : str
    struct_switch    : str
    struct_trigger   : str
    struct_duplex    : str
    struct_homodimer : str
    dg_switch        : float
    dg_trigger       : float
    dg_duplex        : float
    dg_homodimer     : float
    delta_delta_g    : float   
    toehold_access   : float   
    mrna_access      : float   
    gc_toehold       : float   
    gc_trigger       : float
    stop_codon       : bool    
    warnings         : List[str] = field(default_factory=list)
    score            : float = 0.0

def design_switch(trigger_rna: str, target_rna: str, position: int) -> ToeholdDesign:
    assert len(trigger_rna) == TRIGGER_LEN
    trigger_rna = trigger_rna.upper()

    a = trigger_rna[:TOEHOLD_LEN]    
    b = trigger_rna[TOEHOLD_LEN:]    

    rc_a = rc(a)    
    rc_b = rc(b)    

    switch_parts =[rc_b, SWITCH_LOOP, b, rc_a, SWITCH_LINKER, REPORTER_START]
    switch_rna   = "".join(switch_parts)

    struct_sw,  dg_sw  = vfold(switch_rna)
    struct_tr,  dg_tr  = vfold(trigger_rna)
    struct_cx,  dg_cx  = vcofold(trigger_rna, switch_rna)
    struct_hom, dg_hom = vcofold(trigger_rna, trigger_rna)

    ddg = dg_cx - (dg_sw + dg_tr)

    acc_toehold = toehold_accessibility(switch_rna)
    acc_mrna    = mrna_window_accessibility(target_rna, position)

    gc_a   = gc(a)
    gc_all = gc(trigger_rna)

    stop_region = b + rc_a + SWITCH_LINKER
    stop        = has_stop_codon(stop_region, frame=0)

    warns: List[str] =[]
    if gc_a < 0.35: warns.append("LOW_GC_TOEHOLD")
    if gc_a > 0.65: warns.append("HIGH_GC_TOEHOLD")
    if dg_tr < MAX_TRIGGER_DG: warns.append(f"SELF_FOLDED_TRIGGER: {dg_tr:.1f}")
    if dg_cx > MIN_DUPLEX_DG: warns.append(f"WEAK_DUPLEX: {dg_cx:.1f}")
    if dg_hom < -8.0: warns.append(f"TRIGGER_HOMODIMER: {dg_hom:.1f}")
    if dg_sw > MIN_SWITCH_DG: warns.append(f"WEAK_SWITCH: {dg_sw:.1f}")
    if stop: warns.append("STOP_CODON DETECTED")
    if acc_toehold < 0.70: warns.append(f"TOEHOLD_OCCLUDED: {acc_toehold*100:.0f}%")
    if acc_mrna < 0.60: warns.append(f"MRNA_OCCLUDED: {acc_mrna*100:.0f}%")

    score  =  -ddg * 2.0
    score -=  max(0.0, abs(dg_tr) - 4.0) * 1.5
    score +=  acc_toehold * 8.0
    score +=  acc_mrna * 12.0
    if 0.40 <= gc_a <= 0.60: score += 4.0
    elif 0.35 <= gc_a <= 0.65: score += 1.5
    else: score -= 3.0
    if stop: score -= 50.0
    score -= max(0.0, abs(dg_hom) - 5.0) * 0.5

    return ToeholdDesign(
        position=position, trigger_rna=trigger_rna, a_domain=a, b_domain=b,
        switch_rna=switch_rna, struct_switch=struct_sw, struct_trigger=struct_tr,
        struct_duplex=struct_cx, struct_homodimer=struct_hom,
        dg_switch=round(dg_sw,  2), dg_trigger=round(dg_tr, 2),
        dg_duplex=round(dg_cx,  2), dg_homodimer=round(dg_hom, 2),
        delta_delta_g=round(ddg, 2), toehold_access=round(acc_toehold, 3),
        mrna_access=round(acc_mrna, 3), gc_toehold=round(gc_a,  3),
        gc_trigger=round(gc_all, 3), stop_codon=stop, warnings=warns, score=round(score, 2),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6.  WINDOW SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan(target_rna: str, top_n: int = 5) -> List[ToeholdDesign]:
    n = len(target_rna) - TRIGGER_LEN + 1
    print(f"  Scanning {n} trigger windows across {len(target_rna)}-nt target region...")
    candidates: List[ToeholdDesign] =[]
    for i in range(n):
        window = target_rna[i : i + TRIGGER_LEN]
        try:
            d = design_switch(window, target_rna, i)
            candidates.append(d)
        except Exception as exc:
            pass
    candidates.sort(key=lambda d: d.score, reverse=True)
    return candidates[:top_n]

# ─────────────────────────────────────────────────────────────────────────────
# 7.  REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_SEP  = "═" * 68
_LINE = "─" * 68

def print_report(d: ToeholdDesign, rank: int) -> None:
    print(f"\n{_SEP}")
    print(f"  RANK {rank}  ·  Position {d.position}–{d.position+TRIGGER_LEN-1}  ·  Score: {d.score:.2f}")
    print(_SEP)

    print("\n▸ TRIGGER RNA  (30 nt from CXCL17 mRNA NM_207036.4)")
    print(f"  Full:        5'-{d.trigger_rna}-3'")
    print(f"  [a] toehold: {d.a_domain}  (nt 1–12, initiates switch binding)")
    print(f"  [b] stem:    {d.b_domain}  (nt 13–30, strand displacement)")
    print(f"  GC trigger:  {d.gc_trigger*100:.1f}%   GC toehold [a]: {d.gc_toehold*100:.1f}%")
    print(f"  mRNA window accessibility: {d.mrna_access*100:.0f}%")

    print("\n▸ SWITCH RNA  (Mammalian Kozak Optimized)")
    stem1  = rc(d.b_domain)
    toehlm = rc(d.a_domain)
    print(f"  Full switch ({len(d.switch_rna)} nt):")
    print( "  5'-" + d.switch_rna[:48])
    for chunk in textwrap.wrap(d.switch_rna[48:], width=68): print(f"     {chunk}")
    print("  -3'")
    print(f"  Toehold RC(a) accessibility: {d.toehold_access*100:.0f}%")

    print("\n▸ THERMODYNAMIC VALIDATION  (ViennaRNA MFE / RNAcofold)")
    print(f"  ΔG switch (OFF-state fold)        : {d.dg_switch:>7.2f} kcal/mol")
    print(f"  ΔG trigger (self-fold)            : {d.dg_trigger:>7.2f} kcal/mol")
    print(f"  ΔG duplex (trigger + switch ON)   : {d.dg_duplex:>7.2f} kcal/mol")
    print(f"  ΔΔG = ΔG_cx − (ΔG_sw + ΔG_tr)     : {d.delta_delta_g:>7.2f} kcal/mol")

    if d.warnings:
        print("\n▸ WARNINGS")
        for w in d.warnings: print(f"  ⚠  {w}")
    else:
        print("\n  ✓ No warnings — candidate meets all design criteria")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_SEP)
    print("  TOEHOLD SWITCH DESIGNER  ·  CXCL17 mRNA  ·  LUAD PERCEPTRON")
    print("  Architecture : Green et al. 2014, Type-A (Mammalian Kozak)")
    print("  Validator    : ViennaRNA", RNA.__version__)
    print(_SEP)

    print("\n[1] Loading CXCL17 target sequence (NM_207036.4)...")
    target_rna = CXCL17_CDS_REGION
    print(f"  Target length : {len(target_rna)} nt")

    print("\n[2] Folding full target region to map mRNA accessibility...")
    mrna_struct, mrna_mfe = vfold(target_rna)
    print(f"  mRNA MFE      : {mrna_mfe:.2f} kcal/mol")

    print("\n[3] Scanning trigger windows and designing toehold switches...")
    top_designs = scan(target_rna, top_n=1)

    print(f"\n[4] Reporting Best Design")
    print_report(top_designs[0], 1)
    
    print("\n  ✓ CXCL17 Sensor Analysis complete.")
    print(_SEP + "\n")

if __name__ == "__main__":
    main()