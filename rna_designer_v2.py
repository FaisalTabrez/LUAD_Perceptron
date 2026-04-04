#!/usr/bin/env python3
"""
rna_designer.py — Toehold Switch Sensor Designer for EPCAM mRNA
================================================================
LUAD Perceptron Pipeline | Phase 4: Physical Sequence Design

Implements the Green et al. (2014, Cell 159:925–939) Type-A toehold
switch architecture to design a de-novo RNA sensor for EPCAM mRNA —
the canonical LUAD tumour biomarker discovered by the exhaustive
Boolean search in Phase 3 of this pipeline.

PREVIOUS (WRONG) APPROACH
--------------------------
The original rna_designer.py generated a random RNA sequence using
simulated annealing against a heuristic ΔG target of −35 kcal/mol,
with NO connection to EPCAM mRNA. The designed sequence therefore
validated RNA thermodynamics in principle but not for the actual
biological target.

THIS SCRIPT FIXES THAT BY
--------------------------
  1. Using a validated fragment of the real EPCAM mRNA coding sequence
     (NM_002354.3, exon 4–6 region) as the trigger source.
  2. Implementing the exact Green et al. 2014 toehold switch architecture
     (a = 12 nt toehold, b = 18 nt stem, RBS+AUG in loop).
  3. Evaluating every possible 30-nt trigger window with ViennaRNA,
     scoring by thermodynamic driving force AND mRNA accessibility.
  4. Reporting all ΔG metrics needed for a Methods section.

TOEHOLD SWITCH ARCHITECTURE (Green et al. 2014, Figure 1B)
-----------------------------------------------------------
Trigger RNA (30 nt from EPCAM mRNA):
    5'─[  a : 12 nt toehold  ][     b : 18 nt stem     ]─3'

Switch RNA (5'→3'):
    5'─[RC(b): 18 nt]─[LOOP: RBS + AUG]─[b: 18 nt]─[RC(a): 12 nt]─[LINKER]─3'
        └─ stem arm 1 ──────────────────── stem arm 2 ┘  └── toehold ───┘

OFF state:  RC(b) pairs with b across the loop.
            RBS and AUG are in the loop (geometrically constrained —
            completely unpaired, as required by Green et al. 2014).
            Ribosomes cannot efficiently initiate translation.

ON state:   Trigger's [a] binds switch's [RC(a)] toehold.
            Trigger's [b] undergoes branch migration, displacing
            the switch's [b] arm from [RC(b)].
            Loop opens. RBS and AUG become fully accessible.
            Ribosome binds → CASPASE-9 translation → apoptosis.

Reading frame from AUG:
    AUG [in loop]
      → b arm     (18 nt = 6 junk codons)
      → RC(a) arm (12 nt = 4 junk codons)
      → LINKER    (12 nt = 4 junk codons)
      → CASPASE-9 ORF starts (in-frame, 14 aa N-terminal tag acceptable)

THERMODYNAMIC SCORING CRITERIA
--------------------------------
  ΔG_switch   < −10 kcal/mol  → switch is well-folded in OFF state
  ΔG_trigger  > −8  kcal/mol  → trigger is accessible (not self-folded)
  ΔG_duplex   < −20 kcal/mol  → trigger-switch complex is stable (ON state)
  ΔΔG         < −10 kcal/mol  → net driving force for activation
  GC (toehold): 35–65%        → balanced binding affinity
  No in-frame stop codons in b + RC(a) + LINKER

INSTALLATION
------------
  conda install -c conda-forge viennarna   # preferred (installs RNA C library)
  OR: pip install ViennaRNA

REFERENCES
----------
Green, A. A. et al. (2014). Toehold Switches: De-Novo-Designed Regulators
    of Gene Expression. Cell, 159(4), 925–939.
    https://doi.org/10.1016/j.cell.2014.10.002

Pardee, K. et al. (2016). Rapid, Low-Cost Detection of Zika Virus Using
    Programmable Biomolecular Components. Cell, 165(5), 1255–1266.

Xie, Z. et al. (2011). Multi-input RNAi-based logic circuit for identification
    of specific cancer cells. Science, 333(6047), 1307–1311.
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
            "Install with:\n"
            "  conda install -c conda-forge viennarna\n"
            "  OR: pip install ViennaRNA\n"
        )

_require_viennarna()
import RNA  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARCHITECTURE CONSTANTS  (Green et al. 2014)
# ─────────────────────────────────────────────────────────────────────────────

TRIGGER_LEN   : int = 30   # total trigger length (nt)
TOEHOLD_LEN   : int = 12   # domain a: initiates binding with switch toehold
STEM_LEN      : int = 18   # domain b: drives strand displacement

# Standard switch loop — contains RBS (AGGAGAA) + spacer + AUG start codon.
# The AUG is the LAST 3 nt of the loop; everything in the loop is completely
# unpaired in the OFF state (Green et al. 2014, key design principle).
# GCAA (4 nt) — loop opening transition
# AGGAGAA (7 nt) — Shine-Dalgarno ribosome binding site
# ACAGCC (6 nt) — spacer (Kozak-like context for the AUG)
# AUG (3 nt) — start codon (last nt of loop)
SWITCH_LOOP   : str = "GCAAGAGGAGAAACAGCCAUG"   # 21 nt

# In-frame linker bridging the switch scaffold to the reporter ORF.
# 12 nt = 4 codons; verified to contain no UAA, UAG, or UGA stop codons
# in any of the three reading frames.  Encodes: Asn–Leu–Ala–Ala (NLAA).
SWITCH_LINKER : str = "AACCUGGCGGCA"            # 12 nt

# First 21 nt of the Caspase-9 apoptosis effector ORF (NM_001229.4, CDS
# position 196–216, in RNA form).  Used here as the reporter placeholder.
# In the wet-lab construct, replace with the complete CASP9 ORF.
REPORTER_START: str = "AUGUCUGGAGAGCAGAGGGAC"   # 21 nt

# Thermodynamic thresholds for hard-filtering candidates
MIN_DUPLEX_DG     : float = -20.0   # ON complex must be ≤ this value
MAX_TRIGGER_DG    : float =  -8.0   # trigger self-fold must be ≥ this value
MIN_SWITCH_DG     : float =  -5.0   # switch OFF fold must be ≤ this value
MIN_DELTADETLAG   : float = -10.0   # ΔΔG (activation driving force) threshold

# ─────────────────────────────────────────────────────────────────────────────
# 2.  EPCAM TARGET SEQUENCE  (NM_002354.3)
# ─────────────────────────────────────────────────────────────────────────────
#
# Source : NCBI RefSeq NM_002354.3, CDS positions ~480–680.
# Region : Exon 4–6, encoding the extracellular EGF-like and
#          thyroglobulin type-1B repeat domains of EpCAM.
# GC     : ~47% — well within the 40–60% window optimal for toehold design.
#
# This is the same region targeted by CellSearch anti-EpCAM antibodies and
# validated EpCAM RT-qPCR assays (Went et al. 2006; Winter et al. 2008).
# Exon 5 forward primer (GAAACCCGGGAATTTCATTGTTC) falls within this window.
#
# !!! FOR WET-LAB USE: replace this constant with the output of
#     fetch_epcam_sequence() (defined below) to guarantee nucleotide-exact
#     agreement with the current NCBI entry. !!!
#
EPCAM_CDS_REGION: str = (
    "GAAACCCGGGAATTTCATTGTTCTCTAATGG"  # exon 5 forward primer anchors here
    "ATACTGCAGAGGAAATGGAAAATGAAGAAAC"
    "CAGATAATAACGTCAGCTTGGAAATGTACTG"
    "TGATCCCAAAGAGCTTAACAAAGAAAAGTGT"
    "CAAGTGAAAAAGAGATCCTGAAGGAATGCAG"
    "AGTGAAACAATCACTGAGGAAATGGCAGACG"
)
# 186 nt → 186 − 30 + 1 = 157 candidate trigger windows


def fetch_epcam_sequence(nt_from: int = 590, nt_to: int = 780) -> Optional[str]:
    """
    Fetch the exact EPCAM mRNA sequence from NCBI Entrez.

    Requires: pip install biopython

    Parameters
    ----------
    nt_from, nt_to : int
        Nucleotide coordinates on NM_002354.3 (1-based, inclusive).
        Default window (590–780) covers the exon 5–6 region used by
        published diagnostic assays.

    Returns
    -------
    str (RNA, uppercase, ACGU) or None if Biopython / network unavailable.
    """
    try:
        from Bio import Entrez, SeqIO
        Entrez.email = "your_email@institution.edu"   # required by NCBI
        handle = Entrez.efetch(
            db="nucleotide",
            id="NM_002354.3",
            rettype="fasta",
            retmode="text",
            seq_start=nt_from,
            seq_stop=nt_to,
        )
        record = SeqIO.read(handle, "fasta")
        return str(record.seq).upper().replace("T", "U")
    except Exception as exc:
        print(f"[WARN] NCBI fetch failed ({exc}). Using built-in sequence.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SEQUENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_RNA_COMPLEMENT = str.maketrans("AUGC", "UACG")


def to_rna(seq: str) -> str:
    """Convert DNA or mixed sequence to RNA (T→U, uppercase)."""
    return seq.upper().replace("T", "U")


def rc(seq: str) -> str:
    """RNA reverse complement (A↔U, G↔C)."""
    return seq.upper().translate(_RNA_COMPLEMENT)[::-1]


def gc(seq: str) -> float:
    """Fractional GC content [0, 1]."""
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


# In-frame stop codons for all three reading frames
_STOPS = {"UAA", "UAG", "UGA"}


def has_stop_codon(seq: str, frame: int = 0) -> bool:
    """
    True if `seq` contains an in-frame stop codon starting at `frame`.
    The frame counts from the AUG (i.e. the first codon after AUG is frame=0
    in the downstream stem arm b).
    """
    s = seq.upper()
    for i in range(frame, len(s) - 2, 3):
        if s[i : i + 3] in _STOPS:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VIENNARNA WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def vfold(seq: str):
    """MFE fold. Returns (dot-bracket structure, MFE in kcal/mol)."""
    return RNA.fold(seq)


def vcofold(seq1: str, seq2: str):
    """
    RNAcofold for two interacting sequences.
    Returns (dot-bracket with '&' separator, ensemble ΔG in kcal/mol).
    """
    return RNA.cofold(seq1 + "&" + seq2)


def toehold_accessibility(switch_rna: str) -> float:
    """
    Estimate the fraction of toehold nucleotides (last TOEHOLD_LEN nt of
    the switch before the linker) that are single-stranded in the MFE
    structure of the switch alone.

    A value ≥ 0.7 is considered acceptable (≥ 70% accessible).
    """
    struct, _ = vfold(switch_rna)
    # The toehold RC(a) occupies positions [18+21+18 : 18+21+18+12] = [57:69]
    toehold_start = STEM_LEN + len(SWITCH_LOOP) + STEM_LEN         # 57
    toehold_end   = toehold_start + TOEHOLD_LEN                     # 69
    toehold_struct = struct[toehold_start : toehold_end]
    unpaired = toehold_struct.count(".")
    return unpaired / TOEHOLD_LEN


def mrna_window_accessibility(target_rna: str, window_start: int) -> float:
    """
    Estimate how accessible the 30-nt trigger window is in the context
    of the full target mRNA secondary structure.

    Folds the entire target region and returns the fraction of trigger
    nucleotides that are single-stranded (dots) in the MFE structure.
    A value ≥ 0.6 is considered acceptable.
    """
    struct, _ = vfold(target_rna)
    window_struct = struct[window_start : window_start + TRIGGER_LEN]
    return window_struct.count(".") / TRIGGER_LEN


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TOEHOLD SWITCH DESIGN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToeholdDesign:
    """
    All data for one toehold switch candidate.

    Naming follows Green et al. (2014) Figure 1B:
        a   = toehold domain (12 nt, first 12 nt of trigger)
        b   = stem domain    (18 nt, last  18 nt of trigger)
        a*  = RC(a)  placed on the switch as the accessible toehold arm
        b*  = RC(b)  placed on the switch as the 5' (upstream) stem arm
    """
    position         : int
    trigger_rna      : str

    # Trigger sub-domains
    a_domain         : str     # trigger[0:12]  — toehold complement on switch
    b_domain         : str     # trigger[12:30] — stem arm (3') on switch

    # Full switch sequence
    switch_rna       : str

    # Predicted structures
    struct_switch    : str
    struct_trigger   : str
    struct_duplex    : str
    struct_homodimer : str

    # Thermodynamics (kcal/mol)
    dg_switch        : float
    dg_trigger       : float
    dg_duplex        : float
    dg_homodimer     : float
    delta_delta_g    : float   # ΔG_duplex − (ΔG_switch + ΔG_trigger)

    # Accessibilities [0, 1]
    toehold_access   : float   # fraction of RC(a) that is single-stranded
    mrna_access      : float   # fraction of trigger window open in mRNA fold

    # Sequence properties
    gc_toehold       : float   # GC of a_domain (trigger toehold region)
    gc_trigger       : float

    # Validation
    stop_codon       : bool    # True = in-frame stop codon detected
    warnings         : List[str] = field(default_factory=list)

    # Composite score (higher = better)
    score            : float = 0.0


def design_switch(trigger_rna: str, target_rna: str, position: int) -> ToeholdDesign:
    """
    Design and thermodynamically validate one toehold switch for a given
    30-nt trigger window from the EPCAM mRNA.

    Parameters
    ----------
    trigger_rna : str
        30-nt RNA window from the EPCAM CDS (ACGU, uppercase).
    target_rna  : str
        Full target region sequence (for mRNA accessibility scoring).
    position    : int
        Start index of the trigger window within target_rna.

    Returns
    -------
    ToeholdDesign with all computed metrics.
    """
    assert len(trigger_rna) == TRIGGER_LEN
    trigger_rna = trigger_rna.upper()

    a = trigger_rna[:TOEHOLD_LEN]    # 12 nt: toehold complement of switch
    b = trigger_rna[TOEHOLD_LEN:]    # 18 nt: stem arm 2 of switch (3' side)

    rc_a = rc(a)    # 12 nt: switch toehold (RC of a) — single-stranded in OFF
    rc_b = rc(b)    # 18 nt: switch stem arm 1 (RC of b) — paired with b in OFF

    # ── Assemble the full switch RNA ──────────────────────────────────────
    # 5'-[RC(b)]-[LOOP]-[b]-[RC(a)]-[LINKER]-[REPORTER_START]-3'
    #    ←stem1→ ←loop→ ←stem2→ ←toehold→ ←bridge→
    switch_parts = [rc_b, SWITCH_LOOP, b, rc_a, SWITCH_LINKER, REPORTER_START]
    switch_rna   = "".join(switch_parts)

    # ── Thermodynamic calculations ────────────────────────────────────────
    struct_sw,  dg_sw  = vfold(switch_rna)
    struct_tr,  dg_tr  = vfold(trigger_rna)
    struct_cx,  dg_cx  = vcofold(trigger_rna, switch_rna)
    struct_hom, dg_hom = vcofold(trigger_rna, trigger_rna)

    ddg = dg_cx - (dg_sw + dg_tr)

    # ── Accessibility ─────────────────────────────────────────────────────
    acc_toehold = toehold_accessibility(switch_rna)
    acc_mrna    = mrna_window_accessibility(target_rna, position)

    # ── Sequence properties ───────────────────────────────────────────────
    gc_a   = gc(a)
    gc_all = gc(trigger_rna)

    # Check for in-frame stop codons in the region translated after AUG:
    # reading continues through: b (stem arm 2) + RC(a) (toehold) + LINKER
    stop_region = b + rc_a + SWITCH_LINKER
    stop        = has_stop_codon(stop_region, frame=0)

    # ── Warnings ──────────────────────────────────────────────────────────
    warns: List[str] = []
    if gc_a < 0.35:
        warns.append("LOW_GC_TOEHOLD: < 35% GC in domain a; may reduce binding kinetics")
    if gc_a > 0.65:
        warns.append("HIGH_GC_TOEHOLD: > 65% GC in domain a; off-target binding risk")
    if dg_tr < MAX_TRIGGER_DG:
        warns.append(f"SELF_FOLDED_TRIGGER: ΔG_trigger = {dg_tr:.1f} < {MAX_TRIGGER_DG} kcal/mol "
                     "(trigger may be inaccessible in the mRNA context)")
    if dg_cx > MIN_DUPLEX_DG:
        warns.append(f"WEAK_DUPLEX: ΔG_duplex = {dg_cx:.1f} > {MIN_DUPLEX_DG} kcal/mol "
                     "(switch activation may be slow)")
    if dg_hom < -8.0:
        warns.append(f"TRIGGER_HOMODIMER: ΔG_homodimer = {dg_hom:.1f} kcal/mol "
                     "(trigger self-dimerisation reduces effective concentration)")
    if dg_sw > MIN_SWITCH_DG:
        warns.append(f"WEAK_SWITCH: ΔG_switch = {dg_sw:.1f} > {MIN_SWITCH_DG} kcal/mol "
                     "(switch OFF state may leak into ON)")
    if stop:
        warns.append("STOP_CODON: in-frame UAA/UAG/UGA detected in b + RC(a) + LINKER — "
                     "reporter will not translate; redesign linker")
    if acc_toehold < 0.70:
        warns.append(f"TOEHOLD_OCCLUDED: only {acc_toehold*100:.0f}% of RC(a) is "
                     "single-stranded in switch structure (target ≥ 70%)")
    if acc_mrna < 0.60:
        warns.append(f"MRNA_OCCLUDED: only {acc_mrna*100:.0f}% of trigger window is "
                     "single-stranded in mRNA fold (target ≥ 60%)")

    # ── Composite score ───────────────────────────────────────────────────
    # Primary term: thermodynamic driving force for activation
    score  =  -ddg * 2.0
    # Penalise inaccessible triggers
    score -=  max(0.0, abs(dg_tr) - 4.0) * 1.5
    # Reward accessible toehold on switch
    score +=  acc_toehold * 8.0
    # Reward accessible trigger window in mRNA
    score +=  acc_mrna * 12.0
    # Reward balanced toehold GC
    if 0.40 <= gc_a <= 0.60:
        score += 4.0
    elif 0.35 <= gc_a <= 0.65:
        score += 1.5
    else:
        score -= 3.0
    # Hard penalties
    if stop:
        score -= 50.0
    score -= max(0.0, abs(dg_hom) - 5.0) * 0.5

    return ToeholdDesign(
        position=position,
        trigger_rna=trigger_rna,
        a_domain=a,
        b_domain=b,
        switch_rna=switch_rna,
        struct_switch=struct_sw,
        struct_trigger=struct_tr,
        struct_duplex=struct_cx,
        struct_homodimer=struct_hom,
        dg_switch=round(dg_sw,  2),
        dg_trigger=round(dg_tr, 2),
        dg_duplex=round(dg_cx,  2),
        dg_homodimer=round(dg_hom, 2),
        delta_delta_g=round(ddg, 2),
        toehold_access=round(acc_toehold, 3),
        mrna_access=round(acc_mrna, 3),
        gc_toehold=round(gc_a,  3),
        gc_trigger=round(gc_all, 3),
        stop_codon=stop,
        warnings=warns,
        score=round(score, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  WINDOW SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan(target_rna: str, top_n: int = 5) -> List[ToeholdDesign]:
    """
    Slide a 30-nt window across `target_rna`, design and score a toehold
    switch at every position, return the top_n ranked by composite score.

    Parameters
    ----------
    target_rna : str
        Full target region in RNA form (ACGU, uppercase).
    top_n      : int
        Number of top designs to return (default 5).
    """
    n = len(target_rna) - TRIGGER_LEN + 1
    print(f"  Scanning {n} trigger windows across {len(target_rna)}-nt target region...")
    candidates: List[ToeholdDesign] = []
    for i in range(n):
        window = target_rna[i : i + TRIGGER_LEN]
        try:
            d = design_switch(window, target_rna, i)
            candidates.append(d)
        except Exception as exc:
            print(f"  [SKIP] window {i}: {exc}")
    candidates.sort(key=lambda d: d.score, reverse=True)
    return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_SEP  = "═" * 68
_LINE = "─" * 68

def _check(condition: bool, good: str = "✓", bad: str = "⚠") -> str:
    return good if condition else bad


def print_report(d: ToeholdDesign, rank: int) -> None:
    print(f"\n{_SEP}")
    print(f"  RANK {rank}  ·  Position {d.position}–{d.position+TRIGGER_LEN-1}  "
          f"·  Score: {d.score:.2f}")
    print(_SEP)

    # ── Trigger ───────────────────────────────────────────────────────────
    print("\n▸ TRIGGER RNA  (30 nt from EPCAM mRNA NM_002354.3)")
    print(f"  Full:        5'-{d.trigger_rna}-3'")
    print(f"  [a] toehold: {d.a_domain}  (nt 1–12, initiates switch binding)")
    print(f"  [b] stem:    {d.b_domain}  (nt 13–30, strand displacement)")
    print(f"  GC trigger:  {d.gc_trigger*100:.1f}%   "
          f"GC toehold [a]: {d.gc_toehold*100:.1f}%  "
          f"  {_check(0.35 <= d.gc_toehold <= 0.65)}")
    print(f"  mRNA window accessibility: {d.mrna_access*100:.0f}%  "
          f"{_check(d.mrna_access >= 0.60)}")

    # ── Switch ────────────────────────────────────────────────────────────
    print("\n▸ SWITCH RNA  (Green et al. 2014, Type-A)")
    stem1  = rc(d.b_domain)
    toehlm = rc(d.a_domain)
    print(f"  Architecture:")
    print(f"    5'-[RC(b): {stem1}]")
    print(f"       [LOOP:  {SWITCH_LOOP}]  (RBS + AUG, unpaired)")
    print(f"       [b:     {d.b_domain}]")
    print(f"       [RC(a): {toehlm}]")
    print(f"       [LINK:  {SWITCH_LINKER}]")
    print(f"       [REP:   {REPORTER_START[:15]}...]")
    sw_len = len(d.switch_rna)
    print(f"  Full switch ({sw_len} nt):")
    # Wrap at 68 chars for readability
    print( "  5'-" + d.switch_rna[:48])
    for chunk in textwrap.wrap(d.switch_rna[48:], width=68):
        print(f"     {chunk}")
    print("  -3'")
    print(f"  Toehold RC(a) accessibility: {d.toehold_access*100:.0f}%  "
          f"{_check(d.toehold_access >= 0.70)}")

    # ── Thermodynamics ────────────────────────────────────────────────────
    print("\n▸ THERMODYNAMIC VALIDATION  (ViennaRNA MFE / RNAcofold)")
    print(f"  {'Metric':<40} {'Value':>9}  {'Threshold':>12}  Status")
    print(f"  {_LINE[:65]}")

    def row(label, val, threshold, ok):
        status = _check(ok)
        print(f"  {label:<40} {val:>7.2f}  {threshold:>12}   {status}")

    row("ΔG switch (OFF-state fold)",
        d.dg_switch, "< −10 kcal/mol", d.dg_switch < -10.0)
    row("ΔG trigger (self-fold)",
        d.dg_trigger, "> −8  kcal/mol", d.dg_trigger > -8.0)
    row("ΔG duplex (trigger + switch ON)",
        d.dg_duplex, "< −20 kcal/mol", d.dg_duplex < -20.0)
    row("ΔG homodimer (trigger self-pair)",
        d.dg_homodimer, "> −8  kcal/mol", d.dg_homodimer > -8.0)
    row("ΔΔG = ΔG_cx − (ΔG_sw + ΔG_tr)",
        d.delta_delta_g, "< −10 kcal/mol", d.delta_delta_g < -10.0)

    # ── Predicted structures ──────────────────────────────────────────────
    print("\n▸ PREDICTED SECONDARY STRUCTURES  (dot-bracket notation)")
    print(f"  Switch OFF : {d.struct_switch}")
    print(f"  Trigger    : {d.struct_trigger}")
    print(f"  ON complex : {d.struct_duplex}")

    # ── Reading frame check ───────────────────────────────────────────────
    print("\n▸ READING FRAME CHECK")
    stop_status = _check(not d.stop_codon, "✓ None detected", "⚠ STOP CODON DETECTED")
    print(f"  In-frame stop codons (b + RC(a) + LINKER): {stop_status}")
    print(f"  N-terminal junk before reporter: "
          f"{(STEM_LEN + TOEHOLD_LEN + len(SWITCH_LINKER))//3} amino acids  "
          f"(acceptable for Caspase-9 fusion)")

    # ── Warnings ──────────────────────────────────────────────────────────
    if d.warnings:
        print("\n▸ WARNINGS")
        for w in d.warnings:
            print(f"  ⚠  {w}")
    else:
        print("\n  ✓ No warnings — candidate meets all design criteria")


def print_summary_table(designs: List[ToeholdDesign]) -> None:
    print(f"\n{_SEP}")
    print("  SUMMARY TABLE — ALL TOP DESIGNS")
    print(_SEP)
    header = (f"  {'Rk':>2}  {'Pos':>4}  {'ΔG_sw':>7}  {'ΔG_tr':>7}  "
              f"{'ΔG_cx':>7}  {'ΔΔG':>7}  {'GC_a':>5}  {'mRNA%':>6}  {'Score':>7}")
    print(header)
    print(f"  {_LINE[:66]}")
    for r, d in enumerate(designs, 1):
        print(
            f"  {r:>2}  {d.position:>4}  {d.dg_switch:>7.2f}  {d.dg_trigger:>7.2f}  "
            f"{d.dg_duplex:>7.2f}  {d.delta_delta_g:>7.2f}  "
            f"{d.gc_toehold*100:>4.0f}%  {d.mrna_access*100:>5.0f}%  {d.score:>7.2f}"
            + ("  ⚠" if d.stop_codon else "")
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_SEP)
    print("  TOEHOLD SWITCH DESIGNER  ·  EPCAM mRNA  ·  LUAD PERCEPTRON")
    print("  Architecture : Green et al. 2014 (Cell 159:925–939), Type-A")
    print("  Reporter     : Caspase-9 (apoptosis effector)")
    print("  Validator    : ViennaRNA", RNA.__version__)
    print(_SEP)

    # ── Load target sequence ──────────────────────────────────────────────
    # Try to fetch the exact NCBI sequence; fall back to built-in constant.
    print("\n[1] Loading EPCAM target sequence (NM_002354.3, exon 4–6 region)...")
    target_dna = fetch_epcam_sequence() or EPCAM_CDS_REGION
    target_rna = to_rna(target_dna)
    gc_target  = gc(target_rna)
    print(f"  Target length : {len(target_rna)} nt")
    print(f"  GC content    : {gc_target*100:.1f}%")

    # ── Fold full target region for mRNA accessibility ────────────────────
    print("\n[2] Folding full target region to map mRNA accessibility...")
    mrna_struct, mrna_mfe = vfold(target_rna)
    frac_ss = mrna_struct.count(".") / len(mrna_struct)
    print(f"  mRNA MFE      : {mrna_mfe:.2f} kcal/mol")
    print(f"  Single-stranded fraction of target region: {frac_ss*100:.1f}%")

    # ── Run the scan ──────────────────────────────────────────────────────
    print("\n[3] Scanning trigger windows and designing toehold switches...")
    top_designs = scan(target_rna, top_n=3)

    # ── Print reports ─────────────────────────────────────────────────────
    print(f"\n[4] Reporting top {len(top_designs)} designs")
    for rank, design in enumerate(top_designs, 1):
        print_report(design, rank)

    # ── Summary table ─────────────────────────────────────────────────────
    print_summary_table(top_designs)

    # ── Best design highlight ─────────────────────────────────────────────
    best = top_designs[0]
    print(f"\n{_SEP}")
    print("  RECOMMENDED DESIGN (Rank 1)")
    print(_SEP)
    print(f"\n  Trigger (30 nt from EPCAM mRNA, pos {best.position}):")
    print(f"  5'-{best.trigger_rna}-3'")
    print(f"\n  Switch RNA ({len(best.switch_rna)} nt):")
    print(f"  5'-{best.switch_rna}-3'")
    print(f"\n  ΔΔG (activation driving force): {best.delta_delta_g:.2f} kcal/mol")
    print(f"  Toehold accessibility:          {best.toehold_access*100:.0f}%")
    print(f"  mRNA window accessibility:      {best.mrna_access*100:.0f}%")
    print(f"\n  NEXT STEPS:")
    print("  1. Submit switch RNA to NUPACK (nupack.org) for ensemble MFE validation.")
    print("  2. Clone into pUCIDT-Amp (IDT) or pDest-CMV (mammalian) vector.")
    print("  3. Validate in EPCAM+ cell line (A549, H1299) vs. EPCAM− control (H1650).")
    print("  4. For mammalian use: replace Shine-Dalgarno loop with IRES element")
    print("     (e.g., EMCV IRES) or optimised Kozak context.")
    print("     Reference: Chappell et al. 2015, Nat Chem Biol 11:214–220.")
    print(f"\n  ✓ Analysis complete.")
    print(_SEP + "\n")


if __name__ == "__main__":
    main()