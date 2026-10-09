"""Heuristic biological screens for generated antimicrobial peptides.

These are the rule checks used as oracle penalties during TASAR sampling. They
encode the usual AMP design heuristics: a cationic but not over-charged
sequence, a reasonable hydrophobic/hydrophilic balance, no amyloid or
aggregation-prone motifs, no low-complexity repeats, and no unpaired cysteine
that could polymerize.

The thresholds are empirical rather than derived, and are deliberately strict:
they are meant to steer a sampler away from implausible sequences, not to serve
as a classifier of antimicrobial activity.
"""

import re
from collections import defaultdict

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

AGGREGATION_PRONE_MOTIFS = [
    "VLVL", "WWYF", "KLLL", "STVIIE", "VQIVYK", "GNNQQNY", "QYNNQ",
    "QQQQQQQQQ", "NNNNNNNN", "GSVIIE", "SSTY", "SVQIVY",
    "KLVFFA", "LVFFAEDVGSNK", "VTGVTAVAQK", "GSSGSS", "VLIVLG",
    "YVIVFV", "LIVVV", "WLIVI", "VVVIV", "LLVVV", "PXXPXX", "PQPQPQ",
    "LVFFA", "IYFV", "TTVIE", "NFGAIL", "DFNKF",
]

# The softer `is_peptide_probable` screen additionally rejects the "VVV" run.
AGGREGATION_PRONE_MOTIFS_STRICT = AGGREGATION_PRONE_MOTIFS + ["VVV"]

# pKa values of the ionizable groups, used for the charge model.
PKA_VALUES = {
    "N_term": 9.0,
    "C_term": 3.1,
    "D": 3.9,
    "E": 4.3,
    "C": 8.3,
    "Y": 10.1,
    "H": 6.0,
    "K": 10.5,
    "R": 12.5,
}


def is_valid_peptide(seq: str, min_len: int = 2, max_len: int = 50) -> bool:
    """Charset and length check: standard amino acids only."""
    return bool(seq) and min_len <= len(seq) <= max_len and all(c in AMINO_ACIDS for c in seq)


# --------------------------- descriptors ---------------------------

def hydrophobic_hydrophilic_balance(seq: str) -> float:
    hydrophobic = set("AVILMFWPY")
    hydrophilic = set("RNDQEHKSTC")
    if not seq:
        return 0.0
    n_phobic = sum(1 for aa in seq if aa in hydrophobic)
    n_philic = sum(1 for aa in seq if aa in hydrophilic)
    return abs(n_phobic - n_philic) / len(seq)


def count_consecutive_glycine_repeats(seq: str, max_consecutive: int = 2) -> int:
    return sum(1 for run in re.findall(r"(G+)", seq) if len(run) > max_consecutive)


def count_consecutive_bulky_repeats(seq: str, max_consecutive: int = 2) -> int:
    return sum(1 for run in re.findall(r"([WYF]+)", seq) if len(run) > max_consecutive)


def has_aggregation_motif(seq: str, strict: bool = False) -> bool:
    motifs = AGGREGATION_PRONE_MOTIFS_STRICT if strict else AGGREGATION_PRONE_MOTIFS
    return any(motif in seq for motif in motifs)


def proline_content(seq: str) -> float:
    return seq.count("P") / len(seq) if seq else 0.0


def net_positive_charge(seq: str) -> int:
    return sum(seq.count(aa) for aa in "RKH") - sum(seq.count(aa) for aa in "DE")


def cationic_content(seq: str) -> int:
    return sum(seq.count(aa) for aa in "RKH")


def has_repeated_tripeptide(seq: str, max_repeats: int = 3) -> bool:
    counts = defaultdict(int)
    for i in range(len(seq) - 2):
        counts[seq[i : i + 3]] += 1
        if counts[seq[i : i + 3]] > max_repeats:
            return True
    return False


def hydrophilic_abundance(seq: str) -> float:
    """Percentage of hydrophilic residues."""
    if not seq:
        return 0.0
    hydrophilic = set("KRHDESTNQY")
    return round(sum(1 for aa in seq if aa in hydrophilic) / len(seq) * 100, 2)


def max_consecutive_glycine(seq: str) -> int:
    runs = re.findall(r"(G+)", seq)
    return max((len(r) for r in runs), default=0)


def max_consecutive_bulky(seq: str) -> int:
    """Longest run of a single repeated bulky residue."""
    best = 0
    current = 0
    previous = None
    for aa in seq:
        if aa in set("FYWILVMRH") and aa == previous:
            current += 1
        elif aa in set("FYWILVMRH"):
            current = 1
        else:
            current = 0
        previous = aa
        best = max(best, current)
    return best


def count_polymerizing_cysteines(seq: str) -> int:
    return (seq.count("C") // 2) * 2


def _henderson_hasselbalch(pka: float, ph: float, is_acidic: bool) -> float:
    ratio = 10 ** (pka - ph) if is_acidic else 10 ** (ph - pka)
    return -1 / (1 + ratio) if is_acidic else 1 / (1 + ratio)


def net_charge_at_ph(seq: str, ph: float = 7.0) -> float:
    """Net charge from the Henderson-Hasselbalch equation."""
    charge = _henderson_hasselbalch(PKA_VALUES["N_term"], ph, is_acidic=False)
    charge += _henderson_hasselbalch(PKA_VALUES["C_term"], ph, is_acidic=True)
    for aa in seq:
        if aa in PKA_VALUES:
            charge += _henderson_hasselbalch(PKA_VALUES[aa], ph, is_acidic=aa in "DECY")
    return round(charge, 2)


# --------------------------- rule checks ---------------------------

def is_peptide_ok(seq: str) -> bool:
    """Hard design constraints: length, charge, composition, repeats."""
    if not 8 <= len(seq) <= 50:
        return False
    if not 0.30 <= hydrophobic_hydrophilic_balance(seq) <= 0.55:
        return False
    if count_consecutive_glycine_repeats(seq) > 0:
        return False
    if count_consecutive_bulky_repeats(seq) > 1:
        return False
    if has_aggregation_motif(seq):
        return False
    if proline_content(seq) > 0.2:
        return False
    if not 3 <= net_positive_charge(seq) <= 8:
        return False
    if cationic_content(seq) > max(6, int(0.25 * len(seq))):
        return False
    if has_repeated_tripeptide(seq):
        return False
    return True


def is_peptide_probable(seq: str) -> bool:
    """Softer plausibility screen on composition and charge at pH 7.4."""
    if not 30 <= hydrophilic_abundance(seq) <= 55:
        return False
    if max_consecutive_glycine(seq) > 2:
        return False
    if max_consecutive_bulky(seq) > 1:
        return False
    if has_aggregation_motif(seq, strict=True):
        return False
    if proline_content(seq) * 100 > 20:
        return False
    if count_polymerizing_cysteines(seq) > 1:
        return False
    if not 3.5 <= net_charge_at_ph(seq, ph=7.4) <= 6.5:
        return False
    return True
