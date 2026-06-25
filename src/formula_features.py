"""Formula / composition descriptor features for computed-reference models."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# Minimal periodic table for formula parsing (atomic number, atomic weight).
ELEMENT_DATA: dict[str, tuple[int, float]] = {
    "H": (1, 1.008), "He": (2, 4.003), "Li": (3, 6.94), "Be": (4, 9.012),
    "B": (5, 10.81), "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999),
    "F": (9, 18.998), "Ne": (10, 20.180), "Na": (11, 22.990), "Mg": (12, 24.305),
    "Al": (13, 26.982), "Si": (14, 28.085), "P": (15, 30.974), "S": (16, 32.06),
    "Cl": (17, 35.45), "Ar": (18, 39.948), "K": (19, 39.098), "Ca": (20, 40.078),
    "Sc": (21, 44.956), "Ti": (22, 47.867), "V": (23, 50.942), "Cr": (24, 51.996),
    "Mn": (25, 54.938), "Fe": (26, 55.845), "Co": (27, 58.933), "Ni": (28, 58.693),
    "Cu": (29, 63.546), "Zn": (30, 65.38), "Ga": (31, 69.723), "Ge": (32, 72.630),
    "As": (33, 74.922), "Se": (34, 78.971), "Br": (35, 79.904), "Kr": (36, 83.798),
    "Rb": (37, 85.468), "Sr": (38, 87.62), "Y": (39, 88.906), "Zr": (40, 91.224),
    "Nb": (41, 92.906), "Mo": (42, 95.95), "Tc": (43, 98.0), "Ru": (44, 101.07),
    "Rh": (45, 102.91), "Pd": (46, 106.42), "Ag": (47, 107.87), "Cd": (48, 112.41),
    "In": (49, 114.82), "Sn": (50, 118.71), "Sb": (51, 121.76), "Te": (52, 127.60),
    "I": (53, 126.90), "Xe": (54, 131.29), "Cs": (55, 132.91), "Ba": (56, 137.33),
    "La": (57, 138.91), "Ce": (58, 140.12), "Pr": (59, 140.91), "Nd": (60, 144.24),
    "Sm": (62, 150.36), "Eu": (63, 151.96), "Gd": (64, 157.25), "Tb": (65, 158.93),
    "Dy": (66, 162.50), "Ho": (67, 164.93), "Er": (68, 167.26), "Tm": (69, 168.93),
    "Yb": (70, 173.05), "Lu": (71, 174.97), "Hf": (72, 178.49), "Ta": (73, 180.95),
    "W": (74, 183.84), "Re": (75, 186.21), "Os": (76, 190.23), "Ir": (77, 192.22),
    "Pt": (78, 195.08), "Au": (79, 196.97), "Hg": (80, 200.59), "Tl": (81, 204.38),
    "Pb": (82, 207.2), "Bi": (83, 208.98), "Po": (84, 209.0), "At": (85, 210.0),
    "Rn": (86, 222.0),
}

_FORMULA_TOKEN = re.compile(
    r"([A-Z][a-z]?)(\d*\.?\d*)",
)

DESCRIPTOR_COLS = [
    "n_elements",
    "mean_atomic_number",
    "min_atomic_number",
    "max_atomic_number",
    "range_atomic_number",
    "mean_atomic_weight",
    "total_atoms",
    "density_g_cm3_feat",
    "source_trust_score_feat",
]


def parse_formula_counts(formula: str) -> dict[str, float]:
    """Parse a chemical formula into element counts. Returns {} on failure."""
    if not formula or not isinstance(formula, str):
        return {}
    text = formula.strip()
    if not text or text.lower() in {"nan", "none", "—", "-"}:
        return {}
    # Strip common prefixes / IDs
    text = re.sub(r"^JARVIS[_-]?", "", text)
    text = re.sub(r"^matbench[_-]\w+[_-]?", "", text, flags=re.IGNORECASE)
    text = text.replace(" ", "")

    counts: dict[str, float] = {}
    pos = 0
    while pos < len(text):
        m = _FORMULA_TOKEN.match(text, pos)
        if not m:
            return {}
        elem, amt = m.group(1), m.group(2)
        if elem not in ELEMENT_DATA:
            return {}
        n = float(amt) if amt else 1.0
        counts[elem] = counts.get(elem, 0.0) + n
        pos = m.end()
    if pos != len(text) or not counts:
        return {}
    return counts


def formula_to_descriptors(formula: str) -> dict[str, float]:
    """Compute composition descriptors from a formula string."""
    counts = parse_formula_counts(formula)
    if not counts:
        return {c: np.nan for c in DESCRIPTOR_COLS if c not in ("density_g_cm3_feat", "source_trust_score_feat")}

    atomic_numbers = [ELEMENT_DATA[e][0] for e in counts]
    atomic_weights = [ELEMENT_DATA[e][1] for e in counts]
    amounts = list(counts.values())
    total_atoms = float(sum(amounts))
    weighted_z = sum(ELEMENT_DATA[e][0] * counts[e] for e in counts) / total_atoms
    weighted_aw = sum(ELEMENT_DATA[e][1] * counts[e] for e in counts) / total_atoms

    return {
        "n_elements": float(len(counts)),
        "mean_atomic_number": float(weighted_z),
        "min_atomic_number": float(min(atomic_numbers)),
        "max_atomic_number": float(max(atomic_numbers)),
        "range_atomic_number": float(max(atomic_numbers) - min(atomic_numbers)),
        "mean_atomic_weight": float(weighted_aw),
        "total_atoms": total_atoms,
    }


def add_formula_descriptor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add formula descriptor columns to a dataframe copy."""
    out = df.copy()
    formulas = out["formula"].astype(str) if "formula" in out.columns else pd.Series("", index=out.index)

    desc_rows: list[dict[str, Any]] = []
    for formula in formulas:
        desc = formula_to_descriptors(formula)
        desc_rows.append(desc)
    desc_df = pd.DataFrame(desc_rows, index=out.index)

    for col in desc_df.columns:
        out[col] = desc_df[col]

    if "density_g_cm3" in out.columns:
        out["density_g_cm3_feat"] = pd.to_numeric(out["density_g_cm3"], errors="coerce")
    else:
        out["density_g_cm3_feat"] = np.nan

    if "source_trust_score" in out.columns:
        out["source_trust_score_feat"] = pd.to_numeric(out["source_trust_score"], errors="coerce")
    else:
        out["source_trust_score_feat"] = np.nan

    return out


def get_formula_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return usable formula descriptor feature columns."""
    base = [
        "n_elements", "mean_atomic_number", "min_atomic_number", "max_atomic_number",
        "range_atomic_number", "mean_atomic_weight", "total_atoms",
    ]
    cols = [c for c in base if c in df.columns and df[c].notna().any()]
    if "density_g_cm3_feat" in df.columns and df["density_g_cm3_feat"].notna().sum() >= 10:
        cols.append("density_g_cm3_feat")
    if "source_trust_score_feat" in df.columns and df["source_trust_score_feat"].notna().sum() >= 10:
        cols.append("source_trust_score_feat")
    return cols
