"""Data pipeline: load matminer steel data and build JLR-relevant unified schema."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "steel_strength_raw.csv"
UNIFIED_PATH = PROJECT_ROOT / "data" / "processed" / "unified_material_library.csv"

DEMO_ENRICHMENT_COLS = [
    "density_g_cm3", "cost_index", "co2_index", "co2_kg_per_kg",
    "recycled_content_percent", "recyclability_score", "bio_based_content_percent",
    "closed_loop_available", "supplier_name", "supplier_risk_score",
    "traceability_score", "critical_material_risk_score", "certification_tags",
    "corrosion_resistance_score", "fatigue_strength_mpa", "hardness_hv",
    "youngs_modulus_gpa",
]

# Optional subsystem fields — NaN unless sourced from trusted datasets or uploads
OPTIONAL_SCHEMA_COLS = [
    "bulk_modulus_gpa", "shear_modulus_gpa", "formation_energy_per_atom", "band_gap_ev",
    "specific_heat", "viscosity", "freezing_point", "boiling_point",
    "impact_resistance_score", "flame_retardancy_score", "electrical_insulation_score",
    "comfort_score", "durability_score", "voc_emission_score", "fire_safety_score",
    "wear_resistance_score", "rolling_resistance_score", "grip_score",
    "salt_spray_hours", "adhesion_score", "scratch_resistance_score",
    "toxicity_score", "corrosion_inhibition_score", "thermal_conductivity_w_mk",
]

ELEMENT_COLS = [
    "c", "mn", "si", "cr", "ni", "mo", "v", "nb", "al", "co", "n", "cu", "ti", "w", "p", "s",
]


def _normalize_col(col: str) -> str:
    return (
        str(col).strip().lower()
        .replace(" ", "_").replace("-", "_")
        .replace("(", "").replace(")", "").replace("/", "_")
    )


def load_real_steel_data() -> pd.DataFrame:
    """Load the matminer steel_strength dataset, falling back to local CSV."""
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)

    try:
        from matminer.datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "matminer is not installed and no local CSV found. "
            "Run: python scripts/bootstrap_data.py"
        )

    last_err = None
    for name in ["steel_strength", "matbench_steels"]:
        try:
            df = load_dataset(name)
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_PATH, index=False)
            return df
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not download matminer steel dataset ({last_err}). "
        "Run 'python scripts/bootstrap_data.py' once with internet."
    )


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col(cand)
        if key in normalized:
            return normalized[key]
    for key, original in normalized.items():
        for cand in candidates:
            if _normalize_col(cand) in key:
                return original
    return None


def build_unified_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw matminer steel data into MatIntel JLR-relevant unified schema."""
    df = raw.copy()
    df.columns = [_normalize_col(c) for c in df.columns]

    y_col = _find_col(df, ["yield_strength", "ys", "yield_strength_mpa"])
    uts_col = _find_col(df, ["tensile_strength", "ultimate_tensile_strength", "uts"])
    elong_col = _find_col(df, ["elongation", "elongation_percent"])
    formula_col = _find_col(df, ["formula", "composition"])

    if y_col is None:
        raise ValueError(f"Could not identify yield strength column. Columns: {list(df.columns)}")

    unified = pd.DataFrame()
    unified["material_id"] = [f"STEEL_{i:04d}" for i in range(len(df))]
    unified["material_name"] = df[formula_col].astype(str) if formula_col else "Steel alloy"
    unified["formula"] = unified["material_name"]

    unified["material_family"] = "Steel"
    unified["material_subfamily"] = "High-strength steel"
    unified["family"] = unified["material_family"]
    unified["application_subsystem"] = "Structural / Chassis"

    unified["source_dataset"] = "matminer_steel_strength"
    unified["source_type"] = "public_experimental"
    unified["source_trust_score"] = 95
    unified["used_for_ml_training"] = True
    unified["model_registry_eligible"] = True

    # Real measured properties
    unified["yield_strength_mpa"] = pd.to_numeric(df[y_col], errors="coerce")
    unified["ultimate_tensile_strength_mpa"] = (
        pd.to_numeric(df[uts_col], errors="coerce") if uts_col else np.nan
    )
    unified["elongation_percent"] = (
        pd.to_numeric(df[elong_col], errors="coerce") if elong_col else np.nan
    )

    # Demo enrichment — not from source dataset
    unified["density_g_cm3"] = 7.85
    unified["youngs_modulus_gpa"] = 200.0
    unified["hardness_hv"] = np.nan
    unified["fatigue_strength_mpa"] = np.nan
    unified["corrosion_resistance_score"] = np.nan

    for col in OPTIONAL_SCHEMA_COLS:
        unified[col] = np.nan

    unified["cost_index"] = 35
    unified["co2_index"] = 60
    unified["co2_kg_per_kg"] = 1.8
    unified["recycled_content_percent"] = 40
    unified["recyclability_score"] = 85
    unified["bio_based_content_percent"] = 0
    unified["closed_loop_available"] = True

    unified["supplier_name"] = "Open-source dataset"
    unified["supplier_risk_score"] = 25
    unified["traceability_score"] = 90
    unified["critical_material_risk_score"] = 15
    unified["certification_tags"] = ""
    unified["prediction_confidence_score"] = np.nan
    unified["notes"] = ""

    # Preserve real element/composition columns
    for col in ELEMENT_COLS:
        if col in df.columns:
            unified[f"wt_percent_{col}"] = pd.to_numeric(df[col], errors="coerce")

    # Completeness based on real measured fields only (not demo enrichment)
    measured_keys = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_percent"]
    unified["data_completeness_score"] = (
        100 * (1 - unified[measured_keys].isna().mean(axis=1))
    ).round(1)

    UNIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(UNIFIED_PATH, index=False)
    return unified


def load_or_create_unified() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw + unified data, rebuilding unified if schema is stale."""
    raw = load_real_steel_data()
    needs_rebuild = not UNIFIED_PATH.exists()
    if not needs_rebuild:
        unified = pd.read_csv(UNIFIED_PATH)
        required_cols = ["application_subsystem", "model_registry_eligible"] + OPTIONAL_SCHEMA_COLS
        if any(c not in unified.columns for c in required_cols):
            needs_rebuild = True
    if needs_rebuild:
        unified = build_unified_schema(raw)
    else:
        unified = pd.read_csv(UNIFIED_PATH)
    return raw, unified
