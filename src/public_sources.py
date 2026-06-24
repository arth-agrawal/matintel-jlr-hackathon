"""Public dataset registry and optional loaders for MatIntel.

Each registered dataset has provenance, domain, trust, and ML eligibility metadata.
Loaders are optional and fail gracefully — no crashes if dependencies are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PUBLIC_DATASET_REGISTRY: dict[str, dict] = {
    "matminer_steel_strength": {
        "name": "Matminer steel_strength",
        "source_type": "public_experimental",
        "application_subsystem": "Structural / Chassis",
        "trust_score": 95,
        "used_for_ml_training": True,
        "trainable_targets": ["yield_strength_mpa", "ultimate_tensile_strength_mpa"],
        "rows": 312,
        "status": "loaded",
        "description": (
            "Experimental steel alloy strength data (312 alloys with composition + "
            "yield/tensile/elongation). Used for structural/chassis property prediction."
        ),
    },
    "jarvis_dft_3d_sample": {
        "name": "JARVIS-DFT 3D sample",
        "source_type": "computed_database",
        "application_subsystem": "General Material Reuse / Battery / Electronics / Thermal Interface",
        "trust_score": 80,
        "used_for_ml_training": False,
        "trainable_targets": [
            "formation_energy_per_atom", "band_gap_ev",
            "bulk_modulus_gpa", "shear_modulus_gpa",
        ],
        "rows": 0,
        "status": "optional",
        "description": (
            "Broad computed materials reference dataset for inorganic materials screening. "
            "Trainable for computed-reference models, not for experimental models by default."
        ),
    },
    "nasa_tpsx_reference": {
        "name": "NASA TPSX reference/upload pathway",
        "source_type": "public_reference",
        "application_subsystem": "Thermal Fluids / Coolants / Coatings / Composites",
        "trust_score": 80,
        "used_for_ml_training": False,
        "trainable_targets": [
            "thermal_conductivity_w_mk", "specific_heat", "density_g_cm3",
        ],
        "rows": 0,
        "status": "pathway",
        "description": (
            "Thermal/material property reference pathway for thermal protection, coatings, "
            "composites. Requires CSV export from NASA TPSX and engineer review."
        ),
    },
}


def load_jarvis_dft_sample(max_rows: int = 500) -> tuple[pd.DataFrame, str]:
    """Load a sample from JARVIS-DFT 3D dataset.

    Returns (dataframe, status_message). If unavailable, returns empty df + clear message.
    """
    try:
        from jarvis.db.figshare import data as jdata
    except ImportError:
        return pd.DataFrame(), "jarvis-tools not installed. Run: pip install jarvis-tools"

    try:
        raw = jdata("dft_3d")
    except Exception as e:
        return pd.DataFrame(), f"JARVIS data download failed: {e}"

    if not raw:
        return pd.DataFrame(), "JARVIS returned empty data."

    rows = raw[:max_rows] if len(raw) > max_rows else raw

    records = []
    for entry in rows:
        jid = entry.get("jid", "")
        formula = entry.get("formula", entry.get("reduced_formula", ""))
        density = entry.get("density", None)

        bulk = entry.get("bulk_modulus_kv", entry.get("kv", None))
        shear = entry.get("shear_modulus_gv", entry.get("gv", None))

        form_e = entry.get("formation_energy_peratom", entry.get("form_enp", None))
        bandgap = entry.get("optb88vdw_bandgap", entry.get("mbj_bandgap", None))

        records.append({
            "material_id": jid,
            "material_name": formula,
            "formula": formula,
            "material_family": "Inorganic compound",
            "material_subfamily": "",
            "family": "Inorganic compound",
            "source_dataset": "jarvis_dft_3d_sample",
            "source_type": "computed_database",
            "source_trust_score": 80,
            "used_for_ml_training": False,
            "model_registry_eligible": True,
            "application_subsystem": "General Material Reuse",
            "density_g_cm3": _safe_float(density),
            "bulk_modulus_gpa": _safe_float(bulk),
            "shear_modulus_gpa": _safe_float(shear),
            "formation_energy_per_atom": _safe_float(form_e),
            "band_gap_ev": _safe_float(bandgap),
            "yield_strength_mpa": np.nan,
            "ultimate_tensile_strength_mpa": np.nan,
            "elongation_percent": np.nan,
            "youngs_modulus_gpa": np.nan,
            "thermal_conductivity_w_mk": np.nan,
            "specific_heat": np.nan,
            "notes": "Computed reference row; not experimental validation.",
        })

    df = pd.DataFrame(records)

    key_cols = [
        "density_g_cm3", "bulk_modulus_gpa", "shear_modulus_gpa",
        "formation_energy_per_atom", "band_gap_ev",
    ]
    present = [c for c in key_cols if c in df.columns]
    if present:
        df["data_completeness_score"] = (
            100 * (1 - df[present].isna().mean(axis=1))
        ).round(1)
    else:
        df["data_completeness_score"] = 0.0

    return df, f"Loaded {len(df)} JARVIS-DFT rows."


def _safe_float(val) -> float:
    if val is None:
        return np.nan
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan
