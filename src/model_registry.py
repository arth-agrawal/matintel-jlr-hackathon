"""Model registry for MatIntel — tracks implemented, trainable, and waiting models.

Each model spec defines subsystem, target, required features, eligible source types,
and current implementation status.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


MODEL_SPECS: dict[str, dict] = {
    "structural_yield_strength": {
        "subsystem": "Structural / Chassis",
        "target": "yield_strength_mpa",
        "required_features": "wt_percent_* composition columns",
        "eligible_source_types": ["public_experimental", "experimental_test"],
        "model_type": "RandomForestRegressor (250 trees)",
        "status": "implemented",
        "description": "Yield strength prediction from steel composition features.",
    },
    "structural_tensile_strength": {
        "subsystem": "Structural / Chassis",
        "target": "ultimate_tensile_strength_mpa",
        "required_features": "wt_percent_* composition columns",
        "eligible_source_types": ["public_experimental", "experimental_test"],
        "model_type": "RandomForestRegressor",
        "status": "trainable",
        "description": "Tensile strength prediction — trainable if target column has labelled rows.",
    },
    "computed_formation_energy": {
        "subsystem": "General Material Reuse / Battery / Electronics",
        "target": "formation_energy_per_atom",
        "required_features": "composition-derived features",
        "eligible_source_types": ["computed_database"],
        "model_type": "RandomForestRegressor",
        "status": "trainable",
        "description": "Formation energy prediction from computed reference data (e.g. JARVIS).",
    },
    "computed_band_gap": {
        "subsystem": "Electronics / Thermal Interface",
        "target": "band_gap_ev",
        "required_features": "composition-derived features",
        "eligible_source_types": ["computed_database"],
        "model_type": "RandomForestRegressor",
        "status": "trainable",
        "description": "Band gap prediction from computed reference data.",
    },
    "thermal_conductivity": {
        "subsystem": "Thermal Fluids / Coatings / Electronics",
        "target": "thermal_conductivity_w_mk",
        "required_features": "composition or material descriptor features",
        "eligible_source_types": ["public_reference", "experimental_test", "public_experimental"],
        "model_type": "RandomForestRegressor",
        "status": "waiting",
        "description": "Thermal conductivity prediction — waiting for labelled evidence data.",
    },
}


def _eligible_rows(unified_df: pd.DataFrame, target: str, eligible_types: list[str]) -> pd.Series:
    """Rows with labelled target from eligible source types (respects model_registry_eligible)."""
    if target not in unified_df.columns:
        return pd.Series(False, index=unified_df.index)

    target_present = unified_df[target].notna()
    if "source_type" in unified_df.columns:
        type_match = unified_df["source_type"].isin(eligible_types)
    else:
        type_match = pd.Series(False, index=unified_df.index)

    mask = target_present & type_match
    if "model_registry_eligible" in unified_df.columns:
        eligible_flag = unified_df["model_registry_eligible"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        mask = mask & eligible_flag
    return mask


def detect_trainable_targets(unified_df: pd.DataFrame) -> pd.DataFrame:
    """Scan unified data and return model registry status table."""
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        eligible_types = spec["eligible_source_types"]

        eligible_mask = _eligible_rows(unified_df, target, eligible_types)
        eligible_count = int(eligible_mask.sum())

        # Detect features
        if "wt_percent_" in spec["required_features"]:
            feature_cols = [c for c in unified_df.columns if c.startswith("wt_percent_")]
        else:
            feature_cols = [c for c in unified_df.columns if c.startswith("wt_percent_") or c == "density_g_cm3"]
        feature_count = len(feature_cols)

        # Determine status
        if spec["status"] == "implemented":
            status = "Active"
            reason = f"Trained on {eligible_count} eligible rows with {feature_count} features."
        elif eligible_count >= 30 and feature_count >= 3:
            status = "Trainable"
            reason = f"{eligible_count} eligible rows, {feature_count} features available."
        elif eligible_count > 0:
            status = "Insufficient data"
            reason = f"Only {eligible_count} eligible rows (need 30+) or {feature_count} features."
        else:
            status = "Waiting for evidence"
            reason = (
                f"Waiting for eligible evidence — no labelled {target} rows from "
                f"{', '.join(eligible_types)} sources."
            )

        rows.append({
            "model_name": model_name,
            "subsystem": spec["subsystem"],
            "target": target,
            "eligible_rows": eligible_count,
            "feature_count": feature_count,
            "status": status,
            "reason": reason,
            "model_type": spec["model_type"],
        })

    return pd.DataFrame(rows)


def get_active_models(unified_df: pd.DataFrame) -> list[dict]:
    """Return list of implemented/trainable model dicts."""
    registry = detect_trainable_targets(unified_df)
    return registry[registry["status"].isin(["Active", "Trainable"])].to_dict("records")
