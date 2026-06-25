"""Upload-triggered model update assessment and retraining."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.modeling import MODEL_TRAINING_SPECS, train_property_model, load_trained_model

TARGET_TO_MODEL: dict[str, str] = {
    "yield_strength_mpa": "structural_yield_strength",
    "ultimate_tensile_strength_mpa": "structural_tensile_strength",
    "band_gap_ev": "computed_band_gap",
    "formation_energy_per_atom": "computed_formation_energy",
    "bulk_modulus_gpa": "elastic_modulus_proxy",
    "shear_modulus_gpa": "elastic_modulus_proxy",
    "thermal_conductivity_w_mk": "thermal_conductivity",
    "comfort_score": "interior_foam_properties",
    "durability_score": "interior_foam_properties",
    "wear_resistance_score": "tyre_elastomer_properties",
}


def detect_upload_targets(mapped_df: pd.DataFrame) -> list[str]:
    return [t for t in TARGET_TO_MODEL if t in mapped_df.columns and mapped_df[t].notna().any()]


def assess_upload_model_impact(
    mapped_df: pd.DataFrame,
    unified_before: pd.DataFrame,
    trained_models: dict[str, dict],
) -> dict[str, Any]:
    """Assess whether upload can retrain or improves registry/scoring."""
    targets = detect_upload_targets(mapped_df)
    impacts: list[dict[str, Any]] = []

    for target in targets:
        model_key = TARGET_TO_MODEL[target]
        upload_rows = int(mapped_df[target].notna().sum())
        spec = MODEL_TRAINING_SPECS.get(model_key)
        min_rows = int(spec["min_rows"]) if spec else 100
        implemented = model_key in MODEL_TRAINING_SPECS

        if implemented and upload_rows > 0:
            combined_est = len(unified_before) + upload_rows
            if upload_rows >= min_rows or (model_key in trained_models and upload_rows >= 10):
                action = "model_retrain_available"
                message = f"Model update available for {model_key.replace('_', ' ')}."
            else:
                action = "scoring_improved"
                message = (
                    f"Added to scoring and passport library. "
                    f"More rows required before model training ({upload_rows}/{min_rows})."
                )
        elif upload_rows > 0:
            action = "trainable_target_detected"
            message = f"New trainable target detected: {target}."
        else:
            continue

        impacts.append({
            "target": target,
            "model_key": model_key,
            "upload_rows": upload_rows,
            "min_rows": min_rows,
            "action": action,
            "message": message,
            "implemented": implemented,
        })

    scoring_improved = len(targets) > 0
    return {
        "targets_detected": targets,
        "impacts": impacts,
        "scoring_coverage_improved": scoring_improved,
        "any_retrain_available": any(i["action"] == "model_retrain_available" for i in impacts),
    }


def retrain_affected_model(model_key: str, unified_df: pd.DataFrame) -> dict[str, Any]:
    """Retrain one model if implemented. Returns result metadata — no fake metrics."""
    if model_key not in MODEL_TRAINING_SPECS:
        return {
            "success": False,
            "model_key": model_key,
            "message": "Trainable target detected — model training not yet implemented for this property.",
        }

    before = load_trained_model(model_key)
    bundle = train_property_model(model_key, unified_df)
    if bundle is None:
        return {
            "success": False,
            "model_key": model_key,
            "message": "More evidence required before this model can be retrained.",
            "before_r2": before.get("r2") if before else None,
        }

    return {
        "success": True,
        "model_key": model_key,
        "message": "Model retrained",
        "selected_algorithm": bundle.get("selected_algorithm"),
        "r2": bundle.get("r2"),
        "mae": bundle.get("mae"),
        "rmse": bundle.get("rmse"),
        "rows": bundle.get("rows"),
        "before_r2": before.get("r2") if before else None,
    }
