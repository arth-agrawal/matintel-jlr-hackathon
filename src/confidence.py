"""Confidence and risk scoring for material property predictions."""

from __future__ import annotations
import numpy as np
import pandas as pd

SOURCE_PENALTIES: dict[str, int] = {
    "public_experimental": 0,
    "experimental": 0,
    "experimental_test": 2,
    "computed_database": 10,
    "computed": 10,
    "supplier_sheet": 12,
    "sustainability_sheet": 18,
    "procurement_sheet": 18,
    "public_reference": 15,
    "predicted": 20,
    "demo_enriched": 25,
    "unknown": 20,
}


def confidence_score(row: pd.Series, pred_info: dict, bundle: dict) -> dict:
    pred = max(float(pred_info["prediction"]), 1.0)
    width = float(pred_info["uncertainty_width"])

    # Penalty 1: Model uncertainty (wide prediction interval = less certain)
    uncertainty_penalty = min(30, (width / pred) * 100)

    # Penalty 2: Missing key engineering fields
    required = ["material_family", "family", "source_type", "density_g_cm3",
                 "elongation_percent", "youngs_modulus_gpa"]
    missing_count = 0
    for c in required:
        val = row.get(c)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing_count += 1
    missing_data_penalty = missing_count * 3

    # Penalty 3: Source type reliability
    source = str(row.get("source_type", "unknown")).lower()
    source_risk_penalty = SOURCE_PENALTIES.get(source, 15)

    # Penalty 4: Out-of-distribution detection via z-score
    feature_cols = bundle.get("feature_cols", [])
    mean = bundle.get("train_feature_mean", {})
    std = bundle.get("train_feature_std", {})

    z_vals = []
    for col in feature_cols:
        val = row.get(col, np.nan)
        if pd.notna(val):
            z = abs((float(val) - float(mean.get(col, 0))) / float(std.get(col, 1)))
            z_vals.append(z)

    avg_z = float(np.mean(z_vals)) if z_vals else 2.0
    out_of_distribution_penalty = min(25, avg_z * 4)

    score = 100 - uncertainty_penalty - missing_data_penalty - source_risk_penalty - out_of_distribution_penalty
    score = max(0, min(100, score))

    if score >= 80:
        risk = "Green"
    elif score >= 60:
        risk = "Amber"
    else:
        risk = "Red"

    return {
        "confidence": round(score, 1),
        "risk": risk,
        "uncertainty_penalty": round(uncertainty_penalty, 1),
        "missing_data_penalty": round(missing_data_penalty, 1),
        "source_risk_penalty": round(source_risk_penalty, 1),
        "out_of_distribution_penalty": round(out_of_distribution_penalty, 1),
    }
