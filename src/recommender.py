"""JLR-specific material recommender with use-case presets and reason codes."""

from __future__ import annotations
import numpy as np
import pandas as pd


def clamp(x: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, float(x)))


USE_CASE_PRESETS: dict[str, dict] = {
    "Lightweight Structural Component": {
        "weights": {
            "strength_score": 0.25,
            "weight_saving_score": 0.20,
            "stiffness_score": 0.15,
            "confidence_score": 0.10,
            "sustainability_score": 0.10,
            "manufacturability_score": 0.10,
            "cost_score": 0.05,
            "supply_risk_score": 0.05,
        },
        "description": "Structural body/chassis parts where strength-to-weight ratio is critical.",
    },
    "EV Battery Enclosure / Underbody": {
        "weights": {
            "strength_score": 0.20,
            "weight_saving_score": 0.15,
            "stiffness_score": 0.10,
            "confidence_score": 0.10,
            "sustainability_score": 0.10,
            "manufacturability_score": 0.10,
            "cost_score": 0.10,
            "supply_risk_score": 0.05,
            "corrosion_score": 0.10,
        },
        "description": "Battery housing requiring corrosion resistance, stiffness, and crash safety.",
    },
    "Interior Circular Material": {
        "weights": {
            "strength_score": 0.10,
            "weight_saving_score": 0.10,
            "sustainability_score": 0.30,
            "cost_score": 0.15,
            "supply_risk_score": 0.10,
            "confidence_score": 0.10,
            "manufacturability_score": 0.15,
        },
        "description": "Interior trim/panels prioritising recycled content and circularity.",
    },
    "Balanced Material Reuse": {
        "weights": {
            "strength_score": 0.20,
            "weight_saving_score": 0.15,
            "stiffness_score": 0.10,
            "confidence_score": 0.10,
            "sustainability_score": 0.15,
            "manufacturability_score": 0.10,
            "cost_score": 0.10,
            "supply_risk_score": 0.10,
        },
        "description": "Balanced trade-off across performance, sustainability, and supply chain.",
    },
}


def _safe_col(df: pd.DataFrame, col: str, default: float = 50.0) -> pd.Series:
    """Get column with fallback to default value for missing data."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def recommend_materials(
    unified: pd.DataFrame,
    min_strength: float,
    baseline_density: float = 7.85,
    use_case: str = "Lightweight Structural Component",
    min_trust: float = 0,
) -> pd.DataFrame:
    df = unified.copy()

    # Keep rows that have yield strength and density (or default density)
    df["yield_strength_mpa"] = pd.to_numeric(
        df.get("yield_strength_mpa", pd.Series(dtype=float)), errors="coerce"
    )
    df["density_g_cm3"] = pd.to_numeric(
        df.get("density_g_cm3", pd.Series(7.85, index=df.index)), errors="coerce"
    ).fillna(7.85)

    df = df.dropna(subset=["yield_strength_mpa"])

    # Trust filter
    trust = _safe_col(df, "source_trust_score", 50)
    df = df[trust >= min_trust].copy()

    if df.empty:
        return df

    # Sub-scores
    df["strength_score"] = df["yield_strength_mpa"].apply(
        lambda y: 100 if y >= min_strength else clamp((y / max(min_strength, 1)) * 100)
    )

    df["weight_saving_percent"] = (
        (baseline_density - df["density_g_cm3"]) / baseline_density * 100
    ).round(1)
    df["weight_saving_score"] = df["weight_saving_percent"].apply(lambda x: clamp(x * 2))

    ym = _safe_col(df, "youngs_modulus_gpa", 50)
    df["stiffness_score"] = (ym / 250 * 100).clip(0, 100).round(1)

    df["confidence_score"] = _safe_col(df, "source_trust_score", 70)

    recyclability = _safe_col(df, "recyclability_score", 50)
    recycled = _safe_col(df, "recycled_content_percent", 30)
    co2 = _safe_col(df, "co2_index", 50)
    df["sustainability_score"] = (
        0.4 * recyclability + 0.3 * recycled + 0.3 * (100 - co2)
    ).round(1)

    df["manufacturability_score"] = 50.0  # neutral placeholder

    df["cost_score"] = (100 - _safe_col(df, "cost_index", 50)).round(1)
    df["supply_risk_score"] = (100 - _safe_col(df, "supplier_risk_score", 50)).round(1)

    corr = _safe_col(df, "corrosion_resistance_score", 50)
    df["corrosion_score"] = corr.round(1)

    # Weighted suitability score
    preset = USE_CASE_PRESETS.get(use_case, USE_CASE_PRESETS["Lightweight Structural Component"])
    weights = preset["weights"]

    df["suitability_score"] = 0.0
    for metric, weight in weights.items():
        if metric in df.columns:
            df["suitability_score"] += weight * df[metric]
        else:
            df["suitability_score"] += weight * 50  # neutral for missing
    df["suitability_score"] = df["suitability_score"].round(1)

    df["reason_codes"] = df.apply(lambda r: _reason_codes(r, min_strength), axis=1)

    return df.sort_values("suitability_score", ascending=False).reset_index(drop=True)


def _reason_codes(row: pd.Series, min_strength: float) -> str:
    reasons: list[str] = []

    if row.get("strength_score", 0) >= 100:
        reasons.append("+ Meets minimum yield strength")
    else:
        reasons.append("- Below required strength")

    ws = row.get("weight_saving_percent", 0)
    if ws > 20:
        reasons.append("+ Strong weight-saving potential")
    elif ws > 0:
        reasons.append("+ Some weight-saving potential")
    else:
        reasons.append("- No weight saving vs baseline")

    if row.get("recyclability_score", 0) >= 75 or row.get("recycled_content_percent", 0) >= 30:
        reasons.append("+ Strong recyclability / recycled content")

    if row.get("source_trust_score", 0) >= 90:
        reasons.append("+ High source trust (experimental)")

    # Flag missing data
    missing_fields = []
    for field, label in [
        ("fatigue_strength_mpa", "Fatigue"),
        ("corrosion_resistance_score", "Corrosion"),
        ("supplier_risk_score", "Supplier risk"),
        ("youngs_modulus_gpa", "Stiffness"),
        ("hardness_hv", "Hardness"),
    ]:
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing_fields.append(label)

    if missing_fields:
        reasons.append(f"- Data missing: {', '.join(missing_fields)} — validation needed")

    return " | ".join(reasons)
