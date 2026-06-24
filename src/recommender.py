"""JLR-specific material recommender with use-case presets, dynamic thresholds,
density variance detection, and improved strength scoring."""

from __future__ import annotations
import numpy as np
import pandas as pd


def clamp(x: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, float(x)))


STRENGTH_MODES = {
    "Moderate structural": 0.50,
    "High strength structural": 0.75,
    "Extreme strength structural": 0.90,
    "Custom": None,
}

USE_CASE_PRESETS: dict[str, dict] = {
    "Lightweight Structural Component": {
        "weights": {
            "strength_score": 0.25, "weight_saving_score": 0.20,
            "stiffness_score": 0.15, "source_trust_score_norm": 0.10,
            "sustainability_score": 0.10, "manufacturability_score": 0.10,
            "cost_score": 0.05, "supply_risk_score": 0.05,
        },
        "description": "Structural body/chassis parts where strength-to-weight ratio is critical.",
        "subsystem": "Structural / Chassis",
    },
    "EV Battery Enclosure / Underbody": {
        "weights": {
            "strength_score": 0.20, "weight_saving_score": 0.15,
            "stiffness_score": 0.10, "source_trust_score_norm": 0.10,
            "sustainability_score": 0.10, "manufacturability_score": 0.10,
            "cost_score": 0.10, "supply_risk_score": 0.05,
            "corrosion_score": 0.10,
        },
        "description": "Battery housing requiring corrosion resistance, stiffness, and crash safety.",
        "subsystem": "Battery Enclosure / Underbody",
    },
    "Interior Circular Material": {
        "weights": {
            "strength_score": 0.10, "weight_saving_score": 0.10,
            "sustainability_score": 0.30, "cost_score": 0.15,
            "supply_risk_score": 0.10, "source_trust_score_norm": 0.10,
            "manufacturability_score": 0.15,
        },
        "description": "Interior trim/panels prioritising recycled content and circularity.",
        "subsystem": "Interior / Seating / Foam",
    },
    "Balanced Material Reuse": {
        "weights": {
            "strength_score": 0.20, "weight_saving_score": 0.15,
            "stiffness_score": 0.10, "source_trust_score_norm": 0.10,
            "sustainability_score": 0.15, "manufacturability_score": 0.10,
            "cost_score": 0.10, "supply_risk_score": 0.10,
        },
        "description": "Balanced trade-off across performance, sustainability, and supply chain.",
        "subsystem": "General Material Reuse",
    },
}


def _safe_col(df: pd.DataFrame, col: str, default: float = 50.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def compute_dynamic_threshold(series: pd.Series, mode: str = "High strength structural") -> float:
    """Compute yield strength threshold from data percentile."""
    clean = series.dropna()
    if clean.empty:
        return 250.0
    pct = STRENGTH_MODES.get(mode)
    if pct is None:
        return 250.0
    return round(float(clean.quantile(pct)), 1)


def recommend_materials(
    unified: pd.DataFrame,
    min_strength: float,
    baseline_density: float = 7.85,
    use_case: str = "Lightweight Structural Component",
    min_trust: float = 0,
) -> pd.DataFrame:
    df = unified.copy()

    df["yield_strength_mpa"] = pd.to_numeric(
        df.get("yield_strength_mpa", pd.Series(dtype=float)), errors="coerce"
    )
    df["density_g_cm3"] = pd.to_numeric(
        df.get("density_g_cm3", pd.Series(7.85, index=df.index)), errors="coerce"
    ).fillna(7.85)

    df = df.dropna(subset=["yield_strength_mpa"])

    trust = _safe_col(df, "source_trust_score", 50)
    df = df[trust >= min_trust].copy()
    if df.empty:
        return df

    threshold = max(min_strength, 1.0)

    # Improved strength scoring — prevents all candidates from scoring 100
    def _strength_score(y):
        if y >= threshold:
            return 50 + 50 * min((y - threshold) / threshold, 1.0)
        return max(0, 50 * y / threshold)
    df["strength_score"] = df["yield_strength_mpa"].apply(_strength_score).round(1)

    # Density variance detection
    density_std = df["density_g_cm3"].std()
    low_density_variance = density_std < 0.1
    df["_density_variance_flag"] = low_density_variance

    if low_density_variance:
        df["weight_saving_percent"] = 0.0
        df["weight_saving_score"] = 50.0
    else:
        df["weight_saving_percent"] = (
            (baseline_density - df["density_g_cm3"]) / baseline_density * 100
        ).round(1)
        df["weight_saving_score"] = df["weight_saving_percent"].apply(lambda x: clamp(x * 2)).round(1)

    ym = _safe_col(df, "youngs_modulus_gpa", 50)
    df["stiffness_score"] = (ym / 250 * 100).clip(0, 100).round(1)

    # Renamed to avoid collision with confidence.py
    df["source_trust_score_norm"] = _safe_col(df, "source_trust_score", 70).round(1)

    recyclability = _safe_col(df, "recyclability_score", 50)
    recycled = _safe_col(df, "recycled_content_percent", 30)
    co2 = _safe_col(df, "co2_index", 50)
    df["sustainability_score"] = (0.4 * recyclability + 0.3 * recycled + 0.3 * (100 - co2)).round(1)
    df["manufacturability_score"] = 50.0
    df["cost_score"] = (100 - _safe_col(df, "cost_index", 50)).round(1)
    df["supply_risk_score"] = (100 - _safe_col(df, "supplier_risk_score", 50)).round(1)
    df["corrosion_score"] = _safe_col(df, "corrosion_resistance_score", 50).round(1)

    preset = USE_CASE_PRESETS.get(use_case, USE_CASE_PRESETS["Lightweight Structural Component"])
    weights = preset["weights"]

    df["suitability_score"] = 0.0
    for metric, weight in weights.items():
        if metric in df.columns:
            df["suitability_score"] += weight * df[metric]
        else:
            df["suitability_score"] += weight * 50
    df["suitability_score"] = df["suitability_score"].round(1)

    df["reason_codes"] = df.apply(lambda r: _reason_codes(r, threshold, low_density_variance), axis=1)

    return df.sort_values("suitability_score", ascending=False).reset_index(drop=True)


def _reason_codes(row: pd.Series, threshold: float, low_density_variance: bool) -> str:
    reasons: list[str] = []

    if row.get("strength_score", 0) >= 50:
        reasons.append("+ Meets minimum yield strength")
    else:
        reasons.append("- Below required strength")

    if low_density_variance:
        reasons.append(
            "~ Weight saving not differentiating in current source; "
            "upload lightweight candidate evidence or computed reference data"
        )
    else:
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
        reasons.append("+ High source trust")

    missing_fields = []
    for field, label in [
        ("fatigue_strength_mpa", "Fatigue"), ("corrosion_resistance_score", "Corrosion"),
        ("supplier_risk_score", "Supplier risk"), ("youngs_modulus_gpa", "Stiffness"),
        ("hardness_hv", "Hardness"),
    ]:
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing_fields.append(label)

    if missing_fields:
        reasons.append(f"- Validation needed: {', '.join(missing_fields)}")

    return " | ".join(reasons)
