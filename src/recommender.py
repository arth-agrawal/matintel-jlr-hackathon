"""Subsystem-aware material recommender with evidence-backed scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.subsystem_profiles import SUBSYSTEM_PROFILES
from src.evidence_coverage import filter_subsystem_rows, field_has_data, field_has_variance
from src.data_assumptions import is_assumption_field_series


def clamp(x: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, float(x)))


STRENGTH_MODES = {
    "Moderate structural": 0.50,
    "High strength structural": 0.75,
    "Extreme strength structural": 0.90,
    "Custom": None,
}

USE_CASE_PRESETS: dict[str, dict] = {
    "Lightweight structural replacement": {
        "subsystem": "Structural / Chassis",
        "description": "Find lighter structural options with strong predicted yield strength.",
        "active_properties": ["yield_strength_mpa", "density_g_cm3", "youngs_modulus_gpa"],
    },
    "Battery enclosure screening": {
        "subsystem": "Battery Enclosure / Underbody",
        "description": "Screen enclosure materials using formation energy and structural reference data.",
        "active_properties": ["formation_energy_per_atom", "density_g_cm3", "corrosion_resistance_score"],
    },
    "Electronics thermal interface": {
        "subsystem": "Electronics / Thermal Interface",
        "description": "Rank materials for thermal and electrical interface performance.",
        "active_properties": ["band_gap_ev", "thermal_conductivity_w_mk", "electrical_insulation_score"],
    },
    "Circular interior material": {
        "subsystem": "Interior / Seating / Foam",
        "description": "Prioritise interior materials with circularity and comfort evidence.",
        "active_properties": ["recycled_content_percent", "comfort_score", "voc_emission_score"],
    },
    "Thermal / coating material": {
        "subsystem": "Coatings / Corrosion Protection",
        "description": "Screen coatings and thermal materials for corrosion and surface performance.",
        "active_properties": ["corrosion_resistance_score", "thermal_conductivity_w_mk", "adhesion_score"],
    },
    "General material reuse": {
        "subsystem": "General Material Reuse",
        "description": "Balanced reuse screening across the full material universe.",
        "active_properties": ["density_g_cm3", "recyclability_score", "source_trust_score"],
    },
}

SUBSYSTEM_SCORE_CONFIG: dict[str, dict[str, dict]] = {
    "Structural / Chassis": {
        "strength_score": {"fields": ["yield_strength_mpa"], "weight": 0.28},
        "weight_saving_score": {"fields": ["density_g_cm3"], "weight": 0.22},
        "stiffness_score": {"fields": ["youngs_modulus_gpa"], "weight": 0.15},
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.10},
        "sustainability_score": {"fields": ["recyclability_score"], "weight": 0.10},
        "cost_score": {"fields": ["cost_index"], "weight": 0.05},
        "supply_risk_score": {"fields": ["supplier_risk_score"], "weight": 0.05},
        "corrosion_score": {"fields": ["corrosion_resistance_score"], "weight": 0.05},
    },
    "Battery Enclosure / Underbody": {
        "weight_saving_score": {"fields": ["density_g_cm3"], "weight": 0.18},
        "thermal_score": {"fields": ["thermal_conductivity_w_mk"], "weight": 0.18},
        "flame_score": {"fields": ["flame_retardancy_score"], "weight": 0.12},
        "impact_score": {"fields": ["impact_resistance_score"], "weight": 0.12},
        "corrosion_score": {"fields": ["corrosion_resistance_score"], "weight": 0.12},
        "electrical_score": {"fields": ["electrical_insulation_score"], "weight": 0.08},
        "traceability_score_norm": {"fields": ["traceability_score"], "weight": 0.08},
        "critical_material_score": {"fields": ["critical_material_risk_score"], "weight": 0.07},
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.05},
    },
    "Electronics / Thermal Interface": {
        "thermal_score": {"fields": ["thermal_conductivity_w_mk"], "weight": 0.28},
        "electrical_score": {"fields": ["electrical_insulation_score"], "weight": 0.22},
        "flame_score": {"fields": ["flame_retardancy_score"], "weight": 0.12},
        "band_gap_score": {"fields": ["band_gap_ev"], "weight": 0.15},
        "weight_saving_score": {"fields": ["density_g_cm3"], "weight": 0.10},
        "supply_risk_score": {"fields": ["supplier_risk_score"], "weight": 0.08},
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.05},
    },
    "Interior / Seating / Foam": {
        "weight_saving_score": {"fields": ["density_g_cm3"], "weight": 0.10},
        "comfort_score": {"fields": ["comfort_score"], "weight": 0.18},
        "durability_score": {"fields": ["durability_score"], "weight": 0.18},
        "voc_score": {"fields": ["voc_emission_score"], "weight": 0.15},
        "fire_score": {"fields": ["fire_safety_score"], "weight": 0.15},
        "sustainability_score": {"fields": ["recycled_content_percent"], "weight": 0.14},
        "co2_score": {"fields": ["co2_kg_per_kg"], "weight": 0.05},
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.05},
    },
    "Tyres / Elastomers": {
        "wear_score": {"fields": ["wear_resistance_score"], "weight": 0.25},
        "rolling_score": {"fields": ["rolling_resistance_score"], "weight": 0.22},
        "grip_score_dim": {"fields": ["grip_score"], "weight": 0.20},
        "sustainability_score": {"fields": ["recycled_content_percent"], "weight": 0.18},
        "traceability_score_norm": {"fields": ["traceability_score"], "weight": 0.10},
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.05},
    },
    "Thermal Fluids / Coolants": {
        "thermal_score": {"fields": ["thermal_conductivity_w_mk"], "weight": 0.25},
        "specific_heat_score": {"fields": ["specific_heat"], "weight": 0.18},
        "viscosity_score": {"fields": ["viscosity"], "weight": 0.12},
        "stability_score": {"fields": ["freezing_point"], "weight": 0.10},
        "corrosion_inhibition_score_dim": {"fields": ["corrosion_inhibition_score"], "weight": 0.15},
        "toxicity_score_dim": {"fields": ["toxicity_score"], "weight": 0.12},
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.08},
    },
    "Coatings / Corrosion Protection": {
        "corrosion_score": {"fields": ["corrosion_resistance_score"], "weight": 0.28},
        "salt_spray_score": {"fields": ["salt_spray_hours"], "weight": 0.22},
        "adhesion_score_dim": {"fields": ["adhesion_score"], "weight": 0.18},
        "scratch_score": {"fields": ["scratch_resistance_score"], "weight": 0.17},
        "voc_score": {"fields": ["voc_emission_score"], "weight": 0.10},
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.05},
    },
    "General Material Reuse": {
        "source_trust_score_norm": {"fields": ["source_trust_score"], "weight": 0.20},
        "weight_saving_score": {"fields": ["density_g_cm3"], "weight": 0.15},
        "strength_score": {"fields": ["yield_strength_mpa"], "weight": 0.15},
        "sustainability_score": {"fields": ["recyclability_score"], "weight": 0.20},
        "co2_score": {"fields": ["co2_kg_per_kg"], "weight": 0.15},
        "reference_score": {"fields": ["formation_energy_per_atom"], "weight": 0.15},
    },
}

# Maps score dimension -> (primary field, invert, flat_neutral_msg)
_DIM_FIELD_MAP: dict[str, tuple[str, bool, str | None]] = {
    "thermal_score": ("thermal_conductivity_w_mk", False, "Add thermal property data to score this dimension"),
    "electrical_score": ("electrical_insulation_score", False, None),
    "flame_score": ("flame_retardancy_score", False, None),
    "impact_score": ("impact_resistance_score", False, None),
    "corrosion_score": ("corrosion_resistance_score", False, None),
    "comfort_score": ("comfort_score", False, None),
    "durability_score": ("durability_score", False, None),
    "voc_score": ("voc_emission_score", False, None),
    "fire_score": ("fire_safety_score", False, None),
    "wear_score": ("wear_resistance_score", False, None),
    "rolling_score": ("rolling_resistance_score", False, None),
    "grip_score_dim": ("grip_score", False, None),
    "specific_heat_score": ("specific_heat", False, None),
    "viscosity_score": ("viscosity", False, None),
    "corrosion_inhibition_score_dim": ("corrosion_inhibition_score", False, None),
    "toxicity_score_dim": ("toxicity_score", False, None),
    "salt_spray_score": ("salt_spray_hours", False, None),
    "adhesion_score_dim": ("adhesion_score", False, None),
    "scratch_score": ("scratch_resistance_score", False, None),
    "traceability_score_norm": ("traceability_score", False, None),
    "critical_material_score": ("critical_material_risk_score", True, None),
    "band_gap_score": ("band_gap_ev", False, "Add band-gap data to score this dimension"),
    "stability_score": ("freezing_point", False, None),
    "cost_score": ("cost_index", True, None),
    "supply_risk_score": ("supplier_risk_score", True, None),
}


def compute_dynamic_threshold(series: pd.Series, mode: str = "High strength structural") -> float:
    clean = series.dropna()
    if clean.empty:
        return 250.0
    pct = STRENGTH_MODES.get(mode)
    if pct is None:
        return 250.0
    return round(float(clean.quantile(pct)), 1)


def _numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _score_from_field(df: pd.DataFrame, field: str, invert: bool = False) -> pd.Series | None:
    if not field_has_data(df, field):
        return None
    if not field_has_variance(df, field):
        return None
    vals = _numeric_col(df, field)
    assump = (
        is_assumption_field_series(df, field)
        if "assumption_fields" in df.columns
        else pd.Series(False, index=df.index)
    )
    measured = vals[~assump]
    if measured.dropna().empty:
        return None
    if invert:
        scored = (100 - vals).clip(0, 100).round(1)
    else:
        mx = float(measured.max())
        if mx <= 0:
            return None
        scored = (vals / mx * 100).clip(0, 100).round(1)
    scored.loc[assump] = 50.0
    return scored.round(1)


def recommend_materials(
    unified: pd.DataFrame,
    subsystem: str = "Structural / Chassis",
    min_strength: float | None = None,
    baseline_density: float = 7.85,
    use_case: str = "Lightweight structural replacement",
    min_trust: float = 0,
) -> pd.DataFrame:
    preset_sub = USE_CASE_PRESETS.get(use_case, {}).get("subsystem", subsystem)
    subsystem = subsystem or preset_sub

    df = filter_subsystem_rows(unified, subsystem)
    if df.empty:
        return df

    trust = _numeric_col(df, "source_trust_score").fillna(0)
    df = df[trust >= min_trust].copy()
    if df.empty:
        return df

    score_config = SUBSYSTEM_SCORE_CONFIG.get(subsystem, SUBSYSTEM_SCORE_CONFIG["Structural / Chassis"])
    global_reasons: list[str] = []
    used_dims: list[str] = []
    neutral_dims: list[str] = []

    threshold = max(min_strength or 0, 1.0)

    # Strength (structural / general when evidence exists)
    if "strength_score" in score_config:
        if field_has_data(df, "yield_strength_mpa") and field_has_variance(df, "yield_strength_mpa"):
            if min_strength and min_strength > 0 and subsystem == "Structural / Chassis":
                df = df[_numeric_col(df, "yield_strength_mpa").fillna(0) >= min_strength].copy()
            global_reasons.append("~ Strength differentiated using dynamic percentile target")
            ys = _numeric_col(df, "yield_strength_mpa")

            def _strength_score(y):
                if pd.isna(y):
                    return 50.0
                if y >= threshold:
                    return 50 + 50 * min((y - threshold) / threshold, 1.0)
                return max(0, 50 * y / threshold)

            df["strength_score"] = ys.apply(_strength_score).round(1)
            used_dims.append("strength_score")
        else:
            df["strength_score"] = 50.0
            neutral_dims.append("strength_score")

    # Weight saving
    low_density_variance = not field_has_variance(df, "density_g_cm3") if field_has_data(df, "density_g_cm3") else True
    df["_density_variance_flag"] = low_density_variance

    if "weight_saving_score" in score_config:
        if field_has_data(df, "density_g_cm3") and not low_density_variance:
            density = _numeric_col(df, "density_g_cm3")
            assump_density = (
                is_assumption_field_series(df, "density_g_cm3")
                if "assumption_fields" in df.columns
                else pd.Series(False, index=df.index)
            )
            df["weight_saving_percent"] = ((baseline_density - density) / baseline_density * 100).round(1)
            df["weight_saving_percent"] = df["weight_saving_percent"].where(~assump_density, 0.0)
            df["weight_saving_score"] = df["weight_saving_percent"].apply(lambda x: clamp(x * 2)).round(1)
            df.loc[assump_density, "weight_saving_score"] = 50.0
            used_dims.append("weight_saving_score")
            if assump_density.any():
                global_reasons.append(
                    "~ Weight saving neutral for rows with engineering-default density (not measured)"
                )
        else:
            df["weight_saving_percent"] = 0.0
            df["weight_saving_score"] = 50.0
            neutral_dims.append("weight_saving_score")
            global_reasons.append(
                "~ Weight saving neutral: flat density in current evidence; "
                "upload lightweight candidate evidence for aluminium/magnesium/composite comparison"
            )

    # Stiffness
    if "stiffness_score" in score_config:
        scored = _score_from_field(df, "youngs_modulus_gpa")
        if scored is not None:
            df["stiffness_score"] = scored
            used_dims.append("stiffness_score")
        else:
            df["stiffness_score"] = 50.0
            neutral_dims.append("stiffness_score")

    # Source trust
    if "source_trust_score_norm" in score_config:
        if field_has_data(df, "source_trust_score"):
            df["source_trust_score_norm"] = _numeric_col(df, "source_trust_score").clip(0, 100).round(1)
            used_dims.append("source_trust_score_norm")
        else:
            df["source_trust_score_norm"] = 50.0
            neutral_dims.append("source_trust_score_norm")

    # Generic field-backed dimensions
    for dim, (field, invert, msg) in _DIM_FIELD_MAP.items():
        if dim not in score_config:
            continue
        scored = _score_from_field(df, field, invert=invert)
        if scored is not None:
            df[dim] = scored
            used_dims.append(dim)
            if dim == "band_gap_score":
                global_reasons.append("~ Computed reference only: not experimental validation")
        else:
            df[dim] = 50.0
            neutral_dims.append(dim)
            if msg:
                global_reasons.append(f"~ {msg}")

    # Sustainability composite
    if "sustainability_score" in score_config:
        has_sust = (
            field_has_data(df, "recyclability_score")
            or field_has_data(df, "recycled_content_percent")
            or field_has_data(df, "bio_based_content_percent")
        )
        if has_sust and (
            field_has_variance(df, "recyclability_score")
            or field_has_variance(df, "recycled_content_percent")
            or field_has_variance(df, "bio_based_content_percent")
        ):
            recyclability = _numeric_col(df, "recyclability_score").fillna(50)
            recycled = _numeric_col(df, "recycled_content_percent").fillna(30)
            co2 = _numeric_col(df, "co2_index").fillna(50)
            bio = _numeric_col(df, "bio_based_content_percent").fillna(0)
            if subsystem == "Tyres / Elastomers":
                df["sustainability_score"] = (0.5 * bio + 0.5 * recycled).clip(0, 100).round(1)
            else:
                df["sustainability_score"] = (
                    0.4 * recyclability + 0.3 * recycled + 0.3 * (100 - co2)
                ).clip(0, 100).round(1)
            used_dims.append("sustainability_score")
        else:
            df["sustainability_score"] = 50.0
            neutral_dims.append("sustainability_score")

    # CO2
    if "co2_score" in score_config:
        scored = _score_from_field(df, "co2_kg_per_kg", invert=True)
        if scored is not None:
            df["co2_score"] = scored
            used_dims.append("co2_score")
        else:
            df["co2_score"] = 50.0
            neutral_dims.append("co2_score")

    # Reference (computed)
    if "reference_score" in score_config:
        if field_has_data(df, "formation_energy_per_atom") or field_has_data(df, "band_gap_ev"):
            df["reference_score"] = 60.0
            used_dims.append("reference_score")
            global_reasons.append("~ Computed reference only: not experimental validation")
        else:
            df["reference_score"] = 50.0
            neutral_dims.append("reference_score")

    # Weighted suitability
    total_weight = sum(cfg["weight"] for cfg in score_config.values())
    df["suitability_score"] = 0.0
    for dim, cfg in score_config.items():
        w = cfg["weight"] / total_weight if total_weight else cfg["weight"]
        val = df.get(dim, pd.Series(50.0, index=df.index))
        df["suitability_score"] += w * val
    df["suitability_score"] = df["suitability_score"].round(1)

    df["_score_dims_used"] = ", ".join(used_dims)
    df["_score_dims_neutral"] = ", ".join(neutral_dims)

    df["reason_codes"] = df.apply(
        lambda r: _reason_codes(r, subsystem, global_reasons),
        axis=1,
    )
    return df.sort_values("suitability_score", ascending=False).reset_index(drop=True)


def _reason_codes(row: pd.Series, subsystem: str, global_reasons: list[str]) -> str:
    reasons = list(dict.fromkeys(global_reasons))

    source = str(row.get("source_type", "")).lower()
    basis = str(row.get("recommendation_basis", "")).lower()
    if source == "computed_database" or "computed" in basis:
        reasons.append("~ Stable reference profile from public database")
        reasons.append("~ Property available for screening")
    elif source == "public_benchmark" or "benchmark" in basis:
        reasons.append("~ Stable reference profile from benchmark data")
    elif source == "supplier_sheet":
        reasons.append("~ Uploaded supplier data — confirm in validation report")
    elif source in ("public_experimental", "experimental_test") and subsystem == "Structural / Chassis":
        if row.get("strength_score", 50) >= 50:
            reasons.append("+ Strong predicted property")

    if row.get("source_trust_score", 0) >= 90:
        reasons.append("+ High data trust")

    if row.get("weight_saving_score", 50) > 55:
        reasons.append("+ Favourable density vs baseline")

    profile = SUBSYSTEM_PROFILES.get(subsystem, {})
    missing_validation = []
    for field in profile.get("important_fields", []):
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing_validation.append(field.replace("_", " "))

    if missing_validation:
        shown = ", ".join(missing_validation[:4])
        if len(missing_validation) > 4:
            shown += "…"
        reasons.append(f"- Missing validation test: {shown}")

    return " | ".join(dict.fromkeys(reasons))


def get_scoring_dimensions(subsystem: str) -> list[tuple[str, str, float]]:
    cfg = SUBSYSTEM_SCORE_CONFIG.get(subsystem, {})
    total = sum(v["weight"] for v in cfg.values()) or 1
    labels = {
        "strength_score": "Strength", "weight_saving_score": "Weight Saving",
        "stiffness_score": "Stiffness", "source_trust_score_norm": "Source Trust",
        "sustainability_score": "Sustainability", "cost_score": "Cost",
        "supply_risk_score": "Supply Risk", "corrosion_score": "Corrosion",
        "thermal_score": "Thermal", "electrical_score": "Electrical",
        "flame_score": "Flame Retardancy", "impact_score": "Impact",
        "band_gap_score": "Band Gap", "comfort_score": "Comfort",
        "durability_score": "Durability", "voc_score": "VOC", "fire_score": "Fire Safety",
        "wear_score": "Wear", "rolling_score": "Rolling Resistance",
        "grip_score_dim": "Grip", "specific_heat_score": "Specific Heat",
        "viscosity_score": "Viscosity", "co2_score": "CO₂",
        "reference_score": "Reference (computed)", "traceability_score_norm": "Traceability",
        "critical_material_score": "Critical Material Risk",
        "corrosion_inhibition_score_dim": "Corrosion Inhibition",
        "toxicity_score_dim": "Toxicity", "salt_spray_score": "Salt Spray",
        "adhesion_score_dim": "Adhesion", "scratch_score": "Scratch Resistance",
        "stability_score": "Stability",
    }
    return [(k, labels.get(k, k.replace("_", " ").title()), v["weight"] / total) for k, v in cfg.items()]
