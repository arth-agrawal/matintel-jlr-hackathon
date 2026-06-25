"""Evidence coverage, property matrices, and subsystem-aware UI helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.subsystem_profiles import SUBSYSTEM_PROFILES, ALL_SUBSYSTEMS, get_subsystem_readiness
from src.model_registry import MODEL_SPECS, detect_trainable_targets
from src.data_assumptions import is_assumption_field_series


# ── Per-subsystem filter fields (numeric min sliders when data exists) ────────

SUBSYSTEM_FILTER_FIELDS: dict[str, list[str]] = {
    "Structural / Chassis": [
        "yield_strength_mpa", "ultimate_tensile_strength_mpa", "density_g_cm3",
        "youngs_modulus_gpa", "source_trust_score",
    ],
    "Battery Enclosure / Underbody": [
        "density_g_cm3", "thermal_conductivity_w_mk", "flame_retardancy_score",
        "impact_resistance_score", "corrosion_resistance_score", "electrical_insulation_score",
        "traceability_score", "critical_material_risk_score", "source_trust_score",
    ],
    "Electronics / Thermal Interface": [
        "thermal_conductivity_w_mk", "electrical_insulation_score", "flame_retardancy_score",
        "band_gap_ev", "density_g_cm3", "supplier_risk_score", "source_trust_score",
    ],
    "Interior / Seating / Foam": [
        "density_g_cm3", "comfort_score", "durability_score", "voc_emission_score",
        "fire_safety_score", "recycled_content_percent", "co2_kg_per_kg", "source_trust_score",
    ],
    "Tyres / Elastomers": [
        "wear_resistance_score", "rolling_resistance_score", "grip_score",
        "bio_based_content_percent", "recycled_content_percent", "traceability_score",
    ],
    "Thermal Fluids / Coolants": [
        "thermal_conductivity_w_mk", "specific_heat", "viscosity", "freezing_point",
        "boiling_point", "corrosion_inhibition_score", "toxicity_score",
    ],
    "Coatings / Corrosion Protection": [
        "corrosion_resistance_score", "salt_spray_hours", "adhesion_score",
        "scratch_resistance_score", "voc_emission_score",
    ],
    "General Material Reuse": [
        "source_trust_score", "density_g_cm3", "recyclability_score",
        "recycled_content_percent", "co2_kg_per_kg",
    ],
}

SUBSYSTEM_DISPLAY_COLUMNS: dict[str, list[str]] = {
    "Structural / Chassis": [
        "material_name", "material_family", "source_dataset", "source_type",
        "yield_strength_mpa", "ultimate_tensile_strength_mpa", "density_g_cm3",
        "youngs_modulus_gpa", "source_trust_score", "readiness_status",
    ],
    "Battery Enclosure / Underbody": [
        "material_name", "source_dataset", "source_type", "density_g_cm3",
        "thermal_conductivity_w_mk", "flame_retardancy_score", "impact_resistance_score",
        "corrosion_resistance_score", "traceability_score", "readiness_status",
    ],
    "Electronics / Thermal Interface": [
        "material_name", "formula", "source_dataset", "source_type",
        "thermal_conductivity_w_mk", "electrical_insulation_score", "band_gap_ev",
        "density_g_cm3", "source_trust_score", "readiness_status",
    ],
    "Interior / Seating / Foam": [
        "material_name", "source_dataset", "source_type", "density_g_cm3",
        "comfort_score", "durability_score", "voc_emission_score", "fire_safety_score",
        "recycled_content_percent", "co2_kg_per_kg", "readiness_status",
    ],
    "Tyres / Elastomers": [
        "material_name", "source_dataset", "source_type", "wear_resistance_score",
        "rolling_resistance_score", "grip_score", "recycled_content_percent",
        "traceability_score", "readiness_status",
    ],
    "Thermal Fluids / Coolants": [
        "material_name", "source_dataset", "source_type", "thermal_conductivity_w_mk",
        "specific_heat", "viscosity", "freezing_point", "boiling_point",
        "corrosion_inhibition_score", "readiness_status",
    ],
    "Coatings / Corrosion Protection": [
        "material_name", "source_dataset", "source_type", "corrosion_resistance_score",
        "salt_spray_hours", "adhesion_score", "scratch_resistance_score",
        "voc_emission_score", "readiness_status",
    ],
    "General Material Reuse": [
        "material_name", "source_dataset", "source_type", "density_g_cm3",
        "yield_strength_mpa", "band_gap_ev", "formation_energy_per_atom",
        "recyclability_score", "recycled_content_percent", "source_trust_score",
        "readiness_status",
    ],
}

# Property-based subsystem relevance — tightened inference (not OR-of-single-fields)
COMPUTED_SOURCE_TYPES = {"computed_database", "public_benchmark"}
EXPERIMENTAL_SOURCE_TYPES = {"public_experimental", "experimental_test"}
# Subsystem tags on computed/benchmark rows are routing hints — not engineer assignments.
EXPLICIT_TAG_SOURCE_TYPES = EXPERIMENTAL_SOURCE_TYPES | {
    "supplier_sheet", "sustainability_sheet", "procurement_sheet", "unknown",
}


def _explicit_subsystem_tag_mask(df: pd.DataFrame, subsystem: str) -> pd.Series:
    """Rows explicitly assigned to a subsystem (experimental uploads / engineer review)."""
    if df.empty or "application_subsystem" not in df.columns:
        return pd.Series(False, index=df.index)
    match = df["application_subsystem"] == subsystem
    reviewed = (
        df["engineer_reviewed"].astype(str).str.lower().isin(["true", "1"])
        if "engineer_reviewed" in df.columns
        else pd.Series(False, index=df.index)
    )
    if "source_type" in df.columns:
        explicit = df["source_type"].isin(EXPLICIT_TAG_SOURCE_TYPES)
        return match & (reviewed | explicit)
    return match & reviewed


def _measured_mask(df: pd.DataFrame, field: str) -> pd.Series:
    if field not in df.columns:
        return pd.Series(False, index=df.index)
    mask = df[field].notna()
    if "assumption_fields" in df.columns:
        mask = mask & ~is_assumption_field_series(df, field)
    return mask


def _any_measured(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for field in fields:
        mask = mask | _measured_mask(df, field)
    return mask


def _tagged_mask(df: pd.DataFrame, subsystem: str) -> pd.Series:
    return _explicit_subsystem_tag_mask(df, subsystem)


def _inferred_mask(df: pd.DataFrame, subsystem: str) -> pd.Series:
    """Property-aware inference — excludes rows matched only by weak signals."""
    if df.empty:
        return pd.Series(False, index=df.index)

    if subsystem == "General Material Reuse":
        return pd.Series(True, index=df.index)

    if subsystem == "Structural / Chassis":
        return _any_measured(df, [
            "yield_strength_mpa", "ultimate_tensile_strength_mpa",
            "bulk_modulus_gpa", "shear_modulus_gpa",
        ])

    if subsystem == "Electronics / Thermal Interface":
        m = _any_measured(df, [
            "band_gap_ev", "dielectric_constant",
            "electrical_insulation_score", "thermal_conductivity_w_mk",
        ])
        if "source_type" in df.columns:
            m = m & df["source_type"].isin(COMPUTED_SOURCE_TYPES | {"public_reference"})
        return m

    if subsystem == "Battery Enclosure / Underbody":
        m = _any_measured(df, [
            "formation_energy_per_atom", "band_gap_ev", "flame_retardancy_score",
            "impact_resistance_score", "corrosion_resistance_score", "electrical_insulation_score",
        ])
        if "source_type" in df.columns:
            m = m & df["source_type"].isin(COMPUTED_SOURCE_TYPES)
        return m

    if subsystem == "Thermal Fluids / Coolants":
        thermal = _measured_mask(df, "thermal_conductivity_w_mk")
        secondary = _any_measured(df, [
            "specific_heat", "viscosity", "boiling_point", "freezing_point", "corrosion_inhibition_score",
        ])
        return thermal & secondary

    if subsystem == "Interior / Seating / Foam":
        return _any_measured(df, [
            "comfort_score", "durability_score", "voc_emission_score",
            "fire_safety_score", "recycled_content_percent", "co2_kg_per_kg",
        ])

    if subsystem == "Tyres / Elastomers":
        return _any_measured(df, [
            "wear_resistance_score", "rolling_resistance_score", "grip_score", "bio_based_content_percent",
        ])

    if subsystem == "Coatings / Corrosion Protection":
        return _any_measured(df, [
            "corrosion_resistance_score", "salt_spray_hours", "adhesion_score",
            "scratch_resistance_score", "voc_emission_score",
        ])

    return pd.Series(False, index=df.index)


def filter_tagged_rows(df: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    if subsystem == "General Material Reuse":
        return df.copy()
    return df[_tagged_mask(df, subsystem)].copy()


def filter_inferred_rows(df: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """Rows matching property inference (computed/reference), excluding explicit tags."""
    if subsystem == "General Material Reuse":
        return df.copy()
    inferred = _inferred_mask(df, subsystem)
    if "source_type" in df.columns:
        ref_rows = df["source_type"].isin(COMPUTED_SOURCE_TYPES | {"public_reference"})
        inferred = inferred & ref_rows
    tagged = _tagged_mask(df, subsystem)
    return df[inferred & ~tagged].copy()


def count_training_rows(df: pd.DataFrame, subsystem: str) -> int:
    """Rows eligible for experimental ML training in this subsystem context."""
    if df.empty:
        return 0
    sub = filter_subsystem_rows(df, subsystem)
    if "used_for_ml_training" not in sub.columns:
        return 0
    ml = sub["used_for_ml_training"].astype(str).str.lower().isin(["true", "1"])
    exp = sub["source_type"].isin(EXPERIMENTAL_SOURCE_TYPES) if "source_type" in sub.columns else ml
    return int((ml & exp).sum())


def subsystem_coverage_counts(df: pd.DataFrame, subsystem: str) -> dict[str, int]:
    """Honest tagged / inferred / training counts for cockpit cards."""
    if subsystem == "General Material Reuse":
        return {
            "tagged_rows": len(df),
            "inferred_rows": 0,
            "training_rows": count_training_rows(df, subsystem),
            "relevant_rows": len(df),
        }
    tagged = int(_tagged_mask(df, subsystem).sum())
    inferred_only = len(filter_inferred_rows(df, subsystem))
    relevant = len(filter_subsystem_rows(df, subsystem))
    return {
        "tagged_rows": tagged,
        "inferred_rows": inferred_only,
        "training_rows": count_training_rows(df, subsystem),
        "relevant_rows": relevant,
    }


# Legacy field list for property matrix columns
SUBSYSTEM_PROPERTY_MATCH: dict[str, list[str]] = {
    "Structural / Chassis": [
        "yield_strength_mpa", "ultimate_tensile_strength_mpa", "bulk_modulus_gpa", "shear_modulus_gpa",
    ],
    "Battery Enclosure / Underbody": [
        "formation_energy_per_atom", "band_gap_ev", "flame_retardancy_score",
        "impact_resistance_score", "corrosion_resistance_score", "electrical_insulation_score",
    ],
    "Electronics / Thermal Interface": [
        "band_gap_ev", "dielectric_constant", "electrical_insulation_score", "thermal_conductivity_w_mk",
    ],
    "Interior / Seating / Foam": [
        "voc_emission_score", "fire_safety_score", "comfort_score", "durability_score",
        "recycled_content_percent", "co2_kg_per_kg",
    ],
    "Tyres / Elastomers": [
        "wear_resistance_score", "rolling_resistance_score", "grip_score", "bio_based_content_percent",
    ],
    "Thermal Fluids / Coolants": [
        "thermal_conductivity_w_mk", "specific_heat", "viscosity", "freezing_point", "boiling_point",
    ],
    "Coatings / Corrosion Protection": [
        "corrosion_resistance_score", "salt_spray_hours", "adhesion_score", "scratch_resistance_score",
    ],
    "General Material Reuse": [],
}

SUBSYSTEM_TRUSTED_SOURCES: dict[str, list[str]] = {
    "Structural / Chassis": ["matminer_steel_strength", "jarvis_dft_3d", "matminer_matbench_extra", "engineer_reviewed_upload"],
    "Battery Enclosure / Underbody": ["jarvis_dft_3d", "matminer_matbench_extra", "engineer_reviewed_upload"],
    "Electronics / Thermal Interface": ["jarvis_dft_3d", "matminer_matbench_extra", "engineer_reviewed_upload"],
    "Interior / Seating / Foam": ["nasa_tpsx_reference", "engineer_reviewed_upload"],
    "Tyres / Elastomers": ["engineer_reviewed_upload"],
    "Thermal Fluids / Coolants": ["nasa_tpsx_reference", "engineer_reviewed_upload"],
    "Coatings / Corrosion Protection": ["nasa_tpsx_reference", "jarvis_dft_3d", "engineer_reviewed_upload"],
    "General Material Reuse": ["matminer_steel_strength", "jarvis_dft_3d", "matminer_matbench_extra", "engineer_reviewed_upload"],
}

SOURCE_LABELS: dict[str, str] = {
    "matminer_steel_strength": "Matminer steel_strength (experimental steel)",
    "jarvis_dft_3d": "JARVIS-DFT dft_3d (computed reference)",
    "jarvis_dft_3d_sample": "JARVIS-DFT dft_3d (computed reference)",
    "matminer_matbench_extra": "Matminer / Matbench property benchmarks",
    "matbench_dielectric": "Matbench dielectric",
    "matbench_mp_gap": "Matbench MP band gap",
    "matbench_mp_e_form": "Matbench formation energy",
    "nasa_tpsx_reference": "NASA TPSX reference/upload pathway",
    "engineer_reviewed_upload": "Engineer-reviewed evidence upload",
    "engineer_upload": "Engineer-reviewed evidence upload",
}

SUBSYSTEM_CHART_FIELD: dict[str, str] = {
    "Structural / Chassis": "yield_strength_mpa",
    "Battery Enclosure / Underbody": "density_g_cm3",
    "Electronics / Thermal Interface": "band_gap_ev",
    "Interior / Seating / Foam": "recycled_content_percent",
    "Tyres / Elastomers": "wear_resistance_score",
    "Thermal Fluids / Coolants": "thermal_conductivity_w_mk",
    "Coatings / Corrosion Protection": "corrosion_resistance_score",
    "General Material Reuse": "density_g_cm3",
}


def _isnan(val: Any) -> bool:
    try:
        return isinstance(val, float) and math.isnan(val)
    except (TypeError, ValueError):
        return False


def count_nonempty(df: pd.DataFrame, field: str, exclude_assumptions: bool = True) -> int:
    if field not in df.columns:
        return 0
    mask = df[field].notna()
    if exclude_assumptions and "assumption_fields" in df.columns:
        mask = mask & ~is_assumption_field_series(df, field)
    return int(mask.sum())


def field_has_data(df: pd.DataFrame, field: str, min_rows: int = 1) -> bool:
    return count_nonempty(df, field, exclude_assumptions=True) >= min_rows


def field_has_variance(df: pd.DataFrame, field: str, min_std: float = 0.01) -> bool:
    if field not in df.columns:
        return False
    mask = df[field].notna()
    if "assumption_fields" in df.columns:
        mask = mask & ~is_assumption_field_series(df, field)
    series = pd.to_numeric(df.loc[mask, field], errors="coerce").dropna()
    if len(series) < 2:
        return False
    return float(series.std()) >= min_std


def filter_subsystem_rows(df: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """Return rows relevant to subsystem: tagged OR property-inferred (tight rules)."""
    if df.empty:
        return df.copy()

    if subsystem == "General Material Reuse":
        return df.copy()

    tagged = _tagged_mask(df, subsystem)
    inferred = _inferred_mask(df, subsystem)
    combined = tagged | inferred
    return df[combined].copy()


def apply_subsystem_filters(
    df: pd.DataFrame,
    subsystem: str,
    min_filters: dict[str, float],
) -> pd.DataFrame:
    """Apply numeric min filters for fields that have data."""
    out = df.copy()
    for field, min_val in min_filters.items():
        if field not in out.columns:
            continue
        if not field_has_data(out, field):
            continue
        numeric = pd.to_numeric(out[field], errors="coerce")
        out = out[numeric.fillna(0) >= min_val]
    return out


def add_readiness_column(df: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    out = df.copy()
    statuses = []
    for _, row in out.iterrows():
        r = get_subsystem_readiness(row, subsystem)
        statuses.append(r["readiness"])
    out["readiness_status"] = statuses
    return out


def format_display_table(df: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """Build curated display table with safe fallbacks."""
    cols = SUBSYSTEM_DISPLAY_COLUMNS.get(subsystem, SUBSYSTEM_DISPLAY_COLUMNS["Structural / Chassis"])
    out = add_readiness_column(df, subsystem) if not df.empty else df.copy()
    display = pd.DataFrame()
    for col in cols:
        if col in out.columns:
            display[col] = out[col]
        else:
            display[col] = "—"
    return display


def property_coverage_matrix(unified_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = subsystems, columns = important properties, values = non-null count."""
    all_fields: list[str] = []
    for profile in SUBSYSTEM_PROFILES.values():
        for f in profile["important_fields"]:
            if f not in all_fields:
                all_fields.append(f)

    rows = []
    for subsystem in ALL_SUBSYSTEMS:
        sub_df = filter_subsystem_rows(unified_df, subsystem)
        row: dict[str, Any] = {"subsystem": subsystem}
        for field in all_fields:
            row[field] = count_nonempty(sub_df, field)
        rows.append(row)

    matrix = pd.DataFrame(rows).set_index("subsystem")
    return matrix


def subsystem_evidence_dashboard(
    unified_df: pd.DataFrame,
    registry_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compact dashboard: materials count, field coverage, model status, readiness."""
    if registry_df is None:
        registry_df = detect_trainable_targets(unified_df)

    rows = []
    for subsystem in ALL_SUBSYSTEMS:
        profile = SUBSYSTEM_PROFILES[subsystem]
        sub_df = filter_subsystem_rows(unified_df, subsystem)
        n_materials = len(sub_df)

        available_fields = []
        missing_fields = []
        for field in profile["important_fields"]:
            if field_has_data(sub_df, field):
                available_fields.append(field)
            else:
                missing_fields.append(field)

        # Active / waiting models for this subsystem
        sub_models = registry_df[
            registry_df["subsystem"].astype(str).str.contains(
                subsystem.split("/")[0].strip()[:10], case=False, na=False
            ) | registry_df["subsystem"].astype(str).str.contains(subsystem, case=False, na=False)
        ]
        active = sub_models[sub_models["status"] == "Active"]["model_name"].tolist()
        waiting = sub_models[sub_models["status"] == "Waiting for evidence"]["model_name"].tolist()
        trainable = sub_models[sub_models["status"].isin([
            "Trainable", "Trainable next", "Computed-reference trainable",
        ])]["model_name"].tolist()

        has_experimental = False
        has_computed = False
        if "source_type" in sub_df.columns and len(sub_df) > 0:
            has_experimental = bool(sub_df["source_type"].isin(["public_experimental", "experimental_test"]).any())
            has_computed = bool(sub_df["source_type"].isin(["computed_database", "public_benchmark"]).any())

        if active and has_experimental:
            readiness = "Active experimental coverage"
        elif has_computed and n_materials > 0:
            readiness = "Computed-reference coverage"
        elif n_materials > 0 and len(available_fields) >= len(profile["important_fields"]) * 0.3:
            readiness = "Benchmark/reference coverage"
        elif n_materials > 0:
            readiness = "Partial — validation required"
        else:
            readiness = "Upload evidence needed"

        rows.append({
            "subsystem": subsystem,
            "materials": n_materials,
            "fields_available": len(available_fields),
            "fields_missing": len(missing_fields),
            "missing_fields": ", ".join(missing_fields[:4]) + ("…" if len(missing_fields) > 4 else ""),
            "active_models": ", ".join(active) or "—",
            "waiting_models": ", ".join(waiting) or "—",
            "trainable_models": ", ".join(trainable) or "—",
            "readiness": readiness,
        })

    return pd.DataFrame(rows)


def empty_state_info(subsystem: str, unified_df: pd.DataFrame, registry_df: pd.DataFrame) -> dict:
    """Premium empty-state content for a subsystem with no matching rows."""
    profile = SUBSYSTEM_PROFILES.get(subsystem, {})
    sources = [SOURCE_LABELS.get(s, s) for s in SUBSYSTEM_TRUSTED_SOURCES.get(subsystem, [])]
    expected = profile.get("important_fields", [])
    sub_registry = registry_df[
        registry_df["subsystem"].astype(str).str.contains(subsystem.split("/")[0].strip()[:8], na=False)
        | (registry_df["subsystem"].astype(str).str.contains("General", na=False) & subsystem.startswith("General"))
    ]

    return {
        "subsystem": subsystem,
        "trusted_sources": sources,
        "expected_fields": expected,
        "coverage_note": profile.get("coverage_note", ""),
        "scoring_focus": profile.get("scoring_focus", ""),
        "registry_rows": sub_registry[["model_name", "status", "reason"]].to_dict("records"),
        "activation_hint": _activation_hint(subsystem),
    }


def _activation_hint(subsystem: str) -> str:
    hints = {
        "Electronics / Thermal Interface": (
            "Electronics / Thermal Interface is ready in the schema and model registry, but needs "
            "JARVIS-DFT or reviewed supplier evidence with thermal/electrical fields."
        ),
        "Interior / Seating / Foam": (
            "Interior / Seating / Foam requires uploaded supplier or test evidence with VOC, fire, "
            "comfort, and circularity fields."
        ),
        "Tyres / Elastomers": (
            "Tyres / Elastomers requires uploaded tyre-specific evidence with wear, grip, and "
            "rolling resistance data."
        ),
        "Thermal Fluids / Coolants": (
            "Thermal Fluids / Coolants can be activated via NASA TPSX CSV export or uploaded "
            "coolant/fluid test data with thermal and viscosity properties."
        ),
        "Coatings / Corrosion Protection": (
            "Coatings / Corrosion Protection needs coating-specific supplier or salt-spray test "
            "evidence, or NASA TPSX reference data."
        ),
        "Battery Enclosure / Underbody": (
            "Battery Enclosure / Underbody has partial steel structural coverage. Upload corrosion, "
            "thermal, and flame-retardancy evidence for full screening."
        ),
    }
    return hints.get(
        subsystem,
        f"{subsystem} is configured in the schema but has no tagged materials yet. "
        "Upload engineer-reviewed evidence or load optional public reference datasets.",
    )


def get_available_filters(df: pd.DataFrame, subsystem: str) -> tuple[list[str], list[str]]:
    """Return (fields_with_sliders, fields_missing_evidence)."""
    filter_fields = SUBSYSTEM_FILTER_FIELDS.get(subsystem, [])
    with_data = []
    missing = []
    for field in filter_fields:
        if field_has_data(df, field):
            with_data.append(field)
        else:
            missing.append(field)
    return with_data, missing


def get_model_activation_info(model_name: str, registry_df: pd.DataFrame) -> dict:
    spec = MODEL_SPECS.get(model_name, {})
    row = registry_df[registry_df["model_name"] == model_name]
    status = row["status"].iloc[0] if len(row) else "Unknown"
    reason = row["reason"].iloc[0] if len(row) else ""
    eligible = int(row["eligible_rows"].iloc[0]) if len(row) else 0

    activation = {
        "structural_yield_strength": "Already active — trained on matminer experimental steel.",
        "structural_tensile_strength": "Upload or connect experimental steel data with UTS labels.",
        "computed_formation_energy": "Load JARVIS-DFT or matbench formation energy (Evidence Intake → broader reference set).",
        "computed_band_gap": "Load JARVIS-DFT or matbench band gap datasets.",
        "elastic_modulus_proxy": "Load matbench log_kvrh/log_gvrh or JARVIS modulus fields.",
        "thermal_conductivity": "Export NASA TPSX properties or upload reviewed thermal test CSV with thermal_conductivity_w_mk.",
    }

    return {
        "model_name": model_name,
        "subsystem": spec.get("subsystem", "—"),
        "target": spec.get("target", "—"),
        "required_features": spec.get("required_features", "—"),
        "eligible_source_types": ", ".join(spec.get("eligible_source_types", [])),
        "eligible_rows": eligible,
        "status": status,
        "reason": reason,
        "how_to_activate": activation.get(model_name, "Upload labelled evidence matching model target and source type."),
        "description": spec.get("description", ""),
    }
