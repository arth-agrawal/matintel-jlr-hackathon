"""Model registry for MatIntel — tracks implemented, trainable, and waiting models."""

from __future__ import annotations

import pandas as pd

from src.modeling import MODEL_TRAINING_SPECS, load_all_trained_models, load_model_registry


MODEL_SPECS: dict[str, dict] = {
    "structural_yield_strength": {
        "subsystem": "Structural / Chassis",
        "target": "yield_strength_mpa",
        "required_features": "wt_percent_* composition columns",
        "eligible_source_types": ["public_experimental", "experimental_test"],
        "model_type": "Best selected tabular regressor",
        "model_family": "experimental",
        "status": "implemented",
        "description": "Experimental yield strength from steel composition — active model.",
    },
    "structural_tensile_strength": {
        "subsystem": "Structural / Chassis",
        "target": "ultimate_tensile_strength_mpa",
        "required_features": "wt_percent_* composition columns",
        "eligible_source_types": ["public_experimental", "experimental_test"],
        "model_type": "Tabular regressor",
        "model_family": "experimental",
        "status": "trainable",
        "description": "Tensile strength — trainable next if UTS labels exist.",
    },
    "computed_formation_energy": {
        "subsystem": "General Material Reuse / Battery / Electronics",
        "target": "formation_energy_per_atom",
        "required_features": "formula descriptors + density_g_cm3",
        "eligible_source_types": ["computed_database", "public_benchmark"],
        "model_type": "Tabular regressor",
        "model_family": "computed_reference",
        "status": "implemented",
        "description": "Formation energy from JARVIS / matbench computed reference.",
    },
    "computed_band_gap": {
        "subsystem": "Electronics / Thermal Interface",
        "target": "band_gap_ev",
        "required_features": "formula descriptors + density_g_cm3",
        "eligible_source_types": ["computed_database", "public_benchmark"],
        "model_type": "Tabular regressor",
        "model_family": "computed_reference",
        "status": "implemented",
        "description": "Band gap from JARVIS / matbench reference data.",
    },
    "elastic_modulus_proxy": {
        "subsystem": "Structural reference / General Material Reuse",
        "target": "bulk_modulus_gpa",
        "required_features": "formula descriptors + density_g_cm3",
        "eligible_source_types": ["computed_database", "public_benchmark"],
        "model_type": "Tabular regressor",
        "model_family": "computed_reference",
        "status": "implemented",
        "description": "Bulk/shear modulus proxy from matbench/JARVIS reference.",
        "alt_target": "shear_modulus_gpa",
    },
    "thermal_conductivity": {
        "subsystem": "Thermal Fluids / Coatings / Electronics",
        "target": "thermal_conductivity_w_mk",
        "required_features": "composition or material descriptor features",
        "eligible_source_types": ["public_reference", "experimental_test", "public_experimental"],
        "model_type": "Tabular regressor",
        "model_family": "experimental",
        "status": "waiting",
        "description": "Thermal conductivity — waiting for TPSX/upload labelled data.",
    },
    "interior_foam_properties": {
        "subsystem": "Interior / Seating / Foam",
        "target": "voc_emission_score",
        "required_features": "interior-specific test fields",
        "eligible_source_types": ["experimental_test", "supplier_sheet"],
        "model_type": "Tabular regressor",
        "model_family": "experimental",
        "status": "waiting",
        "description": "Interior foam models — waiting for upload evidence.",
    },
    "tyre_elastomer_properties": {
        "subsystem": "Tyres / Elastomers",
        "target": "wear_resistance_score",
        "required_features": "tyre-specific test fields",
        "eligible_source_types": ["experimental_test", "supplier_sheet"],
        "model_type": "Tabular regressor",
        "model_family": "experimental",
        "status": "waiting",
        "description": "Tyre/elastomer models — waiting for upload evidence.",
    },
}

COMPUTED_SOURCE_TYPES = {"computed_database", "public_benchmark"}

SUBSYSTEM_ACTIVE_MODELS: dict[str, list[str]] = {
    "Structural / Chassis": ["structural_yield_strength", "elastic_modulus_proxy"],
    "Electronics / Thermal Interface": ["computed_band_gap"],
    "Battery Enclosure / Underbody": ["computed_formation_energy"],
    "General Material Reuse": ["computed_formation_energy", "elastic_modulus_proxy"],
}


def _eligible_rows(unified_df: pd.DataFrame, target: str, eligible_types: list[str]) -> pd.Series:
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


def _feature_count(unified_df: pd.DataFrame, spec: dict) -> int:
    req = spec["required_features"]
    if "wt_percent_" in req:
        return len([c for c in unified_df.columns if c.startswith("wt_percent_")])
    count = 0
    if "formula" in unified_df.columns and unified_df["formula"].notna().any():
        count += 1
    if "density_g_cm3" in unified_df.columns and unified_df["density_g_cm3"].notna().any():
        count += 1
    count += len([c for c in unified_df.columns if c.startswith("wt_percent_")])
    return count


def _trained_model_keys() -> set[str]:
    trained = load_all_trained_models()
    return set(trained.keys())


def detect_trainable_targets(unified_df: pd.DataFrame) -> pd.DataFrame:
    trained_keys = _trained_model_keys()
    registry_meta = {m["model_key"]: m for m in load_model_registry()}
    rows = []

    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        eligible_types = spec["eligible_source_types"]

        eligible_mask = _eligible_rows(unified_df, target, eligible_types)
        eligible_count = int(eligible_mask.sum())

        alt_target = spec.get("alt_target")
        if alt_target and eligible_count == 0:
            alt_mask = _eligible_rows(unified_df, alt_target, eligible_types)
            if int(alt_mask.sum()) > eligible_count:
                target = alt_target
                eligible_mask = alt_mask
                eligible_count = int(alt_mask.sum())

        feature_count = _feature_count(unified_df, spec)
        is_computed = spec.get("model_family") == "computed_reference"
        min_rows = int(MODEL_TRAINING_SPECS.get(model_name, {}).get("min_rows", 30))

        if model_name in trained_keys:
            meta = registry_meta.get(model_name, {})
            family = spec.get("model_family", "experimental")
            if family == "computed_reference":
                status = "Active computed-reference"
                reason = (
                    f"Trained computed-reference model on {meta.get('rows', eligible_count)} rows "
                    f"({meta.get('selected_algorithm', 'tabular regressor')}). "
                    "Validation required before engineering release."
                )
            else:
                status = "Active"
                reason = (
                    f"Trained experimental model on {meta.get('rows', eligible_count)} eligible rows."
                )
        elif spec["status"] == "waiting":
            if eligible_count > 0:
                status = "Insufficient data"
                reason = f"{eligible_count} rows found but model not yet implemented."
            else:
                status = "Waiting for evidence"
                reason = f"Waiting for {target} from TPSX/upload/supplier evidence."
        elif eligible_count >= min_rows and feature_count >= 1:
            status = "Computed-reference trainable" if is_computed else "Trainable"
            reason = f"{eligible_count} eligible rows from reference sources."
        elif eligible_count > 0:
            status = "Insufficient data"
            reason = f"Only {eligible_count} eligible rows (need {min_rows}+)."
        else:
            status = "Waiting for evidence"
            reason = (
                f"No labelled {spec['target']} from {', '.join(eligible_types)}. "
                "Load JARVIS/matbench or upload evidence."
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
            "model_family": spec.get("model_family", "experimental"),
        })

    return pd.DataFrame(rows)


def get_active_models(unified_df: pd.DataFrame) -> list[dict]:
    registry = detect_trainable_targets(unified_df)
    active_statuses = [
        "Active", "Active computed-reference",
        "Trainable", "Trainable next", "Computed-reference trainable",
    ]
    return registry[registry["status"].isin(active_statuses)].to_dict("records")


def get_trainable_next_models(unified_df: pd.DataFrame) -> list[dict]:
    registry = detect_trainable_targets(unified_df)
    return registry[registry["status"].isin([
        "Trainable next", "Computed-reference trainable", "Trainable",
    ])].to_dict("records")


def get_trained_models_for_subsystem(subsystem: str, trained_models: dict | None = None) -> list[str]:
    """Return model keys trained and mapped to a subsystem."""
    if trained_models is None:
        trained_models = load_all_trained_models()
    keys = SUBSYSTEM_ACTIVE_MODELS.get(subsystem, [])
    return [k for k in keys if k in trained_models]
