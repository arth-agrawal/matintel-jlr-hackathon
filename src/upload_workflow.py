"""Guided CSV upload workflow: templates, mapping review, ambiguity resolution, ingest."""

from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pandas as pd

from src.schema_mapper import (
    STANDARD_FIELDS,
    SOURCE_TYPE_PROFILES,
    suggest_schema_mapping,
    apply_schema_mapping,
    assess_ml_eligibility,
    _detect_unit_conversion,
    esc,
)
from src.subsystem_profiles import ALL_SUBSYSTEMS, SUBSYSTEM_PROFILES

UNIVERSAL_TEMPLATE_COLUMNS = [
    "material_name", "formula", "material_family", "application_subsystem",
    "source_dataset", "source_type", "notes",
]

SUBSYSTEM_TEMPLATE_COLUMNS: dict[str, list[str]] = {
    "Universal": UNIVERSAL_TEMPLATE_COLUMNS,
    "Structural / Chassis": UNIVERSAL_TEMPLATE_COLUMNS + [
        "yield_strength_mpa", "ultimate_tensile_strength_mpa", "density_g_cm3",
        "youngs_modulus_gpa", "elongation_percent", "fatigue_strength_mpa",
        "corrosion_resistance_score",
    ],
    "Battery Enclosure / Underbody": UNIVERSAL_TEMPLATE_COLUMNS + [
        "density_g_cm3", "thermal_conductivity_w_mk", "flame_retardancy_score",
        "impact_resistance_score", "corrosion_resistance_score",
        "electrical_insulation_score", "traceability_score",
    ],
    "Electronics / Thermal Interface": UNIVERSAL_TEMPLATE_COLUMNS + [
        "thermal_conductivity_w_mk", "electrical_insulation_score", "band_gap_ev",
        "dielectric_constant", "flame_retardancy_score", "density_g_cm3",
    ],
    "Interior / Seating / Foam": UNIVERSAL_TEMPLATE_COLUMNS + [
        "density_g_cm3", "comfort_score", "durability_score", "voc_emission_score",
        "fire_safety_score", "recycled_content_percent", "co2_kg_per_kg",
    ],
    "Thermal Fluids / Coolants": UNIVERSAL_TEMPLATE_COLUMNS + [
        "thermal_conductivity_w_mk", "specific_heat", "viscosity", "freezing_point",
        "boiling_point", "corrosion_inhibition_score", "toxicity_score",
    ],
    "Tyres / Elastomers": UNIVERSAL_TEMPLATE_COLUMNS + [
        "wear_resistance_score", "rolling_resistance_score", "grip_score",
        "recycled_content_percent", "bio_based_content_percent", "traceability_score",
    ],
    "Coatings / Corrosion Protection": UNIVERSAL_TEMPLATE_COLUMNS + [
        "corrosion_resistance_score", "salt_spray_hours", "adhesion_score",
        "scratch_resistance_score", "voc_emission_score",
    ],
    "General Material Reuse": UNIVERSAL_TEMPLATE_COLUMNS + [
        "density_g_cm3", "yield_strength_mpa", "band_gap_ev", "formation_energy_per_atom",
        "recyclability_score", "recycled_content_percent", "source_trust_score",
    ],
}

AMBIGUOUS_COLUMN_TARGETS: dict[str, list[str]] = {
    "e": ["youngs_modulus_gpa", "elongation_percent"],
    "ys": ["yield_strength_mpa"],
    "hardness": ["hardness_hv"],
    "density": ["density_g_cm3"],
}

DENSITY_UNIT_OPTIONS = ["g/cm³ (default)", "kg/m³ (convert ÷1000)"]

UPLOAD_GUIDELINES = """
**Add New Material Evidence** — CSV upload in four steps:
1. **Upload CSV** — one row per material or test record
2. **Review column mapping** — match your headers to MatIntel fields
3. **Confirm units and subsystem** — engineer review when ready
4. **Ingest** — update material universe, scoring, and models

If the file contains a usable target property, MatIntel can add it to the model registry and retrain the relevant model.
"""

SUBSYSTEM_COLUMN_GUIDANCE: dict[str, str] = {
    "Structural / Chassis": "yield_strength_mpa, ultimate_tensile_strength_mpa, density_g_cm3, youngs_modulus_gpa, elongation_percent, fatigue_strength_mpa, corrosion_resistance_score",
    "Battery Enclosure / Underbody": "density_g_cm3, thermal_conductivity_w_mk, flame_retardancy_score, impact_resistance_score, corrosion_resistance_score, electrical_insulation_score, traceability_score",
    "Electronics / Thermal Interface": "thermal_conductivity_w_mk, electrical_insulation_score, band_gap_ev, dielectric_constant, flame_retardancy_score, density_g_cm3",
    "Interior / Seating / Foam": "density_g_cm3, comfort_score, durability_score, voc_emission_score, fire_safety_score, recycled_content_percent, co2_kg_per_kg",
    "Thermal Fluids / Coolants": "thermal_conductivity_w_mk, specific_heat, viscosity, freezing_point, boiling_point, corrosion_inhibition_score, toxicity_score",
    "Tyres / Elastomers": "wear_resistance_score, rolling_resistance_score, grip_score, recycled_content_percent, bio_based_content_percent, traceability_score",
    "Coatings / Corrosion Protection": "corrosion_resistance_score, salt_spray_hours, adhesion_score, scratch_resistance_score, voc_emission_score",
}

SOURCE_ROUTE_HINTS: dict[str, list[str]] = {
    "Thermal Fluids / Coolants": ["NASA TPSX CSV import", "Internal lab fluid test sheet", "Supplier coolant datasheet"],
    "Interior / Seating / Foam": ["Supplier sustainability sheet", "VOC/fire test report", "Procurement circularity sheet"],
    "Tyres / Elastomers": ["Supplier tyre compound sheet", "Rolling resistance test report", "Internal lab wear test"],
    "Coatings / Corrosion Protection": ["NASA TPSX CSV import", "Salt-spray test report", "Supplier coating datasheet"],
    "Battery Enclosure / Underbody": ["Internal crash/corrosion test", "Supplier enclosure sheet", "JARVIS computed reference (screening)"],
}


def generate_template_csv(template_name: str, include_example: bool = True) -> str:
    """Return CSV string for download — example row marked, not real data."""
    cols = SUBSYSTEM_TEMPLATE_COLUMNS.get(template_name, UNIVERSAL_TEMPLATE_COLUMNS)
    lines = [",".join(cols)]
    if include_example:
        example = {c: "" for c in cols}
        example["material_name"] = "EXAMPLE_DO_NOT_INGEST"
        example["formula"] = "ExampleAlloy"
        example["application_subsystem"] = template_name if template_name in ALL_SUBSYSTEMS else ""
        example["source_type"] = "experimental_test"
        example["notes"] = "EXAMPLE ROW — remove before ingest"
        lines.append(",".join(str(example.get(c, "")) for c in cols))
    return "\n".join(lines) + "\n"


def template_download_bytes(template_name: str) -> bytes:
    return generate_template_csv(template_name).encode("utf-8")


def enrich_mapping_with_units(upload_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Add detected_unit and needs_confirmation to mapping suggestions."""
    rows = []
    for _, row in mapping_df.iterrows():
        col = row["uploaded_column"]
        target = row["suggested_field"]
        conf = float(row["confidence"])
        factor = _detect_unit_conversion(col, target) if target != "ignore" else 1.0
        unit = "implicit" if factor == 1.0 else f"conversion ×{factor}"
        norm = re.sub(r"[^a-z0-9]", "", str(col).lower())
        ambiguous = False
        question = ""
        if norm in AMBIGUOUS_COLUMN_TARGETS and len(AMBIGUOUS_COLUMN_TARGETS[norm]) > 1:
            ambiguous = True
            opts = " or ".join(AMBIGUOUS_COLUMN_TARGETS[norm])
            question = f"'{col}' could map to {opts}. Confirm target field."
        elif target == "density_g_cm3" and factor == 1.0 and "density" in col.lower():
            if not re.search(r"g.*cm|kg.*m", col.lower()):
                ambiguous = True
                question = f"Density column '{col}' has no unit — choose g/cm³ or kg/m³."
        elif conf < 85:
            ambiguous = True
            question = f"'{col}' → {target} at {conf:.0f}% confidence. Confirm mapping?"
        elif "hrc" in col.lower() and "hardness" in col.lower():
            ambiguous = True
            question = f"'{col}' uses HRC — cannot safely convert to HV. Preserve as extra_* or provide conversion source."

        rows.append({
            **row.to_dict(),
            "detected_unit": unit,
            "needs_confirmation": ambiguous or conf < 85,
            "review_question": question,
        })
    return pd.DataFrame(rows)


def build_review_questions(
    mapping_review: pd.DataFrame,
    overrides: dict[str, str],
    source_type: str,
    subsystem_choice: str,
) -> list[dict[str, Any]]:
    """Deterministic clarification questions before ingest."""
    questions: list[dict[str, Any]] = []

    for _, row in mapping_review.iterrows():
        col = row["uploaded_column"]
        chosen = overrides.get(col, row["suggested_field"])
        if row.get("needs_confirmation") and not row.get("confirmed"):
            questions.append({
                "id": f"map_{col}",
                "type": "mapping",
                "column": col,
                "prompt": row.get("review_question") or f"Confirm mapping for '{col}'",
                "options": ["accept"] + (
                    AMBIGUOUS_COLUMN_TARGETS.get(
                        re.sub(r"[^a-z0-9]", "", col.lower()), [chosen]
                    )
                ),
                "current": chosen,
            })

    if subsystem_choice == "infer from file if present":
        questions.append({
            "id": "subsystem_infer",
            "type": "subsystem",
            "prompt": "Which subsystem should rows belong to if application_subsystem is missing?",
            "options": ALL_SUBSYSTEMS,
            "current": "Structural / Chassis",
        })

    if source_type == "unknown":
        questions.append({
            "id": "source_clarify",
            "type": "source",
            "prompt": "Is this experimental data or supplier datasheet?",
            "options": ["experimental_test", "supplier_sheet", "sustainability_sheet", "procurement_sheet"],
            "current": "supplier_sheet",
        })

    return questions


def critical_mappings_unresolved(
    mapping_review: pd.DataFrame,
    overrides: dict[str, str],
    confirmations: dict[str, bool],
) -> list[str]:
    """Return list of unresolved critical mapping issues."""
    unresolved = []
    for _, row in mapping_review.iterrows():
        col = row["uploaded_column"]
        if not row.get("needs_confirmation"):
            continue
        if not confirmations.get(col, False):
            unresolved.append(col)
    return unresolved


def compute_ml_eligibility(
    mapped_preview: pd.DataFrame,
    source_type: str,
    engineer_reviewed: bool,
) -> dict[str, Any]:
    base = assess_ml_eligibility(mapped_preview, source_type)
    if source_type == "supplier_sheet" and not engineer_reviewed:
        return {
            **base,
            "eligible": False,
            "reason": "Supplier sheet — screening only until engineer_reviewed is checked",
        }
    if source_type == "unknown":
        return {**base, "eligible": False, "reason": "Unknown source — review required before ML eligibility"}
    if engineer_reviewed and source_type == "experimental_test":
        has_target = mapped_preview["yield_strength_mpa"].notna().any() if "yield_strength_mpa" in mapped_preview.columns else False
        has_comp = any(c.startswith("wt_percent_") for c in mapped_preview.columns)
        if has_target and has_comp:
            return {**base, "eligible": True, "reason": "Engineer-reviewed experimental test with labels"}
    return base


def ingest_approved_evidence(
    upload_df: pd.DataFrame,
    overrides: dict[str, str],
    *,
    source_type: str,
    application_subsystem: str | None,
    infer_subsystem: bool,
    engineer_reviewed: bool,
    mapping_notes: str,
    engineer_review_notes: str,
    unit_factors: dict[str, float] | None = None,
    drop_example_rows: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply mapping and attach upload metadata after engineer approval."""
    work_df = upload_df.copy()
    if drop_example_rows and "material_name" in work_df.columns:
        work_df = work_df[
            ~work_df["material_name"].astype(str).str.contains("EXAMPLE_DO_NOT_INGEST", case=False, na=False)
        ]

    mapped = apply_schema_mapping(
        work_df,
        overrides,
        source_label="engineer_reviewed_upload",
        source_type=source_type,
        application_subsystem=application_subsystem,
        infer_subsystem=infer_subsystem,
        engineer_reviewed=engineer_reviewed,
        mapping_notes=mapping_notes,
        engineer_review_notes=engineer_review_notes,
        unit_factors=unit_factors,
    )

    ml_info = compute_ml_eligibility(mapped, source_type, engineer_reviewed)
    if ml_info["eligible"]:
        mapped["used_for_ml_training"] = True
        mapped["model_registry_eligible"] = True

    mapped_fields = [c for c in overrides.values() if c != "ignore" and c in STANDARD_FIELDS]
    ignored = [c for c in overrides if overrides[c] == "ignore"]
    extra_cols = [c for c in mapped.columns if c.startswith("extra_")]

    summary = {
        "rows_ingested": len(mapped),
        "subsystem": application_subsystem or ("inferred from file" if infer_subsystem else "—"),
        "source_type": source_type,
        "source_trust": SOURCE_TYPE_PROFILES.get(source_type, {}).get("trust", 40),
        "ml_eligible_rows": int(mapped["used_for_ml_training"].astype(str).str.lower().isin(["true", "1"]).sum()) if "used_for_ml_training" in mapped.columns else 0,
        "fields_mapped": mapped_fields,
        "fields_ignored": ignored,
        "extra_fields": extra_cols,
        "ml_assessment": ml_info,
        "mapping_notes": mapping_notes,
        "engineer_review_notes": engineer_review_notes,
        "engineer_reviewed": engineer_reviewed,
    }
    return mapped, summary
