"""Deterministic CSV schema mapping for MatIntel unified material library.

Maps arbitrary uploaded CSV columns to MatIntel standard fields using:
1. Exact standard field name match (confidence 100)
2. Exact alias match (confidence 90)
3. Fuzzy match via difflib (confidence 60-85)
4. Unmatched columns -> "ignore"

Also handles unit conversions and source type classification with ML eligibility.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from html import escape as html_escape

import numpy as np
import pandas as pd


def esc(val) -> str:
    """HTML-escape a value for safe rendering in unsafe_allow_html contexts."""
    return html_escape(str(val)) if val is not None else ""


STANDARD_FIELDS: list[str] = [
    "material_id", "material_name", "formula",
    "material_family", "material_subfamily", "application_subsystem",
    "source_dataset", "source_type", "source_trust_score", "used_for_ml_training",
    "density_g_cm3", "yield_strength_mpa", "ultimate_tensile_strength_mpa",
    "youngs_modulus_gpa", "elongation_percent", "hardness_hv",
    "fatigue_strength_mpa", "thermal_conductivity_w_mk", "corrosion_resistance_score",
    "bulk_modulus_gpa", "shear_modulus_gpa",
    "formation_energy_per_atom", "band_gap_ev", "specific_heat",
    "viscosity", "freezing_point", "boiling_point",
    "impact_resistance_score", "flame_retardancy_score", "electrical_insulation_score",
    "comfort_score", "durability_score", "voc_emission_score", "fire_safety_score",
    "wear_resistance_score", "rolling_resistance_score", "grip_score",
    "salt_spray_hours", "adhesion_score", "scratch_resistance_score",
    "toxicity_score", "corrosion_inhibition_score",
    "cost_index", "co2_index", "co2_kg_per_kg",
    "recycled_content_percent", "recyclability_score",
    "bio_based_content_percent", "closed_loop_available",
    "supplier_name", "supplier_risk_score", "traceability_score",
    "critical_material_risk_score", "certification_tags",
    "data_completeness_score", "prediction_confidence_score",
    "model_registry_eligible", "notes",
]

SOURCE_TYPE_PROFILES: dict[str, dict] = {
    "public_experimental": {
        "trust": 95, "ml_eligible": True,
        "ml_condition": "Labelled mechanical properties present",
        "usage": "ML training + decision support",
        "description": "Peer-reviewed or curated experimental measurements.",
    },
    "experimental_test": {
        "trust": 90, "ml_eligible": True,
        "ml_condition": "Composition + measured property columns present",
        "usage": "ML training + decision support",
        "description": "In-house or third-party test reports with measured properties.",
    },
    "supplier_sheet": {
        "trust": 70, "ml_eligible": False,
        "ml_condition": "Not eligible — supplier-reported values require independent verification",
        "usage": "Decision support only (screening, comparison)",
        "description": "Technical data sheets from material suppliers.",
    },
    "computed_database": {
        "trust": 80, "ml_eligible": False,
        "ml_condition": "Not eligible for experimental models — trainable for computed-reference models",
        "usage": "Reference / screening; computed model training",
        "description": "Computationally derived properties (DFT, MD, CALPHAD).",
    },
    "public_reference": {
        "trust": 80, "ml_eligible": False,
        "ml_condition": "Not eligible by default unless exported rows are labelled and reviewed",
        "usage": "Reference / screening only",
        "description": "Public reference databases (e.g. NASA TPSX pathway).",
    },
    "sustainability_sheet": {
        "trust": 65, "ml_eligible": False,
        "ml_condition": "Not eligible — sustainability data only",
        "usage": "Sustainability scoring only",
        "description": "LCA data, EPDs, recycled content declarations.",
    },
    "procurement_sheet": {
        "trust": 60, "ml_eligible": False,
        "ml_condition": "Not eligible — cost/supply data only",
        "usage": "Cost and supply-risk scoring only",
        "description": "Procurement pricing, lead times, supplier risk assessments.",
    },
    "unknown": {
        "trust": 40, "ml_eligible": False,
        "ml_condition": "Not eligible — source provenance unknown",
        "usage": "Manual review required before use",
        "description": "Unverified or unclassified data source.",
    },
}

ALL_SOURCE_TYPES = list(SOURCE_TYPE_PROFILES.keys())

SCHEMA_ALIASES: dict[str, list[str]] = {
    "yield_strength_mpa": [
        "ys", "yield", "yield strength", "yield_mpa", "proof stress",
        "rp0.2", "0.2 proof stress", "yield_strength", "rp02",
    ],
    "ultimate_tensile_strength_mpa": [
        "uts", "tensile", "tensile strength", "ultimate tensile strength",
        "rm", "tensile_strength", "ultimate_tensile_strength",
    ],
    "density_g_cm3": [
        "density", "rho", "specific gravity", "mass density",
        "density kg/m3", "density g/cm3", "density_kg_m3",
    ],
    "youngs_modulus_gpa": [
        "young", "youngs modulus", "elastic modulus", "modulus",
        "e modulus", "young's modulus", "youngs_modulus",
    ],
    "elongation_percent": ["elongation", "elongation %", "elong", "elongation_percent", "elongation at break"],
    "hardness_hv": ["hardness", "hv", "vickers", "vickers hardness"],
    "fatigue_strength_mpa": ["fatigue", "fatigue strength", "endurance limit", "fatigue_strength"],
    "thermal_conductivity_w_mk": ["thermal conductivity", "conductivity", "w/mk", "thermal_conductivity", "k value"],
    "corrosion_resistance_score": ["corrosion", "corrosion resistance", "salt spray", "corrosion score"],
    "bulk_modulus_gpa": ["bulk modulus", "bulk_modulus", "kv", "bulk modulus kv"],
    "shear_modulus_gpa": ["shear modulus", "shear_modulus", "gv", "shear modulus gv"],
    "formation_energy_per_atom": ["formation energy", "form_enp", "formation_energy_peratom"],
    "band_gap_ev": ["band gap", "bandgap", "band_gap", "optb88vdw_bandgap", "mbj_bandgap"],
    "specific_heat": ["specific heat", "cp", "heat capacity"],
    "recycled_content_percent": ["recycled", "recycled content", "recycled %", "pcr content", "recycled_content"],
    "co2_kg_per_kg": ["co2", "carbon footprint", "embodied carbon", "kgco2", "co2/kg", "kg co2e", "co2_kg_per_kg", "gwp"],
    "traceability_score": ["traceability", "chain of custody", "origin known", "supply chain visibility"],
    "supplier_risk_score": ["supplier risk", "supply risk", "sourcing risk", "geopolitical risk", "availability risk"],
    "critical_material_risk_score": ["critical material", "crm risk", "critical_material"],
    "certification_tags": ["certification", "fsc", "iso", "certificate", "certified"],
    "material_name": ["name", "material", "alloy", "alloy name", "grade"],
    "formula": ["composition", "chemical formula", "chem formula"],
    "material_family": ["family", "material family", "material type", "type", "class"],
    "material_subfamily": ["subfamily", "sub family", "subclass", "sub type"],
    "cost_index": ["cost", "price", "cost index", "price index", "usd/kg"],
    "supplier_name": ["supplier", "vendor", "manufacturer"],
    "recyclability_score": ["recyclability", "recyclable", "end of life"],
    "bio_based_content_percent": ["bio based", "bio content", "biobased", "bio %"],
    "co2_index": ["co2 index", "carbon index", "emissions index"],
    "notes": ["note", "comment", "comments", "remark", "remarks"],
}

UNIT_CONVERSIONS: list[dict] = [
    {"pattern": r"density.*kg.*m.?3", "target": "density_g_cm3", "factor": 0.001},
    {"pattern": r"density.*g.*cm.?3", "target": "density_g_cm3", "factor": 1.0},
    {"pattern": r"yield.*gpa", "target": "yield_strength_mpa", "factor": 1000.0},
    {"pattern": r"yield.*(?<!m)pa$", "target": "yield_strength_mpa", "factor": 1e-6},
    {"pattern": r"tensile.*gpa", "target": "ultimate_tensile_strength_mpa", "factor": 1000.0},
    {"pattern": r"modulus.*mpa\b", "target": "youngs_modulus_gpa", "factor": 0.001},
]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().strip())


def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def suggest_schema_mapping(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in uploaded_df.columns:
        norm = _normalize(col)
        matched = False
        for sf in STANDARD_FIELDS:
            if _normalize(sf) == norm:
                rows.append({"uploaded_column": col, "suggested_field": sf, "confidence": 100, "match_type": "exact_standard"})
                matched = True
                break
        if matched:
            continue

        for sf, aliases in SCHEMA_ALIASES.items():
            if any(_normalize(a) == norm for a in aliases):
                rows.append({"uploaded_column": col, "suggested_field": sf, "confidence": 90, "match_type": "alias"})
                matched = True
                break
        if matched:
            continue

        best_score, best_field = 0.0, "ignore"
        for sf in STANDARD_FIELDS:
            score = _fuzzy_score(col, sf)
            if score > best_score:
                best_score, best_field = score, sf
        for sf, aliases in SCHEMA_ALIASES.items():
            for alias in aliases:
                score = _fuzzy_score(col, alias)
                if score > best_score:
                    best_score, best_field = score, sf

        if best_score >= 0.65:
            rows.append({"uploaded_column": col, "suggested_field": best_field, "confidence": round(best_score * 100, 0), "match_type": "fuzzy"})
        else:
            rows.append({"uploaded_column": col, "suggested_field": "ignore", "confidence": round(best_score * 100, 0), "match_type": "none"})

    return pd.DataFrame(rows)


def _detect_unit_conversion(col_name: str, target_field: str) -> float:
    norm = col_name.lower().strip()
    for conv in UNIT_CONVERSIONS:
        if re.search(conv["pattern"], norm) and conv["target"] == target_field:
            return conv["factor"]
    return 1.0


def assess_ml_eligibility(mapped_df: pd.DataFrame, source_type: str) -> dict:
    profile = SOURCE_TYPE_PROFILES.get(source_type, SOURCE_TYPE_PROFILES["unknown"])
    if not profile["ml_eligible"]:
        return {"eligible": False, "reason": profile["ml_condition"], "usage": profile["usage"]}

    has_yield = "yield_strength_mpa" in mapped_df.columns and mapped_df["yield_strength_mpa"].notna().sum() > 0
    has_composition = any(c.startswith("wt_percent_") for c in mapped_df.columns)

    if source_type == "experimental_test" and not (has_yield and has_composition):
        return {"eligible": False, "reason": "Requires composition (wt%) and measured yield strength", "usage": "Decision support only until data added"}

    if not has_yield:
        return {"eligible": False, "reason": "No measured yield strength values found", "usage": "Decision support only"}

    return {"eligible": True, "reason": profile["ml_condition"], "usage": profile["usage"]}


def apply_schema_mapping(
    uploaded_df: pd.DataFrame, mapping_dict: dict[str, str],
    source_label: str = "uploaded_supplier_sheet", source_type: str = "supplier_sheet",
) -> pd.DataFrame:
    profile = SOURCE_TYPE_PROFILES.get(source_type, SOURCE_TYPE_PROFILES["unknown"])
    out = pd.DataFrame()
    n = len(uploaded_df)

    out["material_id"] = [f"UPLOAD_{i:04d}" for i in range(n)]
    out["source_dataset"] = source_label
    out["source_type"] = source_type
    out["source_trust_score"] = profile["trust"]
    out["used_for_ml_training"] = False
    out["model_registry_eligible"] = profile["ml_eligible"]

    for upload_col, std_field in mapping_dict.items():
        if std_field == "ignore" or std_field not in STANDARD_FIELDS:
            continue
        if upload_col not in uploaded_df.columns:
            continue
        factor = _detect_unit_conversion(upload_col, std_field)
        values = uploaded_df[upload_col]
        if factor != 1.0:
            values = pd.to_numeric(values, errors="coerce") * factor
        out[std_field] = values

    for sf in STANDARD_FIELDS:
        if sf not in out.columns:
            out[sf] = np.nan

    for col in uploaded_df.columns:
        if col not in mapping_dict or mapping_dict.get(col) == "ignore":
            safe_name = re.sub(r"[^a-z0-9_]", "_", col.lower().strip())
            out[f"extra_{safe_name}"] = uploaded_df[col].values

    key_cols = ["yield_strength_mpa", "density_g_cm3", "elongation_percent", "youngs_modulus_gpa"]
    present = [c for c in key_cols if c in out.columns]
    if present:
        out["data_completeness_score"] = (100 * (1 - out[present].isna().mean(axis=1))).round(1)
    else:
        out["data_completeness_score"] = 0.0

    return out
