"""Hackathon demo UI labels and display helpers — simple user-facing language."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.upload_workflow import SUBSYSTEM_TEMPLATE_COLUMNS

# Known model-ready target properties
MODEL_TARGET_FIELDS = [
    "yield_strength_mpa", "ultimate_tensile_strength_mpa", "band_gap_ev",
    "formation_energy_per_atom", "bulk_modulus_gpa", "shear_modulus_gpa",
    "thermal_conductivity_w_mk", "comfort_score", "durability_score", "wear_resistance_score",
]

MODEL_CARDS: dict[str, dict[str, Any]] = {
    "structural_yield_strength": {
        "title": "Yield strength",
        "target_label": "Yield strength (MPa)",
        "helps": "Structural",
        "family_label": "Measured-data model",
    },
    "computed_band_gap": {
        "title": "Band gap",
        "target_label": "Band gap (eV)",
        "helps": "Electronics",
        "family_label": "Reference-data model",
    },
    "computed_formation_energy": {
        "title": "Formation energy",
        "target_label": "Formation energy (eV/atom)",
        "helps": "Battery · General reuse",
        "family_label": "Reference-data model",
    },
    "elastic_modulus_proxy": {
        "title": "Elastic modulus proxy",
        "target_label": "Bulk / shear modulus (GPa)",
        "helps": "Structural · General reuse",
        "family_label": "Reference-data model",
    },
}

STATUS_LABELS: dict[str, str] = {
    "Active": "Active model available",
    "Active computed-reference": "Active model available",
    "Trainable": "Ready when data is added",
    "Trainable next": "Ready when data is added",
    "Computed-reference trainable": "Ready when data is added",
    "Waiting for evidence": "Ready for new evidence",
    "Insufficient data": "More evidence required",
    "Active model": "Active model available",
    "Active computed-reference model": "Active model available",
    "Computed-reference coverage": "Reference coverage",
    "Partial coverage": "Reference coverage",
    "Upload evidence needed": "Add evidence to activate",
}

SUBSYSTEM_UNLOCK: dict[str, str] = {
    "Structural / Chassis": "Unlocks yield strength predictions and structural fit scoring.",
    "Battery Enclosure / Underbody": "Unlocks enclosure screening and formation-energy insights.",
    "Electronics / Thermal Interface": "Unlocks band-gap screening and thermal interface scoring.",
    "Interior / Seating / Foam": "Unlocks interior circularity and comfort scoring.",
    "Tyres / Elastomers": "Unlocks wear and rolling-resistance scoring.",
    "Thermal Fluids / Coolants": "Unlocks thermal fluid screening models.",
    "Coatings / Corrosion Protection": "Unlocks corrosion and coating fit scoring.",
    "General Material Reuse": "Unlocks cross-subsystem reuse and reference screening.",
}


def display_status(internal: str) -> str:
    return STATUS_LABELS.get(internal, internal)


def count_model_ready_rows(df: pd.DataFrame) -> int:
    cols = [c for c in MODEL_TARGET_FIELDS if c in df.columns]
    if not cols:
        return 0
    return int(df[cols].notna().any(axis=1).sum())


def upload_template_count() -> int:
    return len(SUBSYSTEM_TEMPLATE_COLUMNS)
