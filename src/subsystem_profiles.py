"""JLR subsystem profiles for material intelligence.

Each subsystem defines important fields, trainable targets, validation gates,
scoring focus, and current public data coverage status.
"""

from __future__ import annotations


SUBSYSTEM_PROFILES: dict[str, dict] = {
    "Structural / Chassis": {
        "important_fields": [
            "yield_strength_mpa", "ultimate_tensile_strength_mpa",
            "elongation_percent", "density_g_cm3", "youngs_modulus_gpa",
            "fatigue_strength_mpa", "hardness_hv",
        ],
        "trainable_targets": ["yield_strength_mpa", "ultimate_tensile_strength_mpa"],
        "validation_gates": [
            "Fatigue testing", "Crash / impact validation",
            "Joining / manufacturing feasibility", "Corrosion testing",
        ],
        "scoring_focus": "Strength-to-weight ratio, stiffness, fatigue life.",
        "coverage_note": (
            "Strong coverage from matminer steel_strength (312 experimental alloys). "
            "Active yield-strength model trained on composition features."
        ),
    },
    "Battery Enclosure / Underbody": {
        "important_fields": [
            "yield_strength_mpa", "density_g_cm3", "youngs_modulus_gpa",
            "corrosion_resistance_score", "thermal_conductivity_w_mk",
            "elongation_percent",
        ],
        "trainable_targets": ["yield_strength_mpa"],
        "validation_gates": [
            "Corrosion testing", "Crash / impact validation",
            "Thermal management validation", "Joining / manufacturing feasibility",
        ],
        "scoring_focus": "Corrosion resistance, stiffness, crash safety, thermal management.",
        "coverage_note": (
            "Partial coverage from steel dataset. Corrosion and thermal data "
            "require uploaded evidence or JARVIS/NASA reference sources."
        ),
    },
    "Interior / Seating / Foam": {
        "important_fields": [
            "density_g_cm3", "recycled_content_percent", "recyclability_score",
            "voc_emission_score", "fire_safety_score", "comfort_score",
            "durability_score",
        ],
        "trainable_targets": [],
        "validation_gates": [
            "VOC emission testing", "Fire safety certification",
            "Durability / wear testing", "Recyclability assessment",
        ],
        "scoring_focus": "Sustainability, comfort, fire safety, VOC compliance.",
        "coverage_note": (
            "No public experimental dataset currently loaded. "
            "Requires uploaded supplier/test evidence."
        ),
    },
    "Tyres / Elastomers": {
        "important_fields": [
            "density_g_cm3", "wear_resistance_score", "rolling_resistance_score",
            "grip_score", "recycled_content_percent", "durability_score",
        ],
        "trainable_targets": [],
        "validation_gates": [
            "Wear resistance testing", "Rolling resistance validation",
            "Grip / wet performance", "Durability / aging",
        ],
        "scoring_focus": "Wear resistance, rolling resistance, grip, sustainability.",
        "coverage_note": (
            "No public dataset currently loaded. "
            "Requires tyre-specific supplier or test evidence."
        ),
    },
    "Thermal Fluids / Coolants": {
        "important_fields": [
            "thermal_conductivity_w_mk", "specific_heat", "density_g_cm3",
            "viscosity", "freezing_point", "boiling_point",
            "corrosion_inhibition_score", "toxicity_score",
        ],
        "trainable_targets": ["thermal_conductivity_w_mk", "specific_heat"],
        "validation_gates": [
            "Thermal performance testing", "Corrosion inhibition validation",
            "Toxicity / environmental assessment", "Long-term stability testing",
        ],
        "scoring_focus": "Thermal conductivity, heat capacity, corrosion inhibition.",
        "coverage_note": (
            "NASA TPSX pathway available for thermal property reference. "
            "Requires CSV export and engineer review."
        ),
    },
    "Coatings / Corrosion Protection": {
        "important_fields": [
            "corrosion_resistance_score", "salt_spray_hours", "adhesion_score",
            "scratch_resistance_score", "thermal_conductivity_w_mk",
            "co2_kg_per_kg",
        ],
        "trainable_targets": [],
        "validation_gates": [
            "Salt spray testing", "Adhesion testing",
            "Scratch / chip resistance", "UV / weathering resistance",
        ],
        "scoring_focus": "Corrosion protection, adhesion, durability.",
        "coverage_note": (
            "No public dataset currently loaded. "
            "JARVIS reference may provide some material properties. "
            "Requires coating-specific supplier or test evidence."
        ),
    },
    "Electronics / Thermal Interface": {
        "important_fields": [
            "thermal_conductivity_w_mk", "electrical_insulation_score",
            "band_gap_ev", "density_g_cm3", "youngs_modulus_gpa",
        ],
        "trainable_targets": ["band_gap_ev", "thermal_conductivity_w_mk"],
        "validation_gates": [
            "Thermal interface testing", "Electrical insulation validation",
            "Reliability / aging testing", "EMC compliance",
        ],
        "scoring_focus": "Thermal conductivity, electrical insulation, band gap.",
        "coverage_note": (
            "JARVIS-DFT provides band gap and modulus reference data. "
            "Thermal interface materials require uploaded evidence."
        ),
    },
    "General Material Reuse": {
        "important_fields": [
            "density_g_cm3", "yield_strength_mpa", "recyclability_score",
            "recycled_content_percent", "co2_kg_per_kg", "cost_index",
            "formation_energy_per_atom",
        ],
        "trainable_targets": ["formation_energy_per_atom", "bulk_modulus_gpa"],
        "validation_gates": [
            "Material characterisation", "Recyclability / circularity assessment",
            "Cost validation", "Supplier traceability",
        ],
        "scoring_focus": "Broad screening across density, strength, sustainability, cost.",
        "coverage_note": (
            "JARVIS-DFT provides broad inorganic materials reference. "
            "Steel dataset covers structural steels. "
            "Other families require uploaded evidence."
        ),
    },
}

ALL_SUBSYSTEMS = list(SUBSYSTEM_PROFILES.keys())


def get_subsystem_readiness(row, subsystem: str) -> dict:
    """Assess readiness of a material row for a given subsystem."""
    profile = SUBSYSTEM_PROFILES.get(subsystem)
    if not profile:
        return {"available": [], "missing": [], "readiness": "Unknown subsystem"}

    available = []
    missing = []
    for field in profile["important_fields"]:
        val = row.get(field)
        if val is not None and not (isinstance(val, float) and _isnan(val)):
            available.append(field)
        else:
            missing.append(field)

    total = len(profile["important_fields"])
    pct = len(available) / total * 100 if total > 0 else 0

    if pct >= 70:
        readiness = "Good coverage"
    elif pct >= 40:
        readiness = "Partial — needs additional evidence"
    else:
        readiness = "Low — upload subsystem-specific data"

    return {
        "available": available,
        "missing": missing,
        "readiness": readiness,
        "coverage_pct": round(pct, 0),
    }


def _isnan(val) -> bool:
    try:
        import math
        return math.isnan(val)
    except (TypeError, ValueError):
        return False
