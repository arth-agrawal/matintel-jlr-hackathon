"""Engineering-style decision report generator for MatIntel."""

from __future__ import annotations
import numpy as np
import pandas as pd


VALIDATION_GATES: list[dict[str, str]] = [
    {"gate": "Fatigue Testing", "detail": "Confirm cyclic load performance under JLR fatigue spec."},
    {"gate": "Corrosion Testing", "detail": "Salt spray / electrochemical validation per JLR corrosion standard."},
    {"gate": "Crash / Impact Validation", "detail": "Dynamic performance under JLR crash and impact specification."},
    {"gate": "Joining / Manufacturing Feasibility", "detail": "Weldability, formability, assembly compatibility."},
    {"gate": "Supplier Traceability", "detail": "Confirm supply chain origin, certification, and traceability."},
    {"gate": "Cost / LCA Confirmation", "detail": "Verify unit cost, tooling cost, and lifecycle carbon footprint."},
]


def _fmt(val, suffix: str = "", fallback: str = "Not available") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    return f"{val}{suffix}"


def _property_basis(row: pd.Series) -> str:
    source = str(row.get("source_type", "")).lower()
    if source in ("public_experimental", "experimental", "experimental_test"):
        return "Directly observed (experimental measurement)"
    elif source == "computed_database":
        return "Computed reference (DFT / simulation)"
    return "Model-predicted or reference-only"


def build_report(
    row: pd.Series,
    min_strength: float,
    use_case: str = "Lightweight Structural Component",
    model_metrics: dict | None = None,
    n_reference_rows: int = 312,
    n_uploaded_rows: int = 0,
    active_model_name: str = "structural_yield_strength",
    subsystem: str = "",
) -> str:
    name = row.get("material_name", "Selected material")
    mid = row.get("material_id", "N/A")
    family = row.get("material_family", row.get("family", "N/A"))
    source = row.get("source_dataset", "N/A")
    source_type = row.get("source_type", "N/A")
    trust = _fmt(row.get("source_trust_score"), "/100")
    subsystem_val = subsystem or row.get("application_subsystem", "N/A")
    prop_basis = _property_basis(row)

    ys = _fmt(row.get("yield_strength_mpa"), " MPa")
    uts = _fmt(row.get("ultimate_tensile_strength_mpa"), " MPa")
    elong = _fmt(row.get("elongation_percent"), "%")
    density = _fmt(row.get("density_g_cm3"), " g/cm³")
    modulus = _fmt(row.get("youngs_modulus_gpa"), " GPa")
    fatigue = _fmt(row.get("fatigue_strength_mpa"), " MPa")
    corrosion = _fmt(row.get("corrosion_resistance_score"), "/100")
    hardness = _fmt(row.get("hardness_hv"), " HV")

    score = _fmt(row.get("suitability_score"), "/100")
    ws = _fmt(row.get("weight_saving_percent"), "%")
    reasons = row.get("reason_codes", "No reason codes available")

    sub_score_fields = [
        ("Strength", "strength_score"), ("Weight Saving", "weight_saving_score"),
        ("Stiffness", "stiffness_score"), ("Source Trust", "source_trust_score_norm"),
        ("Sustainability", "sustainability_score"), ("Manufacturability", "manufacturability_score"),
        ("Cost", "cost_score"), ("Supply Risk", "supply_risk_score"),
    ]
    score_rows = ""
    for label, col in sub_score_fields:
        val = row.get(col)
        score_rows += f"| {label} | {_fmt(val, '/100')} |\n"

    missing: list[str] = []
    for field, label in [
        ("fatigue_strength_mpa", "Fatigue strength"), ("corrosion_resistance_score", "Corrosion resistance"),
        ("hardness_hv", "Hardness"), ("thermal_conductivity_w_mk", "Thermal conductivity"),
        ("supplier_risk_score", "Supplier risk"), ("traceability_score", "Traceability"),
    ]:
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing.append(f"- **{label}**: not available in source data")
    missing_section = "\n".join(missing) if missing else "- All key fields present."

    model_line = ""
    if model_metrics:
        model_line = (
            f"\n- Active model: **{active_model_name}** — RandomForest ensemble "
            f"(R² = {model_metrics.get('R2', 'N/A')}, MAE = {model_metrics.get('MAE', 'N/A')} MPa, "
            f"trained on {model_metrics.get('train_rows', 'N/A')} experimental rows)"
        )

    uploaded_line = ""
    if n_uploaded_rows > 0:
        uploaded_line = f"\n- Uploaded evidence: **{n_uploaded_rows} rows** (decision support)"

    reason_lines = "\n".join(f"- {r.strip()}" for r in reasons.split("|") if r.strip())
    gate_lines = "\n".join(f"{i+1}. **{g['gate']}** — {g['detail']}" for i, g in enumerate(VALIDATION_GATES))

    density_flag = ""
    if row.get("_density_variance_flag", False):
        density_flag = (
            "\n> Weight saving not differentiating in current source; "
            "upload lightweight candidate evidence or computed reference data."
        )

    subsystem_section = ""
    if subsystem:
        try:
            from src.subsystem_profiles import get_subsystem_readiness, SUBSYSTEM_PROFILES
            readiness = get_subsystem_readiness(row, subsystem)
            profile = SUBSYSTEM_PROFILES.get(subsystem, {})
            missing_fields = ", ".join(readiness["missing"]) if readiness["missing"] else "None"
            subsystem_section = f"""
## Subsystem Readiness — {subsystem}
- **Coverage:** {readiness['coverage_pct']}% ({readiness['readiness']})
- **Scoring focus:** {profile.get('scoring_focus', '—')}
- **Public data coverage:** {profile.get('coverage_note', '—')}
- **Missing evidence fields:** {missing_fields}
"""
        except ImportError:
            pass

    model_registry_section = f"""
## Model Registry
- **Model used for screening:** {active_model_name}
- **Property basis:** {prop_basis}
- **Platform note:** Current active experimental model is trained on structural steel data. The platform is designed to scale subsystem-wise as JLR/supplier/public evidence sources are connected.
"""

    return f"""# MatIntel Decision Report

## Use Case
**{use_case}** — {subsystem_val} subsystem screening for JLR.

## Selected Material
| Field | Value |
|-------|-------|
| Material ID | {mid} |
| Material Name | {name} |
| Family | {family} |
| Subsystem | {subsystem_val} |
| Source | {source} |
| Source Type | {source_type} |
| Source Trust | {trust} |
| Property Basis | {prop_basis} |

## Data Sources
- Public reference: matminer steel_strength ({n_reference_rows} experimental steel alloys, trust 95/100){uploaded_line}{model_line}

## Properties
| Property | Value |
|----------|-------|
| Yield Strength | {ys} |
| Tensile Strength | {uts} |
| Elongation | {elong} |
| Density | {density} |
| Young's Modulus | {modulus} |
| Fatigue Strength | {fatigue} |
| Corrosion Resistance | {corrosion} |
| Hardness | {hardness} |

## Suitability Score
**{score}** (min yield: {min_strength} MPa) · Weight saving: **{ws}**
{density_flag}

### Score Breakdown
| Component | Value |
|-----------|-------|
{score_rows}

## Rationale
{reason_lines}

## Missing Data
{missing_section}
{subsystem_section}
{model_registry_section}
## Validation Gates
{gate_lines}

## Next Action
Prioritise **{name}** for lab validation under the **{use_case}** programme.

---
*MatIntel — Screening only, not final engineering approval.*
"""
