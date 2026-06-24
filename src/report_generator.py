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


def _is_observed(row: pd.Series) -> bool:
    """Whether yield strength was directly observed (from experimental source)."""
    source = str(row.get("source_type", "")).lower()
    return source in ("experimental", "public_experimental", "experimental_test")


def build_report(
    row: pd.Series,
    min_strength: float,
    use_case: str = "Lightweight Structural Component",
    model_metrics: dict | None = None,
    n_reference_rows: int = 312,
    n_uploaded_rows: int = 0,
) -> str:
    name = row.get("material_name", "Selected material")
    mid = row.get("material_id", "N/A")
    family = row.get("material_family", row.get("family", "N/A"))
    source = row.get("source_dataset", "N/A")
    source_type = row.get("source_type", "N/A")
    trust = _fmt(row.get("source_trust_score"), "/100")

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

    observed = _is_observed(row)
    property_basis = "Directly observed (experimental measurement)" if observed else "Model-predicted (RandomForest ensemble)"

    # Score breakdown table
    sub_score_fields = [
        ("Strength", "strength_score"),
        ("Weight Saving", "weight_saving_score"),
        ("Stiffness", "stiffness_score"),
        ("Confidence", "confidence_score"),
        ("Sustainability", "sustainability_score"),
        ("Manufacturability", "manufacturability_score"),
        ("Cost", "cost_score"),
        ("Supply Risk", "supply_risk_score"),
    ]
    score_rows = ""
    for label, col in sub_score_fields:
        val = row.get(col)
        score_rows += f"| {label} | {_fmt(val, '/100')} |\n"

    # Missing data flags
    missing: list[str] = []
    for field, label in [
        ("fatigue_strength_mpa", "Fatigue strength"),
        ("corrosion_resistance_score", "Corrosion resistance"),
        ("hardness_hv", "Hardness"),
        ("thermal_conductivity_w_mk", "Thermal conductivity"),
        ("supplier_risk_score", "Supplier risk assessment"),
        ("traceability_score", "Supply chain traceability"),
    ]:
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing.append(f"- **{label}**: not available in source data")

    missing_section = "\n".join(missing) if missing else "- All key fields present."

    # Model info
    model_line = ""
    if model_metrics:
        model_line = (
            f"\n- Predictive model: RandomForest ensemble (R² = {model_metrics.get('R2', 'N/A')}, "
            f"MAE = {model_metrics.get('MAE', 'N/A')} MPa, "
            f"trained on {model_metrics.get('train_rows', 'N/A')} experimental rows)"
        )

    # Reason codes
    reason_lines = "\n".join(
        f"- {r.strip()}" for r in reasons.split("|") if r.strip()
    )

    # Uploaded sources line
    uploaded_line = ""
    if n_uploaded_rows > 0:
        uploaded_line = f"\n- Uploaded evidence: **{n_uploaded_rows} rows** (supplier sheet, decision support only)"

    # Validation gates
    gate_lines = "\n".join(
        f"{i+1}. **{g['gate']}** — {g['detail']}" for i, g in enumerate(VALIDATION_GATES)
    )

    return f"""# MatIntel Decision Report

## Use Case
**{use_case}** — Alternative material screening for JLR automotive application.

## Selected Material
| Field | Value |
|-------|-------|
| Material ID | {mid} |
| Material Name | {name} |
| Family | {family} |
| Source Dataset | {source} |
| Source Type | {source_type} |
| Source Trust | {trust} |
| Property Basis | {property_basis} |

## Data Sources Used
- **Layer 1 — Public reference data:** matminer steel_strength ({n_reference_rows} experimental steel alloys, trust 95/100){uploaded_line}{model_line}

> Public datasets bootstrap the predictive layer before private JLR data is connected.
> Uploaded sheets are scored and mapped but not blindly ingested into ML training.

## Known / Predicted Properties
| Property | Value |
|----------|-------|
| Yield Strength | {ys} |
| Ultimate Tensile Strength | {uts} |
| Elongation | {elong} |
| Density | {density} |
| Young's Modulus | {modulus} |
| Fatigue Strength | {fatigue} |
| Corrosion Resistance | {corrosion} |
| Hardness | {hardness} |

## Suitability Score
**{score}** (minimum required yield strength: {min_strength} MPa)

Weight saving vs steel baseline: **{ws}**

### Score Breakdown
| Sub-Score | Value |
|-----------|-------|
{score_rows}
> Final screening score is a weighted engineering score, not a black-box ML output. \
Weights are set per JLR use-case preset.

## Why Recommended
{reason_lines}

## Missing Data / Risk Flags
{missing_section}

> Fields labelled "demo enrichment" are indicative defaults, not measured values from the source dataset.

## Validation Gates
Before engineering release, the following gates must be passed:
{gate_lines}

## Next Action
Prioritise **{name}** for lab validation and compare against the current steel baseline under the **{use_case}** programme.

---
*Generated by MatIntel — Material Intelligence Platform. Screening only; not final engineering approval.*
"""
