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
        return "Observed-data based (experimental measurement)"
    if source == "computed_database":
        return "Computed-reference based (DFT / simulation — not experimental validation)"
    if source == "supplier_sheet":
        return "Upload-evidence based (supplier sheet — requires engineer review)"
    if source == "public_reference":
        return "Reference-only (public database pathway)"
    return "Insufficient evidence / model-predicted"


def _recommendation_basis(row: pd.Series) -> str:
    basis = _property_basis(row)
    if row.get("suitability_score") is None or (isinstance(row.get("suitability_score"), float) and np.isnan(row.get("suitability_score"))):
        return "Insufficient evidence"
    if "Computed" in basis:
        return "Computed-reference based screening"
    if "Upload" in basis:
        return "Upload-evidence based screening"
    if "Observed" in basis:
        return "Observed-data based screening"
    return basis


def build_report(
    row: pd.Series,
    min_strength: float,
    use_case: str = "Lightweight Structural Component",
    model_metrics: dict | None = None,
    n_reference_rows: int = 312,
    n_uploaded_rows: int = 0,
    active_model_name: str = "structural_yield_strength",
    model_status: str = "Active",
    model_bundle: dict | None = None,
    subsystem: str = "",
    registry_df: pd.DataFrame | None = None,
) -> str:
    name = row.get("material_name", "Selected material")
    mid = row.get("material_id", "N/A")
    family = row.get("material_family", row.get("family", "N/A"))
    source = row.get("source_dataset", "N/A")
    source_type = row.get("source_type", "N/A")
    trust = _fmt(row.get("source_trust_score"), "/100")
    subsystem_val = subsystem or row.get("application_subsystem", "N/A")
    prop_basis = _property_basis(row)
    rec_basis = _recommendation_basis(row)

    score = _fmt(row.get("suitability_score"), "/100")
    ws = _fmt(row.get("weight_saving_percent"), "%")
    reasons = row.get("reason_codes", "No reason codes available")
    dims_used = row.get("_score_dims_used", "—")
    dims_neutral = row.get("_score_dims_neutral", "—")

    from src.recommender import get_scoring_dimensions
    from src.subsystem_profiles import get_subsystem_readiness, SUBSYSTEM_PROFILES

    score_rows = ""
    for dim_key, label, weight in get_scoring_dimensions(subsystem_val):
        val = row.get(dim_key)
        score_rows += f"| {label} | {_fmt(val, '/100')} | {weight:.0%} |\n"

    readiness = get_subsystem_readiness(row, subsystem_val)
    profile = SUBSYSTEM_PROFILES.get(subsystem_val, {})
    available_fields = ", ".join(readiness["available"]) or "None"
    missing_fields = ", ".join(readiness["missing"]) or "None"

    model_line = ""
    if model_status == "Active" and model_metrics:
        algo = (model_bundle or {}).get("model_name", "Best selected model")
        model_line = (
            f"\n- **Active model:** {active_model_name} ({algo}) — "
            f"R² = {model_metrics.get('R2', 'N/A')}, MAE = {model_metrics.get('MAE', 'N/A')} MPa, "
            f"trained on {model_metrics.get('train_rows', 'N/A')} experimental rows"
        )
    else:
        model_line = f"\n- **Model status:** {model_status} — no active prediction model for this property/subsystem yet."

    uploaded_line = ""
    if n_uploaded_rows > 0:
        uploaded_line = f"\n- Uploaded evidence: **{n_uploaded_rows} rows** (decision support)"

    reason_lines = "\n".join(f"- {r.strip()}" for r in str(reasons).split("|") if r.strip())
    profile_gates = profile.get("validation_gates", [])
    if profile_gates:
        gate_lines = "\n".join(f"{i+1}. **{g}**" for i, g in enumerate(profile_gates))
    else:
        gate_lines = "\n".join(f"{i+1}. **{g['gate']}** — {g['detail']}" for i, g in enumerate(VALIDATION_GATES))

    density_flag = ""
    if row.get("_density_variance_flag", False):
        density_flag = (
            "\n> Weight saving neutral: flat density in current evidence. "
            "Upload lightweight alternatives for meaningful comparison."
        )

    registry_note = ""
    if registry_df is not None and not registry_df.empty:
        waiting = registry_df[registry_df["status"] == "Waiting for evidence"]["model_name"].tolist()
        if waiting:
            registry_note = f"\n- Waiting models: {', '.join(waiting)}"

    return f"""# MatIntel Decision Report

## Selected Subsystem
**{subsystem_val}** — {profile.get('scoring_focus', 'Property-aware screening')}

## Use Case
**{use_case}**

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
| Recommendation Basis | {rec_basis} |

## Evidence Source Provenance
- Public reference: matminer steel_strength ({n_reference_rows} experimental steel alloys, trust 95/100){uploaded_line}{model_line}{registry_note}

## Evidence Coverage
- **Available important fields:** {available_fields}
- **Missing important fields:** {missing_fields}
- **Coverage:** {readiness['coverage_pct']}% — {readiness['readiness']}
- **Public data note:** {profile.get('coverage_note', '—')}

## Suitability Score
**{score}** · Weight saving: **{ws}**
{density_flag}

### Score Dimensions Used
{dims_used}

### Score Dimensions Neutral (missing/flat evidence)
{dims_neutral}

### Score Breakdown
| Component | Value | Weight |
|-----------|-------|--------|
{score_rows}

## Rationale
{reason_lines}

## Recommended Validation Tests
{gate_lines}

## Model Registry
- Model for screening: **{active_model_name}** — status: **{model_status}**
- Property basis: {prop_basis}

## Next Action
Prioritise **{name}** for lab validation under the **{subsystem_val}** programme.

---
*MatIntel is property-aware, not one-score-fits-all. Screening only — not final engineering approval.*
"""
