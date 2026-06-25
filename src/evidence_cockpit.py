"""Subsystem evidence cockpit: intelligence map, flow diagram, hero metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evidence_coverage import (
    filter_subsystem_rows,
    field_has_data,
    subsystem_coverage_counts,
    SUBSYSTEM_TRUSTED_SOURCES,
    SOURCE_LABELS,
)
from src.subsystem_profiles import ALL_SUBSYSTEMS, SUBSYSTEM_PROFILES
from src.upload_workflow import SOURCE_ROUTE_HINTS, SUBSYSTEM_COLUMN_GUIDANCE
from src.schema_mapper import esc
from src.modeling import load_all_trained_models, count_trained_models
from src.model_registry import get_trained_models_for_subsystem


EVIDENCE_TYPE_LABELS = {
    "public_experimental": "Experimental",
    "experimental_test": "Experimental",
    "computed_database": "Computed-reference",
    "public_benchmark": "Public benchmark",
    "public_reference": "Public reference",
    "supplier_sheet": "Uploaded evidence",
    "sustainability_sheet": "Uploaded evidence",
    "procurement_sheet": "Uploaded evidence",
    "unknown": "Uploaded evidence",
}


def evidence_types_for_subsystem(df: pd.DataFrame, subsystem: str) -> list[str]:
    sub = filter_subsystem_rows(df, subsystem)
    if sub.empty or "source_type" not in sub.columns:
        return []
    types = {EVIDENCE_TYPE_LABELS.get(str(st), str(st)) for st in sub["source_type"].dropna().unique()}
    return sorted(types)


def _count_summary_line(subsystem: str, counts: dict[str, int]) -> str:
    tagged = counts["tagged_rows"]
    inferred = counts["inferred_rows"]
    training = counts["training_rows"]

    if subsystem == "General Material Reuse":
        return f"{counts['relevant_rows']} unified material records"

    if subsystem == "Structural / Chassis" and training > 0:
        parts = [f"{training} experimental training rows"]
        if inferred > 0:
            parts.append(f"{inferred} inferred reference rows")
        return " + ".join(parts)

    if subsystem == "Electronics / Thermal Interface" and inferred > 0 and tagged == 0:
        return f"{inferred} computed-reference rows with band gap / dielectric / thermal fields"

    if subsystem == "Battery Enclosure / Underbody" and inferred > 0:
        if tagged > 0:
            return f"{tagged} tagged + {inferred} computed-reference candidates"
        return f"{inferred} computed-reference candidates"

    if inferred > 0 and tagged > 0:
        return f"{tagged} tagged + {inferred} inferred reference rows"
    if inferred > 0:
        return f"{inferred} inferred reference rows"
    if tagged > 0:
        return f"{tagged} tagged rows"
    return "No coverage yet — upload evidence"


def subsystem_card_status(
    counts: dict[str, int],
    has_experimental: bool,
    has_computed: bool,
    active_models: list[str],
    n_fields_available: int,
    n_fields_total: int,
) -> str:
    if active_models and counts["training_rows"] > 0:
        return "Active model"
    if active_models and has_computed:
        return "Active computed-reference model"
    if has_computed and counts["relevant_rows"] > 0:
        return "Computed-reference coverage"
    if counts["relevant_rows"] > 0 and n_fields_available >= max(1, n_fields_total * 0.2):
        return "Partial coverage"
    return "Upload evidence needed"


def build_subsystem_card(subsystem: str, unified_df: pd.DataFrame, registry_df: pd.DataFrame) -> dict[str, Any]:
    profile = SUBSYSTEM_PROFILES[subsystem]
    counts = subsystem_coverage_counts(unified_df, subsystem)
    sub_df = filter_subsystem_rows(unified_df, subsystem)

    available_fields = [f for f in profile["important_fields"] if field_has_data(sub_df, f)]
    missing_fields = [f for f in profile["important_fields"] if f not in available_fields]

    has_experimental = False
    has_computed = False
    if "source_type" in sub_df.columns and len(sub_df) > 0:
        has_experimental = bool(sub_df["source_type"].isin(["public_experimental", "experimental_test"]).any())
        has_computed = bool(sub_df["source_type"].isin(["computed_database", "public_benchmark"]).any())

    sub_models = registry_df[
        registry_df["subsystem"].astype(str).str.contains(subsystem.split("/")[0].strip()[:12], case=False, na=False)
        | registry_df["subsystem"].astype(str).str.contains("General", na=False)
    ]
    trained_keys = get_trained_models_for_subsystem(subsystem)
    active = [k for k in trained_keys] or sub_models[sub_models["status"].isin([
        "Active", "Active computed-reference",
    ])]["model_name"].tolist()
    trainable = sub_models[sub_models["status"].isin([
        "Trainable", "Trainable next", "Computed-reference trainable",
    ])]["model_name"].tolist()
    waiting = sub_models[sub_models["status"] == "Waiting for evidence"]["model_name"].tolist()

    status = subsystem_card_status(
        counts, has_experimental, has_computed, active,
        len(available_fields), len(profile["important_fields"]),
    )

    return {
        "subsystem": subsystem,
        "counts": counts,
        "count_summary": _count_summary_line(subsystem, counts),
        "evidence_types": evidence_types_for_subsystem(unified_df, subsystem),
        "key_properties": available_fields[:5],
        "missing_properties": missing_fields[:4],
        "validation_gates": profile.get("validation_gates", [])[:3],
        "active_models": active,
        "trainable_targets": trainable[:2],
        "waiting_models": waiting[:2],
        "status": status,
        "upload_routes": SOURCE_ROUTE_HINTS.get(subsystem, [
            "Engineer-reviewed CSV upload",
            "Supplier technical datasheet",
            "Internal lab test export",
        ]),
        "column_guidance": SUBSYSTEM_COLUMN_GUIDANCE.get(subsystem, ""),
        "trusted_sources": [SOURCE_LABELS.get(s, s) for s in SUBSYSTEM_TRUSTED_SOURCES.get(subsystem, [])],
    }


def build_all_subsystem_cards(unified_df: pd.DataFrame, registry_df: pd.DataFrame) -> list[dict[str, Any]]:
    return [build_subsystem_card(s, unified_df, registry_df) for s in ALL_SUBSYSTEMS]


def build_coverage_table(unified_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subsystem in ALL_SUBSYSTEMS:
        c = subsystem_coverage_counts(unified_df, subsystem)
        rows.append({
            "subsystem": subsystem,
            "tagged_rows": c["tagged_rows"],
            "inferred_reference_rows": c["inferred_rows"],
            "training_rows": c["training_rows"],
            "relevant_total": c["relevant_rows"],
        })
    return pd.DataFrame(rows)


def compute_hero_metrics(
    unified_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    trained_models: dict | None,
    n_steel: int,
) -> dict[str, Any]:
    comp_types = {"computed_database", "public_benchmark"}

    sources_active = (
        int(unified_df["source_dataset"].nunique())
        if "source_dataset" in unified_df.columns else 1
    )
    ml_rows = int(
        unified_df.get("used_for_ml_training", pd.Series(dtype=bool))
        .astype(str).str.lower().isin(["true", "1"]).sum()
    ) if "used_for_ml_training" in unified_df.columns else n_steel

    computed_rows = int(unified_df["source_type"].isin(comp_types).sum()) if "source_type" in unified_df.columns else 0

    if trained_models is None:
        trained_models = load_all_trained_models()
    counts = count_trained_models(trained_models)

    trainable_next = registry_df[
        registry_df["status"].isin(["Trainable next", "Computed-reference trainable", "Trainable"])
    ]
    waiting = int(registry_df["status"].eq("Waiting for evidence").sum())

    cards = build_all_subsystem_cards(unified_df, registry_df)
    n_subsystems_covered = sum(1 for c in cards if c["counts"]["relevant_rows"] > 0)

    avg_trust = float(round(unified_df["source_trust_score"].mean(), 1)) if "source_trust_score" in unified_df.columns else 0.0

    return {
        "evidence_sources_active": sources_active,
        "unified_materials": len(unified_df),
        "experimental_ml_rows": ml_rows,
        "computed_reference_rows": computed_rows,
        "active_trained_models": counts["active_trained_models"],
        "experimental_trained_models": counts["experimental_trained_models"],
        "computed_reference_trained_models": counts["computed_reference_trained_models"],
        "trainable_next_targets": len(trainable_next),
        "waiting_for_evidence": waiting,
        "subsystems_with_coverage": n_subsystems_covered,
        "avg_source_trust": avg_trust,
    }


def render_subsystem_card_html(card: dict[str, Any]) -> str:
    status = card["status"]
    pill = {
        "Active model": "pill-green",
        "Active computed-reference model": "pill-green",
        "Computed-reference coverage": "pill-grey",
        "Partial coverage": "pill-amber",
    }.get(status, "pill-amber")

    ev_types = esc(", ".join(card["evidence_types"]) or "None yet")
    keys = esc(", ".join(card["key_properties"]) or "—")
    gates = esc(" · ".join(card["validation_gates"]) or "—")
    active = esc(", ".join(card["active_models"]) or "—")
    trainable = esc(", ".join(card["trainable_targets"]) or "—")
    summary = esc(card["count_summary"])

    extra = ""
    if card["counts"]["relevant_rows"] == 0:
        routes = "".join(f"<li>{esc(r)}</li>" for r in card["upload_routes"][:4])
        expected = esc(card["column_guidance"] or "See upload template")
        extra = (
            f'<p class="card-meta"><strong>Expected:</strong> {expected}</p>'
            f'<p class="card-meta"><strong>Route:</strong></p><ul class="card-list">{routes}</ul>'
            f'<p class="card-cta">Upload reviewed evidence below</p>'
        )
    elif card["missing_properties"]:
        extra = f'<p class="card-meta"><strong>Still needed:</strong> {esc(", ".join(card["missing_properties"]))}</p>'

    return (
        f'<div class="subsys-card">'
        f'<h4>{esc(card["subsystem"])}</h4>'
        f'<p class="card-rows">{summary}</p>'
        f'<p class="card-meta"><strong>Evidence:</strong> {ev_types}</p>'
        f'<p class="card-meta"><strong>Key properties:</strong> {keys}</p>'
        f'<p class="card-meta"><strong>Active model:</strong> {active}</p>'
        f'<p class="card-meta"><strong>Trainable:</strong> {trainable}</p>'
        f'<p class="card-meta"><strong>Gates:</strong> {gates}</p>'
        f'{extra}'
        f'<p style="margin-top:8px;"><span class="pill {pill}">{esc(status)}</span></p>'
        f'</div>'
    )


def render_evidence_flow_html(
    unified_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    n_jarvis: int,
    n_matbench: int,
    n_uploaded: int,
    n_steel: int,
) -> str:
    active = registry_df[registry_df["status"].isin(["Active", "Active computed-reference"])]
    trainable = registry_df[registry_df["status"].isin([
        "Trainable next", "Computed-reference trainable",
    ])]["model_name"].tolist()[:4]

    sources = [
        ("matminer steel_strength", f"{n_steel} experimental"),
        ("JARVIS-DFT", f"{n_jarvis} computed"),
        ("Matbench cache", f"{n_matbench} benchmark" if n_matbench else "optional"),
        ("NASA TPSX pathway", "CSV import"),
        ("Engineer upload", f"{n_uploaded} rows" if n_uploaded else "guided CSV"),
    ]

    subsystems = [
        "Structural · Electronics · Battery · General reuse",
        "Thermal · Interior · Tyres · Coatings → upload",
    ]
    outputs = (
        active["model_name"].tolist() if not active.empty else []
    ) + trainable + ["JLR fit scoring", "Validation report"]

    src_html = "".join(
        f'<div class="flow-node">{esc(a)}<br><small>{esc(b)}</small></div>' for a, b in sources
    )
    sub_html = "".join(f'<div class="flow-node flow-sub">{esc(s)}</div>' for s in subsystems)
    out_html = "".join(f'<div class="flow-node flow-out">{esc(o)}</div>' for o in outputs)

    return (
        '<div class="flow-section">'
        f'<div class="flow-col"><div class="flow-title">Evidence Sources</div>{src_html}</div>'
        '<div class="flow-arrow">→</div>'
        f'<div class="flow-col"><div class="flow-title">Subsystems</div>{sub_html}</div>'
        '<div class="flow-arrow">→</div>'
        f'<div class="flow-col"><div class="flow-title">Models &amp; Decisions</div>{out_html}</div>'
        '</div>'
    )
