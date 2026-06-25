"""Subsystem evidence cockpit: intelligence map, flow diagram, hero metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.demo_ui import (
    MODEL_CARDS, SUBSYSTEM_UNLOCK, display_status, count_model_ready_rows, upload_template_count,
)
from src.evidence_coverage import (
    filter_subsystem_rows,
    field_has_data,
    subsystem_coverage_counts,
    SUBSYSTEM_TRUSTED_SOURCES,
    SOURCE_LABELS,
)
from src.subsystem_profiles import ALL_SUBSYSTEMS, SUBSYSTEM_PROFILES
from src.upload_workflow import SUBSYSTEM_COLUMN_GUIDANCE
from src.schema_mapper import esc
from src.modeling import load_all_trained_models, count_trained_models
from src.model_registry import get_trained_models_for_subsystem


def _model_display_name(model_key: str) -> str:
    return MODEL_CARDS.get(model_key, {}).get("title", model_key.replace("_", " ").title())


def _count_summary_line(subsystem: str, counts: dict[str, int]) -> str:
    if subsystem == "General Material Reuse":
        return f"{counts['relevant_rows']:,} materials in unified universe"

    if counts["training_rows"] > 0:
        return f"{counts['training_rows']:,} materials with measured training data"

    if counts["relevant_rows"] > 0:
        return f"{counts['relevant_rows']:,} materials with relevant properties"

    return "Ready for new evidence"


def subsystem_card_status(
    subsystem: str,
    counts: dict[str, int],
    active_models: list[str],
    n_fields_available: int,
    n_fields_total: int,
) -> str:
    if subsystem == "General Material Reuse":
        return "Full universe screening"
    if active_models:
        return "Active model available"
    if counts["relevant_rows"] > 0 and n_fields_available >= max(1, n_fields_total * 0.2):
        return "Reference coverage"
    return "Ready for new evidence"


def build_subsystem_card(subsystem: str, unified_df: pd.DataFrame, registry_df: pd.DataFrame) -> dict[str, Any]:
    profile = SUBSYSTEM_PROFILES[subsystem]
    counts = subsystem_coverage_counts(unified_df, subsystem)
    sub_df = filter_subsystem_rows(unified_df, subsystem)

    available_fields = [f for f in profile["important_fields"] if field_has_data(sub_df, f)]
    missing_fields = [f for f in profile["important_fields"] if f not in available_fields]

    trained_keys = get_trained_models_for_subsystem(subsystem)
    active = [_model_display_name(k) for k in trained_keys]

    sub_models = registry_df[
        registry_df["subsystem"].astype(str).str.contains(subsystem.split("/")[0].strip()[:12], case=False, na=False)
        | registry_df["subsystem"].astype(str).str.contains("General", na=False)
    ]
    trainable = sub_models[sub_models["status"].isin([
        "Trainable", "Trainable next", "Computed-reference trainable",
    ])]["model_name"].tolist()

    status = subsystem_card_status(
        subsystem, counts, trained_keys, len(available_fields), len(profile["important_fields"]),
    )

    return {
        "subsystem": subsystem,
        "counts": counts,
        "count_summary": _count_summary_line(subsystem, counts),
        "key_properties": available_fields[:5],
        "missing_properties": missing_fields[:4],
        "validation_gates": profile.get("validation_gates", [])[:3],
        "active_models": active,
        "trainable_targets": trainable[:2],
        "status": status,
        "unlock_hint": SUBSYSTEM_UNLOCK.get(subsystem, "Unlocks subsystem fit scoring after upload."),
        "column_guidance": SUBSYSTEM_COLUMN_GUIDANCE.get(subsystem, ""),
        "configured": True,
    }


def build_all_subsystem_cards(unified_df: pd.DataFrame, registry_df: pd.DataFrame) -> list[dict[str, Any]]:
    return [build_subsystem_card(s, unified_df, registry_df) for s in ALL_SUBSYSTEMS]


def build_coverage_table(unified_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subsystem in ALL_SUBSYSTEMS:
        c = subsystem_coverage_counts(unified_df, subsystem)
        rows.append({
            "subsystem": subsystem,
            "materials_matched": c["relevant_rows"],
            "with_training_data": c["training_rows"],
            "reference_matched": c["inferred_rows"],
        })
    return pd.DataFrame(rows)


def compute_hero_metrics(
    unified_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    trained_models: dict | None,
    n_steel: int,
) -> dict[str, Any]:
    if trained_models is None:
        trained_models = load_all_trained_models()
    counts = count_trained_models(trained_models)

    cards = build_all_subsystem_cards(unified_df, registry_df)
    n_subsystems_covered = sum(1 for c in cards if c["counts"]["relevant_rows"] > 0)

    avg_trust = float(round(unified_df["source_trust_score"].mean(), 1)) if "source_trust_score" in unified_df.columns else 0.0

    return {
        "unified_materials": len(unified_df),
        "active_trained_models": counts["active_trained_models"],
        "model_ready_rows": count_model_ready_rows(unified_df),
        "subsystems_covered": n_subsystems_covered,
        "upload_templates": upload_template_count(),
        "avg_data_trust": avg_trust,
        # kept for internal/sidebar
        "experimental_trained_models": counts["experimental_trained_models"],
        "computed_reference_trained_models": counts["computed_reference_trained_models"],
    }


def render_subsystem_card_html(card: dict[str, Any]) -> str:
    status = display_status(card["status"])
    pill = {
        "Active model available": "pill-green",
        "Reference coverage": "pill-grey",
        "Full universe screening": "pill-green",
    }.get(status, "pill-amber")

    keys = esc(", ".join(card["key_properties"]) or "Configured — add data to populate")
    active = esc(", ".join(card["active_models"]) or "Ready when data is added")
    summary = esc(card["count_summary"])
    unlock = esc(card.get("unlock_hint", ""))

    if card["counts"]["relevant_rows"] == 0:
        expected = esc(card["column_guidance"] or "See upload templates")
        extra = (
            f'<p class="card-meta">Templates, fields, validation checks, and upload review are configured.</p>'
            f'<p class="card-meta"><strong>Add:</strong> {expected}</p>'
            f'<p class="card-meta"><strong>Unlocks:</strong> {unlock}</p>'
            f'<p class="card-cta">Upload evidence below</p>'
        )
    else:
        extra = f'<p class="card-meta"><strong>Active models:</strong> {active}</p>'
        if card["missing_properties"]:
            extra += f'<p class="card-meta"><strong>Grow coverage:</strong> {esc(", ".join(card["missing_properties"]))}</p>'

    return (
        f'<div class="subsys-card">'
        f'<h4>{esc(card["subsystem"])}</h4>'
        f'<p class="card-rows">{summary}</p>'
        f'<p class="card-meta"><strong>Key properties:</strong> {keys}</p>'
        f'{extra}'
        f'<p style="margin-top:8px;"><span class="pill {pill}">{esc(status)}</span></p>'
        f'</div>'
    )


def render_evidence_flow_html(
    n_unified: int,
    n_models: int,
    n_uploaded: int,
) -> str:
    steps = [
        ("Trusted public data", "Steel · JARVIS · Matbench cache"),
        ("Material universe", f"{n_unified:,} unified records"),
        ("Property models", f"{n_models} trained models"),
        ("Recommendations", "Subsystem fit scoring"),
        ("Validation report", "Screening deliverable"),
    ]
    if n_uploaded:
        steps.insert(4, ("Your uploads", f"{n_uploaded} engineer-reviewed rows"))

    html = '<div class="flow-section">'
    for i, (title, detail) in enumerate(steps):
        if i > 0:
            html += '<div class="flow-arrow">→</div>'
        html += (
            f'<div class="flow-card">'
            f'<div class="flow-title">{esc(title)}</div>'
            f'<div class="flow-subtitle">{esc(detail)}</div>'
            f'</div>'
        )
    html += '</div>'
    return html
