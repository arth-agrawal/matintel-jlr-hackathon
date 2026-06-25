"""MatIntel — Governed Material Intelligence for JLR Engineering.

Single-command entry: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_pipeline import (
    load_or_create_unified, load_public_reference,
    DEMO_ENRICHMENT_COLS, QUARANTINED_DEMO_FIELDS,
)
from src.data_assumptions import format_field_display, is_assumption_field, ASSUMPTION_NOTE
from src.modeling import (
    load_or_train_model, load_or_train_models, load_all_trained_models,
    predict_with_interval, MODEL_TRAINING_SPECS,
)
from src.confidence import confidence_score
from src.recommender import (
    recommend_materials, USE_CASE_PRESETS,
    STRENGTH_MODES, compute_dynamic_threshold, get_scoring_dimensions,
)
from src.report_generator import build_report, VALIDATION_GATES
from src.schema_mapper import (
    suggest_schema_mapping, apply_schema_mapping,
    STANDARD_FIELDS, SOURCE_TYPE_PROFILES, ALL_SOURCE_TYPES, esc,
)
from src.public_sources import (
    PUBLIC_DATASET_REGISTRY, load_jarvis_dft, load_trusted_public_bundle,
    load_matminer_extra_datasets, summarize_sources, jarvis_property_coverage,
    auto_load_trusted_reference, build_jarvis_cache,
    CACHE_JARVIS_500, CACHE_JARVIS_5000, CACHE_MATBENCH_COMBINED,
)
from src.evidence_cockpit import (
    build_all_subsystem_cards, compute_hero_metrics, build_coverage_table,
    render_subsystem_card_html, render_evidence_flow_html,
)
from src.upload_workflow import (
    UPLOAD_GUIDELINES, SUBSYSTEM_COLUMN_GUIDANCE, SUBSYSTEM_TEMPLATE_COLUMNS,
    template_download_bytes, enrich_mapping_with_units,
    critical_mappings_unresolved,
    ingest_approved_evidence, compute_ml_eligibility, DENSITY_UNIT_OPTIONS,
)
from src.model_registry import detect_trainable_targets, get_active_models, get_trainable_next_models, MODEL_SPECS
from src.subsystem_profiles import SUBSYSTEM_PROFILES, ALL_SUBSYSTEMS, get_subsystem_readiness
from src.evidence_coverage import (
    filter_subsystem_rows, apply_subsystem_filters, format_display_table,
    property_coverage_matrix, subsystem_evidence_dashboard, empty_state_info,
    get_available_filters, get_model_activation_info, SUBSYSTEM_CHART_FIELD,
    SUBSYSTEM_FILTER_FIELDS, field_has_data,
)

MAX_UPLOAD_ROWS = 5_000

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="MatIntel", page_icon="◆", layout="wide")

# ── Premium CSS ──────────────────────────────────────────────────────────────

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --racing-green: #1B4332; --green-mid: #2D6A4F; --green-light: #40916C;
    --graphite: #2B2D30; --charcoal: #3D3F43;
    --ivory: #FAF9F6; --warm-white: #F5F3EF;
    --gold: #B08D57; --gold-muted: #C4A97D;
    --text-primary: #1A1A1A; --text-secondary: #5A5A5A;
    --border-light: #E0DCD4; --border-mid: #C8C3B8;
    --red-muted: #C0392B; --amber-muted: #D4A017; --green-status: #27AE60;
}
.stApp { font-family: 'Inter', -apple-system, sans-serif; }
.hero-block {
    background: linear-gradient(135deg, var(--racing-green) 0%, var(--green-mid) 100%);
    border-radius: 10px; padding: 28px 32px 22px; margin-bottom: 20px; color: white;
}
.hero-block h1 { font-size: 2rem; font-weight: 700; margin: 0 0 2px; letter-spacing: -0.5px; color: white; }
.hero-block .subtitle { font-size: 0.95rem; color: var(--gold-muted); font-weight: 500; margin: 0 0 8px; }
.hero-block .tagline { font-size: 0.82rem; color: rgba(255,255,255,0.75); margin: 0; }
.metric-strip { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.metric-chip { background: white; border: 1px solid var(--border-light); border-radius: 8px;
    padding: 10px 16px; flex: 1; min-width: 130px; text-align: center; }
.metric-chip .mv { font-size: 1.3rem; font-weight: 700; color: var(--racing-green); margin: 0; }
.metric-chip .ml { font-size: 0.7rem; font-weight: 500; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.5px; margin: 0; }
.src-card { border: 1px solid var(--border-light); border-radius: 8px; padding: 16px; background: white; }
.src-card h4 { margin: 0 0 8px; font-size: 0.95rem; font-weight: 600; color: var(--racing-green); }
.src-card table { width: 100%; font-size: 0.82rem; border-collapse: collapse; }
.src-card td { padding: 3px 0; color: var(--text-secondary); }
.src-card td:last-child { font-weight: 600; color: var(--text-primary); text-align: right; }
.src-card.active { border-left: 3px solid var(--green-light); }
.src-card.pending { border-left: 3px solid var(--gold); }
.src-card.empty { border-left: 3px solid var(--border-mid); opacity: 0.55; }
.empty-state { border: 1px solid var(--border-light); border-radius: 10px; padding: 24px;
    background: linear-gradient(180deg, var(--warm-white) 0%, white 100%); margin: 12px 0; }
.empty-state h3 { margin: 0 0 8px; color: var(--racing-green); font-size: 1.1rem; }
.empty-state p { margin: 4px 0; font-size: 0.85rem; color: var(--text-secondary); }
.empty-state ul { margin: 8px 0; padding-left: 18px; font-size: 0.82rem; color: var(--text-secondary); }
.pill { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.72rem;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }
.pill-green { background: #E8F5E9; color: #2E7D32; }
.pill-amber { background: #FFF8E1; color: #F57F17; }
.pill-red { background: #FFEBEE; color: #C62828; }
.pill-grey { background: #ECEFF1; color: #546E7A; }
.passport { border: 1px solid var(--border-light); border-radius: 8px; padding: 14px 16px;
    background: white; margin-bottom: 10px; }
.passport-header { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: var(--gold); margin: 0 0 8px; padding-bottom: 4px;
    border-bottom: 1px solid var(--border-light); }
.passport table { width: 100%; font-size: 0.82rem; border-collapse: collapse; }
.passport td { padding: 3px 0; }
.passport td:first-child { color: var(--text-secondary); width: 50%; }
.passport td:last-child { font-weight: 500; color: var(--text-primary); text-align: right; }
.rec-card { border: 1px solid var(--border-light); border-radius: 10px; padding: 18px;
    background: white; position: relative; }
.rec-card.rank-1 { border-top: 3px solid var(--gold); }
.rec-card.rank-2 { border-top: 3px solid #9E9E9E; }
.rec-card.rank-3 { border-top: 3px solid #8D6E63; }
.rec-rank { font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--gold); margin: 0 0 4px; }
.rec-name { font-size: 0.95rem; font-weight: 600; color: var(--text-primary);
    margin: 0 0 8px; line-height: 1.2; }
.rec-score { font-size: 1.5rem; font-weight: 700; color: var(--racing-green); margin: 0; }
.rec-score-label { font-size: 0.7rem; color: var(--text-secondary); margin: 0 0 8px; }
.rec-meta { font-size: 0.78rem; color: var(--text-secondary); margin: 2px 0; }
.rec-meta strong { color: var(--text-primary); }
.qg-row { display: flex; align-items: center; padding: 4px 0; font-size: 0.82rem; }
.qg-dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
.qg-ok { background: var(--green-status); }
.qg-miss { background: var(--amber-muted); }
.qg-label { color: var(--text-primary); }
.qg-status { margin-left: auto; font-size: 0.75rem; color: var(--text-secondary); }
.sec-header { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: var(--green-mid); margin: 20px 0 10px;
    padding-bottom: 4px; border-bottom: 2px solid var(--green-light); }
.subsys-card { border: 1px solid var(--border-light); border-radius: 10px; padding: 14px 16px;
    background: white; min-height: 280px; margin-bottom: 8px; }
.subsys-card h4 { margin: 0 0 6px; font-size: 0.88rem; font-weight: 600; color: var(--racing-green); line-height: 1.25; }
.subsys-card .card-num { font-size: 1.4rem; font-weight: 700; color: var(--racing-green); }
.subsys-card .card-rows { margin: 0 0 8px; font-size: 0.78rem; color: var(--text-secondary); }
.subsys-card .card-meta { margin: 3px 0; font-size: 0.74rem; color: var(--text-secondary); line-height: 1.35; }
.subsys-card .card-list { margin: 4px 0; padding-left: 16px; font-size: 0.72rem; }
.subsys-card .card-cta { font-size: 0.72rem; color: var(--gold); font-weight: 600; margin-top: 6px; }
.callout-premium { border: 1px solid var(--gold-muted); border-radius: 10px; padding: 16px 20px;
    background: linear-gradient(135deg, #FFFDF8 0%, white 100%); margin: 12px 0; }
.flow-section { display: flex; align-items: stretch; gap: 8px; margin: 16px 0; flex-wrap: wrap; }
.flow-col { flex: 1; min-width: 180px; border: 1px solid var(--border-light); border-radius: 8px;
    padding: 12px; background: white; }
.flow-title { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;
    color: var(--gold); margin-bottom: 8px; }
.flow-node { font-size: 0.78rem; padding: 6px 8px; margin: 4px 0; background: var(--warm-white);
    border-radius: 6px; border-left: 3px solid var(--green-light); }
.flow-node.flow-sub { border-left-color: var(--gold); }
.flow-node.flow-out { border-left-color: var(--racing-green); }
.flow-arrow { display: flex; align-items: center; font-size: 1.2rem; color: var(--green-mid); font-weight: 700; }
section[data-testid="stSidebar"] { background: var(--graphite) !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
section[data-testid="stSidebar"] .stMarkdown strong { color: var(--gold-muted) !important; }
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading material data …")
def cached_data():
    return load_or_create_unified()

@st.cache_data(show_spinner="Preparing training universe …")
def cached_training_unified():
    """Steel + cached reference rows used for property-specific model training."""
    _, steel = load_or_create_unified()
    ref, _ = auto_load_trusted_reference()
    if ref.empty:
        cached_ref = load_public_reference()
        ref = cached_ref if not cached_ref.empty else pd.DataFrame()
    if ref.empty:
        return steel
    return pd.concat([steel, ref], ignore_index=True)

@st.cache_resource(show_spinner="Loading prediction models …")
def cached_models(_rows: int):
    """Load saved models from bootstrap; train only if none exist on disk."""
    trained = load_all_trained_models()
    if trained:
        return trained
    return load_or_train_models(cached_training_unified())

try:
    raw, unified_base = cached_data()
except Exception as e:
    st.error(f"**Data loading failed:** {e}")
    st.info("Run `python scripts/bootstrap_data.py` to initialise data.")
    st.stop()

try:
    training_unified = cached_training_unified()
    all_models = cached_models(len(training_unified))
    model_bundle = all_models.get("structural_yield_strength")
    if model_bundle is None:
        model_bundle = load_or_train_model(unified_base)
        all_models["structural_yield_strength"] = model_bundle
except Exception as e:
    st.error(f"**Model training failed:** {e}")
    st.stop()

# ── Session state ────────────────────────────────────────────────────────────

if "uploaded_rows" not in st.session_state:
    st.session_state["uploaded_rows"] = pd.DataFrame()
if "upload_confirmations" not in st.session_state:
    st.session_state["upload_confirmations"] = {}
if "last_ingest_summary" not in st.session_state:
    st.session_state["last_ingest_summary"] = None
if "public_reference_rows" not in st.session_state:
    ref_df, ref_msgs = auto_load_trusted_reference()
    if ref_df.empty:
        cached_ref = load_public_reference()
        st.session_state["public_reference_rows"] = cached_ref if not cached_ref.empty else pd.DataFrame()
    else:
        st.session_state["public_reference_rows"] = ref_df
    st.session_state["reference_load_messages"] = ref_msgs
    st.session_state["reference_load_meta"] = {
        "jarvis_cached": CACHE_JARVIS_500.exists() or CACHE_JARVIS_5000.exists(),
        "jarvis_5000": CACHE_JARVIS_5000.exists(),
        "matbench_cached": CACHE_MATBENCH_COMBINED.exists(),
    }

ref_meta = st.session_state.get(
    "reference_load_meta",
    {
        "jarvis_cached": CACHE_JARVIS_500.exists() or CACHE_JARVIS_5000.exists(),
        "jarvis_5000": CACHE_JARVIS_5000.exists(),
        "matbench_cached": CACHE_MATBENCH_COMBINED.exists(),
    },
)

parts = [unified_base]
if not st.session_state["public_reference_rows"].empty:
    parts.append(st.session_state["public_reference_rows"])
if not st.session_state["uploaded_rows"].empty:
    parts.append(st.session_state["uploaded_rows"])
unified = pd.concat(parts, ignore_index=True) if len(parts) > 1 else unified_base.copy()

n_uploaded = len(st.session_state["uploaded_rows"])
n_public_ref = len(st.session_state["public_reference_rows"])
n_jarvis = int(
    st.session_state["public_reference_rows"]["source_dataset"].astype(str).str.startswith("jarvis").sum()
) if n_public_ref > 0 and "source_dataset" in st.session_state["public_reference_rows"].columns else 0
n_matbench = int(
    st.session_state["public_reference_rows"]["source_dataset"].astype(str).str.startswith("matbench").sum()
) if n_public_ref > 0 and "source_dataset" in st.session_state["public_reference_rows"].columns else 0
M = model_bundle["metrics"]
MODEL_ALGO = model_bundle.get("model_name", "RandomForestRegressor")
registry_df = detect_trainable_targets(unified)
trainable_next = get_trainable_next_models(unified)
active_models = get_active_models(unified)
hero = compute_hero_metrics(
    unified, registry_df, all_models,
    n_steel=int(unified_base["source_dataset"].eq("matminer_steel_strength").sum())
    if "source_dataset" in unified_base.columns else len(unified_base),
)
subsystem_cards = build_all_subsystem_cards(unified, registry_df)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fv(val, suffix="", fallback="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    return f"{esc(str(val))}{suffix}"


def _fv_field(row, field, suffix="", fallback="—"):
    """Format a passport field with assumption / validation pills."""
    return format_field_display(row, field, suffix=suffix)


def _decision_status(row):
    trust = row.get("source_trust_score", 0)
    if trust is None or (isinstance(trust, float) and np.isnan(trust)):
        trust = 0
    completeness = row.get("data_completeness_score", 0)
    if completeness is None or (isinstance(completeness, float) and np.isnan(completeness)):
        completeness = 0
    # Fatigue/corrosion are validation gates, not prerequisites
    if trust >= 80 and completeness >= 50:
        return "Ready for validation", "pill-green"
    elif trust >= 50 and completeness >= 30:
        return "Needs review", "pill-amber"
    return "Insufficient evidence", "pill-red"


def render_material_passport(row, pred=None, conf=None):
    """Render grouped Material Passport cards with HTML-escaped values."""
    status_text, status_class = _decision_status(row)
    ml_flag = str(row.get("used_for_ml_training", False)).lower() in ("true", "1")
    subsystem = esc(str(row.get("application_subsystem", "—")))

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Identity</div>
    <table>
    <tr><td>Material Name</td><td>{_fv(row.get('material_name'))}</td></tr>
    <tr><td>Family</td><td>{_fv(row.get('material_family', row.get('family')))}</td></tr>
    <tr><td>Subsystem</td><td>{subsystem}</td></tr>
    <tr><td>Source</td><td>{_fv(row.get('source_dataset'))}</td></tr>
    <tr><td>Source Type</td><td>{_fv(row.get('source_type'))}</td></tr>
    <tr><td>Source Trust</td><td>{_fv(row.get('source_trust_score'), '/100')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Performance</div>
    <table>
    <tr><td>Yield Strength</td><td>{_fv(row.get('yield_strength_mpa'), ' MPa')}</td></tr>
    <tr><td>Tensile Strength</td><td>{_fv(row.get('ultimate_tensile_strength_mpa'), ' MPa')}</td></tr>
    <tr><td>Density</td><td>{_fv_field(row, 'density_g_cm3', ' g/cm³')}</td></tr>
    <tr><td>Young's Modulus</td><td>{_fv_field(row, 'youngs_modulus_gpa', ' GPa')}</td></tr>
    <tr><td>Elongation</td><td>{_fv(row.get('elongation_percent'), '%')}</td></tr>
    <tr><td>Bulk Modulus</td><td>{_fv(row.get('bulk_modulus_gpa'), ' GPa')}</td></tr>
    <tr><td>Band Gap</td><td>{_fv(row.get('band_gap_ev'), ' eV')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Sustainability</div>
    <table>
    <tr><td>Recycled Content</td><td>{_fv_field(row, 'recycled_content_percent', '%')}</td></tr>
    <tr><td>CO2 Index</td><td>{_fv_field(row, 'co2_index', '/100')}</td></tr>
    <tr><td>CO2 / kg</td><td>{_fv_field(row, 'co2_kg_per_kg', ' kg')}</td></tr>
    <tr><td>Recyclability</td><td>{_fv_field(row, 'recyclability_score', '/100')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    fatigue_v = row.get("fatigue_strength_mpa")
    fatigue_ok = fatigue_v is not None and not (isinstance(fatigue_v, float) and np.isnan(fatigue_v))
    corrosion_v = row.get("corrosion_resistance_score")
    corrosion_ok = corrosion_v is not None and not (isinstance(corrosion_v, float) and np.isnan(corrosion_v))

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Risk & Governance</div>
    <table>
    <tr><td>Supplier Risk</td><td>{_fv_field(row, 'supplier_risk_score', '/100')}</td></tr>
    <tr><td>Critical Material Risk</td><td>{_fv_field(row, 'critical_material_risk_score', '/100')}</td></tr>
    <tr><td>ML Eligible</td><td>{'Yes' if ml_flag else 'No'}</td></tr>
    <tr><td>Fatigue Data</td><td>{'Available' if fatigue_ok else '<span class="pill pill-amber">Validation needed</span>'}</td></tr>
    <tr><td>Corrosion Data</td><td>{'Available' if corrosion_ok else '<span class="pill pill-amber">Validation needed</span>'}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Decision Status</div>
    <p style="margin:4px 0;"><span class="pill {status_class}">{status_text}</span></p>
    </div>""", unsafe_allow_html=True)


def render_quality_gate(row, subsystem: str = "Structural / Chassis"):
    """Compact data quality gate checklist for selected subsystem."""
    profile = SUBSYSTEM_PROFILES.get(subsystem, SUBSYSTEM_PROFILES.get("Structural / Chassis"))
    if not profile:
        return
    fields = profile["important_fields"]
    html = ""
    for field in fields:
        label = field.replace("_", " ").title()
        val = row.get(field)
        ok = val is not None and not (isinstance(val, float) and np.isnan(val))
        if is_assumption_field(row, field):
            display = _fv_field(row, field)
        elif ok:
            display = _fv(val)
        else:
            display = '<span class="pill pill-amber">Validation needed</span>'
        dot = "qg-ok" if ok and not is_assumption_field(row, field) else "qg-miss"
        status = "Available" if ok and not is_assumption_field(row, field) else (
            "Assumption / validation needed" if is_assumption_field(row, field) else "Validation needed"
        )
        html += f'<div class="qg-row"><span class="qg-dot {dot}"></span><span class="qg-label">{esc(label)}</span><span class="qg-status">{status}</span></div>'
    st.markdown(f'<div class="passport"><div class="passport-header">Data Quality Gate — {esc(subsystem)}</div>{html}</div>', unsafe_allow_html=True)


def _render_source_card(key: str, info: dict, card_class: str = "active"):
    ml_mode = info.get("ml_mode", "ML training" if info.get("used_for_ml_training") else "reference screening")
    targets = ", ".join(info.get("trainable_targets", [])[:3])
    st.markdown(
        f"""<div class="src-card {card_class}">
        <h4>{esc(info['name'])}</h4>
        <table>
        <tr><td>Domain</td><td>{esc(info['application_subsystem'])}</td></tr>
        <tr><td>Source Type</td><td>{esc(info['source_type'])}</td></tr>
        <tr><td>Trust</td><td>{info['trust_score']} / 100</td></tr>
        <tr><td>Mode</td><td>{esc(str(ml_mode))}</td></tr>
        <tr><td>Targets</td><td>{esc(targets)}</td></tr>
        <tr><td>Status</td><td>{esc(info.get('status', 'optional'))}</td></tr>
        </table></div>""",
        unsafe_allow_html=True,
    )


def render_empty_state(subsystem: str):
    info = empty_state_info(subsystem, unified, registry_df)
    sources_html = "".join(f"<li>{esc(s)}</li>" for s in info["trusted_sources"])
    fields_html = "".join(f"<li>{esc(f.replace('_', ' '))}</li>" for f in info["expected_fields"][:8])
    registry_html = "".join(
        f"<li><strong>{esc(r['model_name'])}</strong> — {esc(r['status'])}</li>"
        for r in info.get("registry_rows", [])[:4]
    )
    st.markdown(
        f"""<div class="empty-state">
        <h3>Waiting for subsystem evidence</h3>
        <p><strong>{esc(subsystem)}</strong></p>
        <p>{esc(info['activation_hint'])}</p>
        <p><strong>Trusted sources that can activate this subsystem:</strong></p>
        <ul>{sources_html}</ul>
        <p><strong>Expected fields:</strong></p>
        <ul>{fields_html}</ul>
        <p><strong>Model registry:</strong></p>
        <ul>{registry_html or '<li>No models registered yet</li>'}</ul>
        </div>""",
        unsafe_allow_html=True,
    )
    st.info("Upload engineer-reviewed evidence in the Evidence Intake tab, or load optional public reference datasets.")


def render_passport_for_subsystem(row, subsystem: str):
    """Property-aware passport sections for selected subsystem."""
    profile = SUBSYSTEM_PROFILES.get(subsystem, {})
    fields = profile.get("important_fields", [])
    status_text, status_class = _decision_status(row)
    ml_flag = str(row.get("used_for_ml_training", False)).lower() in ("true", "1")
    source_type = str(row.get("source_type", ""))

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Identity & Provenance</div>
    <table>
    <tr><td>Material</td><td>{_fv(row.get('material_name'))}</td></tr>
    <tr><td>Subsystem</td><td>{esc(subsystem)}</td></tr>
    <tr><td>Source</td><td>{_fv(row.get('source_dataset'))}</td></tr>
    <tr><td>Source Type</td><td>{_fv(source_type)}</td></tr>
    <tr><td>Source Trust</td><td>{_fv(row.get('source_trust_score'), '/100')}</td></tr>
    <tr><td>ML Eligible</td><td>{'Yes' if ml_flag else 'No'}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    perf_rows = ""
    for field in fields:
        label = field.replace("_", " ").title()
        val = row.get(field)
        ok = val is not None and not (isinstance(val, float) and np.isnan(val))
        if is_assumption_field(row, field):
            display = _fv_field(row, field)
        elif ok:
            display = _fv(val)
        else:
            display = '<span class="pill pill-amber">Validation needed</span>'
        perf_rows += f"<tr><td>{esc(label)}</td><td>{display}</td></tr>"

    assumption_note = row.get("data_assumption_notes")
    if assumption_note and not (isinstance(assumption_note, float) and np.isnan(assumption_note)):
        perf_rows += (
            f'<tr><td>Data Assumption Notes</td><td>{_fv(assumption_note)}</td></tr>'
        )

    st.markdown(
        f"""<div class="passport">
        <div class="passport-header">Subsystem Properties — {esc(subsystem)}</div>
        <table>{perf_rows}</table></div>""",
        unsafe_allow_html=True,
    )

    if source_type == "computed_database":
        st.caption("Computed reference only — not experimental validation.")

    st.markdown(
        f"""<div class="passport"><div class="passport-header">Decision Status</div>
        <p style="margin:4px 0;"><span class="pill {status_class}">{status_text}</span></p></div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### MatIntel")
    st.caption("Governed Material Intelligence")
    st.divider()
    st.markdown(
        "**L1** Public reference data  \n"
        "**L2** Evidence upload + review  \n"
        "**L3** Prediction + uncertainty  \n"
        "**L4** JLR fit scoring  \n"
        "**L5** Validation report"
    )
    st.divider()
    st.markdown(
        f"**Sources:** {hero['evidence_sources_active']}  \n"
        f"**Materials:** {hero['unified_materials']}  \n"
        f"**Trained models:** {hero['active_trained_models']} "
        f"({hero['experimental_trained_models']} exp + {hero['computed_reference_trained_models']} computed)"
    )
    st.divider()
    st.markdown(f"**Best model:** {MODEL_ALGO}  \nR² **{M['R2']}** · MAE **{M['MAE']}** MPa")
    st.divider()
    st.caption("Screening only. Not final engineering approval.")

# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""<div class="hero-block">
<h1>MatIntel</h1>
<p class="subtitle">Governed Material Intelligence for JLR Engineering</p>
<p class="tagline">All unified records are used in the intelligence layer for passport, search, and scoring. MatIntel trains separate property-specific models instead of forcing one universal model across unrelated targets. Computed-reference models are screening tools — validation required before engineering release.</p>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="metric-strip">
<div class="metric-chip"><p class="mv">{hero['evidence_sources_active']}</p><p class="ml">Evidence Sources Active</p></div>
<div class="metric-chip"><p class="mv">{hero['unified_materials']}</p><p class="ml">Unified Material Records</p></div>
<div class="metric-chip"><p class="mv">{hero['experimental_ml_rows']}</p><p class="ml">Experimental Training Rows</p></div>
<div class="metric-chip"><p class="mv">{hero['computed_reference_rows']}</p><p class="ml">Computed/Reference Records</p></div>
<div class="metric-chip"><p class="mv">{hero['active_trained_models']}</p><p class="ml">Active Trained Models</p></div>
<div class="metric-chip"><p class="mv">{hero['experimental_trained_models']}</p><p class="ml">Experimental Trained Models</p></div>
<div class="metric-chip"><p class="mv">{hero['computed_reference_trained_models']}</p><p class="ml">Computed-Reference Trained</p></div>
<div class="metric-chip"><p class="mv">{hero['waiting_for_evidence']}</p><p class="ml">Waiting for Evidence</p></div>
</div>""", unsafe_allow_html=True)
st.caption(
    "Default universe loads from cache. Heavy public datasets can be refreshed below in Evidence Intake. "
    "All records participate in the intelligence layer; each model trains only on rows with valid targets and features."
)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Evidence Intake", "Material Passport Library",
    "Predictive Model", "JLR Fit Scoring", "Validation Report",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EVIDENCE INTAKE
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="sec-header">Subsystem Intelligence Map</div>', unsafe_allow_html=True)
    st.caption(
        "Trusted reference data auto-loads at startup. Counts distinguish tagged rows, "
        "inferred reference coverage, and experimental training rows."
    )

    jarvis_5000 = ref_meta.get("jarvis_5000", CACHE_JARVIS_5000.exists())
    if not jarvis_5000 and n_jarvis <= 500:
        st.markdown(
            """<div class="callout-premium">
            <strong>Broader trusted reference cache not built yet</strong><br>
            <span style="font-size:0.85rem;color:#5A5A5A;">
            A 500-row JARVIS sample may be loaded. Build the 5,000-row cache for the full default universe
            (steel + JARVIS + cached Matbench).
            </span></div>""",
            unsafe_allow_html=True,
        )
        if st.button("Build 5,000-row trusted reference cache", key="build_5000_cache"):
            with st.spinner("Building JARVIS 5,000-row cache (may take a few minutes)…"):
                ref_df, msgs = build_jarvis_cache(target_rows=5000, timeout_seconds=300)
                if CACHE_MATBENCH_COMBINED.exists():
                    mb_df, _ = load_matminer_extra_datasets(require_cache=True, matbench_light_only=True)
                    if not mb_df.empty:
                        ref_df = pd.concat([ref_df, mb_df], ignore_index=True) if not ref_df.empty else mb_df
            if not ref_df.empty:
                st.session_state["public_reference_rows"] = ref_df
                st.session_state["reference_load_meta"] = {
                    "jarvis_cached": True,
                    "jarvis_5000": CACHE_JARVIS_5000.exists(),
                    "matbench_cached": CACHE_MATBENCH_COMBINED.exists(),
                }
                st.session_state["reference_load_messages"] = msgs
                st.success(" | ".join(msgs))
                st.rerun()
            else:
                st.warning(" | ".join(msgs))

    if n_public_ref > 0:
        auto_msgs = st.session_state.get("reference_load_messages", [])
        tier = "5,000-row" if jarvis_5000 else "500-row sample"
        load_note = f"{n_public_ref} reference records auto-loaded ({tier} JARVIS"
        if n_matbench:
            load_note += f" + {n_matbench} matbench cached"
        load_note += ")"
        if auto_msgs:
            st.caption(f"✓ {load_note}")

    card_rows = [subsystem_cards[i : i + 4] for i in range(0, len(subsystem_cards), 4)]
    for row_cards in card_rows:
        cols = st.columns(4)
        for col, card in zip(cols, row_cards):
            with col:
                st.markdown(render_subsystem_card_html(card), unsafe_allow_html=True)

    st.markdown('<div class="sec-header">Evidence Sources → Subsystems → Models &amp; Decisions</div>', unsafe_allow_html=True)
    st.markdown(
        render_evidence_flow_html(
            unified, registry_df, n_jarvis, n_matbench, n_uploaded, len(unified_base),
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-header">Source-to-Subsystem Coverage</div>', unsafe_allow_html=True)
    coverage_table = build_coverage_table(unified)
    st.dataframe(coverage_table, use_container_width=True, hide_index=True)
    st.caption(
        "Tagged = explicit application_subsystem. Inferred = property-matched reference rows "
        "(not hard-tagged). Training = experimental ML-eligible rows."
    )

    # ── Guided upload center ──
    st.markdown('<div class="sec-header">Upload Reviewed Evidence Source</div>', unsafe_allow_html=True)
    st.markdown(UPLOAD_GUIDELINES)

    with st.expander("Subsystem column guidance", expanded=False):
        for sub, cols in SUBSYSTEM_COLUMN_GUIDANCE.items():
            st.markdown(f"**{sub}:** `{cols}`")

    # Templates
    st.markdown("**Download CSV templates** (headers + example row marked EXAMPLE_DO_NOT_INGEST)")
    tmpl_cols = st.columns(4)
    template_names = list(SUBSYSTEM_TEMPLATE_COLUMNS.keys())
    for i, tname in enumerate(template_names):
        with tmpl_cols[i % 4]:
            st.download_button(
                label=tname[:28] + ("…" if len(tname) > 28 else ""),
                data=template_download_bytes(tname),
                file_name=f"matintel_template_{tname.replace(' / ', '_').replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key=f"dl_tmpl_{i}",
            )

    uploaded_file = st.file_uploader("Upload reviewed evidence CSV", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as ex:
            st.error(f"Could not read CSV: {ex}")
            upload_df = None

        if upload_df is not None and not upload_df.empty:
            if len(upload_df) > MAX_UPLOAD_ROWS:
                st.warning(f"Truncated to {MAX_UPLOAD_ROWS} rows.")
                upload_df = upload_df.head(MAX_UPLOAD_ROWS)

            st.markdown("#### Source classification")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                chosen_source_type = st.selectbox(
                    "Source type", ALL_SOURCE_TYPES,
                    index=ALL_SOURCE_TYPES.index("supplier_sheet"),
                    key="upload_src_type",
                )
            with sc2:
                subsystem_options = ["infer from file if present"] + ALL_SUBSYSTEMS
                chosen_subsystem = st.selectbox(
                    "Application subsystem", subsystem_options, key="upload_subsystem",
                )
            with sc3:
                engineer_reviewed = st.checkbox("Engineer reviewed", key="upload_eng_reviewed")

            infer_sub = chosen_subsystem == "infer from file if present"
            app_sub = None if infer_sub else chosen_subsystem
            profile = SOURCE_TYPE_PROFILES[chosen_source_type]

            mapping_df = enrich_mapping_with_units(upload_df, suggest_schema_mapping(upload_df))
            needs_confirm = mapping_df[mapping_df["needs_confirmation"] == True]  # noqa: E712

            st.markdown("#### Schema mapping review")
            options = ["ignore"] + STANDARD_FIELDS
            overrides: dict[str, str] = {}
            unit_factors: dict[str, float] = {}
            confirmations: dict[str, bool] = st.session_state.get("upload_confirmations", {})

            upload_cols = list(upload_df.columns)
            for i in range(0, len(upload_cols), 3):
                row_cols = st.columns(3)
                for j, col_name in enumerate(upload_cols[i:i + 3]):
                    suggested_row = mapping_df[mapping_df["uploaded_column"] == col_name]
                    default = suggested_row["suggested_field"].values[0] if len(suggested_row) > 0 else "ignore"
                    default_idx = options.index(default) if default in options else 0
                    with row_cols[j]:
                        overrides[col_name] = st.selectbox(
                            col_name, options, index=default_idx, key=f"map_{col_name}",
                        )

            if not needs_confirm.empty:
                st.markdown("#### Mapping Review & Questions — Needs confirmation")
                st.dataframe(
                    needs_confirm[[
                        "uploaded_column", "suggested_field", "confidence",
                        "detected_unit", "review_question",
                    ]],
                    use_container_width=True, hide_index=True,
                )
                for _, nrow in needs_confirm.iterrows():
                    col = nrow["uploaded_column"]
                    norm = col.lower().strip()
                    if "density" in norm and nrow["suggested_field"] == "density_g_cm3":
                        unit_choice = st.radio(
                            f"Unit for '{col}'",
                            DENSITY_UNIT_OPTIONS,
                            key=f"unit_{col}",
                            horizontal=True,
                        )
                        unit_factors[col] = 0.001 if "kg/m" in unit_choice else 1.0
                    confirmations[col] = st.checkbox(
                        f"Confirmed by engineer: {col} → {overrides.get(col, nrow['suggested_field'])}",
                        key=f"confirm_{col}",
                    )

            engineer_notes = st.text_area(
                "Engineer review notes (stored in source metadata)",
                key="upload_eng_notes", placeholder="e.g. Supplier datasheet Rev 3, lab batch 2024-Q2",
            )

            temp_mapped = apply_schema_mapping(
                upload_df, overrides, source_type=chosen_source_type,
                application_subsystem=app_sub, infer_subsystem=infer_sub,
                engineer_reviewed=engineer_reviewed,
                unit_factors=unit_factors,
            )
            ml_assessment = compute_ml_eligibility(temp_mapped, chosen_source_type, engineer_reviewed)

            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Source Trust", f"{profile['trust']}/100")
            ac2.metric("Mapping Confidence", f"{mapping_df['confidence'].mean():.0f}%")
            ac3.metric("ML Eligible (auto)", "Yes" if ml_assessment["eligible"] else "No")
            ac4.metric("Engineer Reviewed", "Yes" if engineer_reviewed else "No")
            st.caption(f"ML assessment: {ml_assessment['reason']}")

            if chosen_source_type == "unknown":
                st.warning("Unknown source type — confirm classification before ingest.")

            unresolved = critical_mappings_unresolved(mapping_df, overrides, confirmations)
            if unresolved:
                st.warning(f"Unresolved mappings need engineer confirmation: {', '.join(unresolved)}")

            if st.button("Approve and ingest evidence", type="primary", key="approve_ingest"):
                if chosen_source_type == "unknown" and not engineer_reviewed:
                    st.error("Unknown source requires engineer review before ingest.")
                elif unresolved:
                    st.error("Confirm all flagged mappings before ingest.")
                else:
                    mapped, summary = ingest_approved_evidence(
                        upload_df, overrides,
                        source_type=chosen_source_type,
                        application_subsystem=app_sub,
                        infer_subsystem=infer_sub,
                        engineer_reviewed=engineer_reviewed,
                        mapping_notes=f"Confirmed mappings: {len(overrides)} columns",
                        engineer_review_notes=engineer_notes,
                        unit_factors=unit_factors,
                    )
                    st.session_state["uploaded_rows"] = pd.concat(
                        [st.session_state["uploaded_rows"], mapped], ignore_index=True,
                    ) if not st.session_state["uploaded_rows"].empty else mapped
                    st.session_state["last_ingest_summary"] = summary
                    st.session_state["upload_confirmations"] = confirmations
                    st.success(f"Ingested {summary['rows_ingested']} rows · ML eligible: {summary['ml_eligible_rows']}")
                    st.rerun()

    if st.session_state.get("last_ingest_summary"):
        s = st.session_state["last_ingest_summary"]
        st.markdown("#### Last ingestion result")
        st.json(s)

    # ── Below the fold: optional loaders & diagnostics ──
    with st.expander("Optional reference cache controls (advanced)", expanded=False):
        st.caption("Not required for demo — startup auto-loads from disk cache.")
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            if st.button("Build 5,000-row JARVIS cache", key="ref_build_5000"):
                with st.spinner("Building…"):
                    ref_df, msgs = build_jarvis_cache(target_rows=5000, timeout_seconds=300)
                if not ref_df.empty:
                    existing = st.session_state.get("public_reference_rows", pd.DataFrame())
                    non_jarvis = existing[
                        ~existing["source_dataset"].astype(str).str.startswith("jarvis")
                    ] if not existing.empty and "source_dataset" in existing.columns else pd.DataFrame()
                    combined = pd.concat([f for f in [ref_df, non_jarvis] if not f.empty], ignore_index=True)
                    st.session_state["public_reference_rows"] = combined
                    st.success(" | ".join(msgs))
                    st.rerun()
                else:
                    st.warning(" | ".join(msgs))
        with lc2:
            if st.button("Load broader reference set", key="ref_broad"):
                ref_df, msgs = load_trusted_public_bundle(
                    mode="broad", include_jarvis=True, include_matbench=True, matbench_light_only=True,
                )
                if not ref_df.empty:
                    st.session_state["public_reference_rows"] = ref_df
                    st.success(" | ".join(msgs))
                    st.rerun()
        with lc3:
            st.caption("⚠ Slow")
            if st.button("Load matbench extras", key="ref_matbench"):
                mb_df, mmsg = load_matminer_extra_datasets(
                    matbench_light_only=False, timeout_seconds=300,
                )
                existing = st.session_state.get("public_reference_rows", pd.DataFrame())
                keep = existing[
                    ~existing["source_dataset"].astype(str).str.startswith("matbench")
                ] if not existing.empty and "source_dataset" in existing.columns else pd.DataFrame()
                parts = [f for f in [keep, mb_df] if not f.empty]
                if parts:
                    st.session_state["public_reference_rows"] = pd.concat(parts, ignore_index=True)
                    st.success(mmsg)
                    st.rerun()
                else:
                    st.warning(mmsg)
        with lc4:
            st.caption("⚠ Very slow")
            if st.button("Load full JARVIS", key="ref_full_jarvis"):
                jarvis_df, jmsg = load_jarvis_dft(max_rows=None, cache=True)
                existing = st.session_state.get("public_reference_rows", pd.DataFrame())
                non_jarvis = existing[
                    ~existing["source_dataset"].astype(str).str.startswith("jarvis")
                ] if not existing.empty and "source_dataset" in existing.columns else pd.DataFrame()
                parts = [f for f in [jarvis_df, non_jarvis] if not f.empty]
                if parts:
                    st.session_state["public_reference_rows"] = pd.concat(parts, ignore_index=True)
                    st.success(jmsg)
                    st.rerun()

    with st.expander("Dataset registry & diagnostics", expanded=False):
        reg_cols = st.columns(2)
        ref_summary = summarize_sources(st.session_state["public_reference_rows"])
        for idx, (key, info) in enumerate(PUBLIC_DATASET_REGISTRY.items()):
            with reg_cols[idx % 2]:
                card_class = "empty"
                status = info.get("status", "optional")
                if key == "matminer_steel_strength":
                    card_class, status = "active", f"active — {len(unified_base)} experimental rows"
                elif key == "jarvis_dft_3d" and n_jarvis > 0:
                    card_class, status = "pending", f"auto-loaded — {n_jarvis} rows"
                elif key == "matminer_matbench_extra" and n_matbench > 0:
                    card_class, status = "pending", f"cached — {n_matbench} rows"
                elif key == "engineer_reviewed_upload" and n_uploaded > 0:
                    card_class, status = "pending", f"active — {n_uploaded} uploaded rows"
                _render_source_card(key, {**info, "status": status}, card_class)

        coverage_dash = subsystem_evidence_dashboard(unified, registry_df)
        st.dataframe(coverage_dash, use_container_width=True, hide_index=True)
        st.dataframe(property_coverage_matrix(unified), use_container_width=True)
        st.info(ASSUMPTION_NOTE)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATERIAL PASSPORT LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="sec-header">Material Passport Library</div>', unsafe_allow_html=True)
    st.caption("Select a subsystem first — filters and columns adapt to relevant properties.")

    subsystem_filter = st.selectbox(
        "Application subsystem", ALL_SUBSYSTEMS,
        index=ALL_SUBSYSTEMS.index("Structural / Chassis"), key="lib_subsystem",
    )

    with st.expander("Evidence coverage for selected subsystem"):
        dash_row = subsystem_evidence_dashboard(unified, registry_df)
        row = dash_row[dash_row["subsystem"] == subsystem_filter]
        if not row.empty:
            st.dataframe(row, use_container_width=True, hide_index=True)

    sub_df = filter_subsystem_rows(unified, subsystem_filter)
    with_sliders, missing_filter_fields = get_available_filters(sub_df, subsystem_filter)

    min_filters: dict[str, float] = {}
    if with_sliders:
        st.markdown("**Property filters** (only fields with evidence)")
        n_cols = min(3, len(with_sliders))
        filter_cols = st.columns(n_cols)
        for i, field in enumerate(with_sliders):
            label = field.replace("_", " ").title()
            col_data = pd.to_numeric(sub_df[field], errors="coerce").dropna()
            with filter_cols[i % n_cols]:
                if not col_data.empty:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    if max_val <= min_val:
                        st.caption(f"{label}: {min_val:.2g} (flat — no filter)")
                        min_filters[field] = min_val
                    else:
                        min_filters[field] = st.slider(
                            label, min_val, max_val, min_val, key=f"filter_{subsystem_filter}_{field}",
                        )

    if missing_filter_fields:
        with st.expander("Missing evidence / validation needed"):
            for f in missing_filter_fields:
                st.caption(f"• {f.replace('_', ' ')} — no labelled data in current evidence")

    filtered = apply_subsystem_filters(sub_df, subsystem_filter, min_filters)

    if filtered.empty:
        render_empty_state(subsystem_filter)
    else:
        display_df = format_display_table(filtered, subsystem_filter)
        st.dataframe(display_df.head(200), use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-header">Material Passport</div>', unsafe_allow_html=True)
        passport_id = st.selectbox("Select material", filtered["material_id"].tolist(), key="passport_select")
        if passport_id:
            p_row = filtered[filtered["material_id"] == passport_id].iloc[0]
            p1, p2 = st.columns(2)
            with p1:
                render_passport_for_subsystem(p_row, subsystem_filter)
                readiness = get_subsystem_readiness(p_row, subsystem_filter)
                st.markdown(f"""<div class="passport">
                <div class="passport-header">Subsystem Readiness</div>
                <table>
                <tr><td>Coverage</td><td>{readiness['coverage_pct']}%</td></tr>
                <tr><td>Status</td><td>{esc(readiness['readiness'])}</td></tr>
                <tr><td>Available</td><td>{len(readiness['available'])} fields</td></tr>
                <tr><td>Missing</td><td>{len(readiness['missing'])} fields</td></tr>
                </table></div>""", unsafe_allow_html=True)
            with p2:
                chart_field = SUBSYSTEM_CHART_FIELD.get(subsystem_filter)
                if chart_field and field_has_data(filtered, chart_field):
                    fig = px.histogram(
                        filtered.dropna(subset=[chart_field]),
                        x=chart_field, nbins=30,
                        labels={chart_field: chart_field.replace("_", " ").title()},
                    )
                    fig.update_layout(
                        bargap=0.05, height=400, margin=dict(l=20, r=20, t=30, b=20),
                        title_text=f"{chart_field.replace('_', ' ').title()} Distribution",
                        title_font_size=13,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No chartable data for {chart_field or 'primary property'} in this subsystem yet.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE MODEL
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="sec-header">Property-Specific Model Registry</div>', unsafe_allow_html=True)
    st.caption(
        "MatIntel trains one model per target/property — not a single universal model across all 8,312 records. "
        "All rows remain available for passport, search, and scoring."
    )

    active_df = registry_df[registry_df["status"].isin(["Active", "Active computed-reference"])]
    trainable_df = registry_df[registry_df["status"].isin([
        "Trainable", "Trainable next", "Computed-reference trainable",
    ])]
    waiting_df = registry_df[registry_df["status"].isin(["Waiting for evidence", "Insufficient data"])]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active trained", len(all_models))
    m2.metric("Experimental", hero["experimental_trained_models"])
    m3.metric("Computed-reference", hero["computed_reference_trained_models"])
    m4.metric("Waiting", len(waiting_df))

    st.dataframe(
        registry_df[[
            "model_name", "subsystem", "target", "eligible_rows",
            "status", "model_family", "reason",
        ]],
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="sec-header">Active Trained Models</div>', unsafe_allow_html=True)

    if not all_models:
        st.warning(
            "No trained models on disk. Run `python scripts/bootstrap_data.py` to train "
            "structural and computed-reference models."
        )
    else:
        for model_key, bundle in all_models.items():
            spec = MODEL_SPECS.get(model_key, {})
            family = bundle.get("model_family", spec.get("model_family", "experimental"))
            is_computed = family == "computed_reference"
            limitation = bundle.get(
                "limitation_label",
                "Computed-reference screening model — validation required before engineering release"
                if is_computed else "Measured-data prediction",
            )
            unit = bundle.get("unit", "MPa" if model_key == "structural_yield_strength" else "")
            algo = bundle.get("selected_algorithm", bundle.get("model_name", "—"))
            metrics = bundle.get("metrics", {})
            target_col = bundle.get("target", spec.get("target", ""))
            sources = ", ".join(bundle.get("source_datasets", [])) or "—"

            st.markdown(
                f"""<div class="passport" style="margin-bottom:16px;">
                <div class="passport-header">{esc(model_key.replace('_', ' ').title())}</div>
                <table>
                <tr><td>Target</td><td>{esc(target_col)}</td></tr>
                <tr><td>Model family</td><td>{esc(family.replace('_', ' '))}</td></tr>
                <tr><td>Source basis</td><td>{esc(sources)}</td></tr>
                <tr><td>Rows used</td><td>{bundle.get('rows', metrics.get('train_rows', '—'))}</td></tr>
                <tr><td>Algorithm</td><td>{esc(algo)}</td></tr>
                <tr><td>R² / MAE / RMSE</td><td>{metrics.get('R2', '—')} / {metrics.get('MAE', '—')} / {metrics.get('RMSE', '—')}</td></tr>
                <tr><td>Limitation</td><td>{esc(limitation)}</td></tr>
                </table></div>""",
                unsafe_allow_html=True,
            )

            test_actuals = bundle.get("test_actuals", [])
            test_preds = bundle.get("test_predictions", [])
            if test_actuals and test_preds:
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=test_actuals, y=test_preds,
                    mode="markers", marker=dict(size=7, color="#1B4332", opacity=0.65),
                    name="Test holdout",
                ))
                lo = min(min(test_actuals), min(test_preds))
                hi = max(max(test_actuals), max(test_preds))
                fig_sc.add_trace(go.Scatter(
                    x=[lo, hi], y=[lo, hi],
                    mode="lines", line=dict(dash="dash", color="#B08D57"),
                    name="Perfect prediction",
                ))
                fig_sc.update_layout(
                    title=f"Actual vs Predicted — {target_col}",
                    xaxis_title=f"Actual ({unit})".strip(),
                    yaxis_title=f"Predicted ({unit})".strip(),
                    height=320, margin=dict(l=40, r=20, t=40, b=40),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sc, use_container_width=True, key=f"scatter_{model_key}")

            importance = bundle.get("feature_importance", {})
            if importance:
                imp_df = pd.DataFrame([
                    {"feature": k, "importance": v}
                    for k, v in sorted(importance.items(), key=lambda x: -x[1])[:10]
                ])
                fig_imp = px.bar(
                    imp_df, x="importance", y="feature", orientation="h",
                    title="Top feature importance",
                    color_discrete_sequence=["#2D6A4F"],
                )
                fig_imp.update_layout(
                    height=280, margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_{model_key}")

            # Interactive prediction for this model
            if target_col in unified.columns:
                candidates = unified[unified[target_col].notna()]["material_id"].tolist()
                if candidates:
                    mat_id = st.selectbox(
                        f"Inspect material — {model_key}",
                        candidates[:500],
                        key=f"pred_mat_{model_key}",
                    )
                    row = unified[unified["material_id"] == mat_id].iloc[0]
                    pred_info = predict_with_interval(bundle, row)
                    conf = confidence_score(row, pred_info, bundle)
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("Predicted", f"{pred_info['prediction']} {unit}".strip())
                    if pred_info["actual"] is not None:
                        pc2.metric("Actual", f"{pred_info['actual']} {unit}".strip())
                    else:
                        pc2.metric("Actual", "—")
                    pc3.metric("Prediction confidence", f"{conf['confidence']}%")
            st.divider()

    st.markdown('<div class="sec-header">Models Waiting for Evidence</div>', unsafe_allow_html=True)
    if waiting_df.empty:
        st.caption("No models currently waiting.")
    else:
        for _, wrow in waiting_df.iterrows():
            activation = get_model_activation_info(wrow["model_name"], registry_df)
            st.markdown(
                f"""<div class="empty-state" style="padding:14px;">
                <p><strong>{esc(wrow['model_name'])}</strong> — {esc(wrow['status'])}</p>
                <p>{esc(activation['how_to_activate'])}</p>
                </div>""",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — JLR FIT SCORING
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="sec-header">JLR Fit Scoring</div>', unsafe_allow_html=True)
    st.caption("Property-aware weighted score — only dimensions with evidence for the selected subsystem.")

    r1, r2, r3 = st.columns(3)
    with r1:
        use_case = st.selectbox("Use case preset", list(USE_CASE_PRESETS.keys()), key="rec_usecase")
    with r2:
        sel_subsystem = st.selectbox(
            "Subsystem context", ALL_SUBSYSTEMS,
            index=ALL_SUBSYSTEMS.index(USE_CASE_PRESETS[use_case]["subsystem"]),
            key="rec_subsystem",
        )
    with r3:
        min_trust = st.number_input("Min source_trust_score", value=0, min_value=0, max_value=100, key="rec_trust")

    sub_profile = SUBSYSTEM_PROFILES.get(sel_subsystem, {})
    st.caption(f"**Scoring focus:** {sub_profile.get('scoring_focus', '—')}")
    st.caption(f"**Coverage:** {sub_profile.get('coverage_note', '—')}")

    min_strength = 0.0
    baseline_density = 7.85
    if sel_subsystem in ("Structural / Chassis", "General Material Reuse"):
        sr1, sr2, sr3 = st.columns(3)
        with sr1:
            strength_mode = st.selectbox("Strength target", list(STRENGTH_MODES.keys()), index=1, key="rec_mode")
        with sr2:
            baseline_density = st.number_input("Baseline density", value=7.85, min_value=0.1, key="rec_density")
        with sr3:
            sub_df_sc = filter_subsystem_rows(unified, sel_subsystem)
            candidates_ys = sub_df_sc["yield_strength_mpa"].dropna() if "yield_strength_mpa" in sub_df_sc.columns else pd.Series(dtype=float)
            if strength_mode == "Custom":
                min_strength = st.number_input("Custom min yield (MPa)", value=250, min_value=0, key="rec_custom_str")
            elif not candidates_ys.empty:
                min_strength = compute_dynamic_threshold(candidates_ys, strength_mode)
                st.caption(f"Dynamic threshold: **{min_strength} MPa** ({strength_mode})")
            else:
                st.caption("No yield strength evidence — strength scoring neutral.")

    ranked = recommend_materials(
        unified, subsystem=sel_subsystem, min_strength=min_strength,
        baseline_density=baseline_density, use_case=use_case, min_trust=min_trust,
    )

    if ranked.empty:
        render_empty_state(sel_subsystem)
    else:
        st.markdown('<div class="sec-header">Data Quality Gate — Top Candidate</div>', unsafe_allow_html=True)
        top_row = ranked.iloc[0]
        gq1, gq2 = st.columns([1, 2])
        with gq1:
            render_quality_gate(top_row, sel_subsystem)
        with gq2:
            with st.expander("Scoring dimensions for this subsystem"):
                dims = get_scoring_dimensions(sel_subsystem)
                w_df = pd.DataFrame([
                    {"Dimension": label, "Weight": f"{w:.0%}", "Column": key}
                    for key, label, w in dims
                ])
                st.dataframe(w_df, use_container_width=True, hide_index=True)
            st.caption(f"**Used:** {top_row.get('_score_dims_used', '—')}")
            st.caption(f"**Neutral (missing/flat):** {top_row.get('_score_dims_neutral', '—')}")

        if ranked.iloc[0].get("_density_variance_flag", False):
            st.info(
                "Weight saving neutral: flat density in current evidence; "
                "upload lightweight candidate evidence for aluminium/magnesium/composite comparison."
            )

        st.markdown('<div class="sec-header">Top Candidates</div>', unsafe_allow_html=True)
        top3 = ranked.head(3)
        card_cols = st.columns(3)
        sub_score_cols = get_scoring_dimensions(sel_subsystem)

        for idx, (card_col, (_, mat)) in enumerate(zip(card_cols, top3.iterrows())):
            with card_col:
                rank_labels = ["1st", "2nd", "3rd"]
                mat_name = esc(str(mat.get("material_name", "N/A"))[:30])
                status_text, status_class = _decision_status(mat)
                score_val = mat.get("suitability_score", 0)

                st.markdown(
                    f"""<div class="rec-card rank-{idx+1}">
                    <p class="rec-rank">{rank_labels[idx]} Candidate</p>
                    <p class="rec-name">{mat_name}</p>
                    <p class="rec-score">{score_val}</p>
                    <p class="rec-score-label">Screening Score / 100</p>
                    <p style="margin:4px 0;"><span class="pill {status_class}">{status_text}</span></p>
                    <p class="rec-meta"><strong>Subsystem:</strong> {esc(sel_subsystem)}</p>
                    <p class="rec-meta"><strong>Source:</strong> {esc(str(mat.get('source_type', '—')))} · Trust {_fv(mat.get('source_trust_score'))}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                for r_code in str(mat.get("reason_codes", "")).split("|"):
                    r_code = r_code.strip()
                    if r_code.startswith("+"):
                        st.success(r_code, icon="✅")
                    elif r_code.startswith("-"):
                        st.warning(r_code, icon="⚠️")
                    elif r_code.startswith("~"):
                        st.info(r_code)

                with st.expander("Sub-scores"):
                    for label, col, _w in sub_score_cols:
                        val = mat.get(col, 50)
                        if isinstance(val, float) and np.isnan(val):
                            val = 50
                        st.progress(min(val / 100, 1.0), text=f"{label}: {round(val, 1)}")

        with st.expander("Full ranking table"):
            rank_cols = ["material_id", "material_name", "source_type", "source_trust_score", "suitability_score"]
            rank_cols += [c for c, _, _ in sub_score_cols if c in ranked.columns]
            available_rank = [c for c in rank_cols if c in ranked.columns]
            st.dataframe(ranked[available_rank].head(30), use_container_width=True, hide_index=True)

        fig = px.bar(
            ranked.head(10), x="material_id", y="suitability_score",
            hover_data=["source_trust_score"],
            labels={"suitability_score": "Score", "material_id": ""},
        )
        fig.update_layout(
            bargap=0.15, height=320, margin=dict(l=40, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title_text=f"Top 10 — {sel_subsystem}", title_font_size=13,
        )
        fig.update_traces(marker_color="#2D6A4F")
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — VALIDATION REPORT
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="sec-header">Validation Report</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        report_usecase = st.selectbox("Use case preset", list(USE_CASE_PRESETS.keys()), key="rpt_uc")
    with rc2:
        rpt_subsystem = st.selectbox(
            "Subsystem", ALL_SUBSYSTEMS,
            index=ALL_SUBSYSTEMS.index(USE_CASE_PRESETS[report_usecase]["subsystem"]),
            key="rpt_subsystem",
        )

    report_strength = 0.0
    report_density = 7.85
    if rpt_subsystem in ("Structural / Chassis", "General Material Reuse"):
        rc3, rc4 = st.columns(2)
        with rc3:
            report_strength = st.number_input("Min yield (MPa)", value=250, min_value=0, key="rpt_str")
        with rc4:
            report_density = st.number_input("Baseline density", value=7.85, min_value=0.1, key="rpt_den")

    ranked_report = recommend_materials(
        unified, subsystem=rpt_subsystem, min_strength=report_strength,
        baseline_density=report_density, use_case=report_usecase,
    )

    if ranked_report.empty:
        render_empty_state(rpt_subsystem)
    else:
        report_options = ranked_report.head(20)
        report_labels = [
            f"#{i+1} {r['material_id']} — Score {r.get('suitability_score', '—')}"
            for i, (_, r) in enumerate(report_options.iterrows())
        ]
        selected_label = st.selectbox("Select material", report_labels, key="rpt_pick")
        selected_idx = report_labels.index(selected_label)
        selected = report_options.iloc[selected_idx]

        active_models = get_active_models(unified)
        active_name = "structural_yield_strength"
        model_status = "Active"
        if active_models:
            active_name = active_models[0].get("model_name", active_name)
            model_status = active_models[0].get("status", "Active")

        report = build_report(
            selected, min_strength=report_strength, use_case=report_usecase,
            model_metrics=model_bundle["metrics"],
            n_reference_rows=len(unified_base), n_uploaded_rows=n_uploaded,
            active_model_name=active_name, model_status=model_status,
            model_bundle=model_bundle, subsystem=rpt_subsystem,
            registry_df=registry_df,
        )

        st.markdown(report)

        with st.expander("Model Registry"):
            st.dataframe(
                registry_df[["model_name", "subsystem", "target", "status", "reason"]],
                use_container_width=True, hide_index=True,
            )

        with st.expander(f"Property Coverage — {rpt_subsystem}"):
            matrix = property_coverage_matrix(unified)
            if rpt_subsystem in matrix.index:
                st.dataframe(matrix.loc[[rpt_subsystem]].T, use_container_width=True)

        st.markdown('<div class="sec-header">Validation Gate Tracker</div>', unsafe_allow_html=True)
        profile = SUBSYSTEM_PROFILES.get(rpt_subsystem, {})
        for gate in profile.get("validation_gates", [g["gate"] for g in VALIDATION_GATES]):
            st.checkbox(gate, value=False, key=f"gate_{gate}")

        st.download_button("Download Report (.md)", data=report, file_name="matintel_report.md", mime="text/markdown")
