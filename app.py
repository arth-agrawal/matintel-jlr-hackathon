"""MatIntel — Governed Material Intelligence for JLR Engineering.

Single-command entry: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_pipeline import load_or_create_unified, DEMO_ENRICHMENT_COLS
from src.modeling import load_or_train_model, predict_with_interval
from src.confidence import confidence_score
from src.recommender import (
    recommend_materials, USE_CASE_PRESETS,
    STRENGTH_MODES, compute_dynamic_threshold, get_scoring_dimensions,
)
from src.report_generator import build_report, VALIDATION_GATES
from src.schema_mapper import (
    suggest_schema_mapping, apply_schema_mapping, assess_ml_eligibility,
    STANDARD_FIELDS, SOURCE_TYPE_PROFILES, ALL_SOURCE_TYPES, esc,
)
from src.public_sources import PUBLIC_DATASET_REGISTRY, load_jarvis_dft_sample
from src.subsystem_profiles import SUBSYSTEM_PROFILES, ALL_SUBSYSTEMS, get_subsystem_readiness
from src.model_registry import detect_trainable_targets, get_active_models, MODEL_SPECS
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

@st.cache_resource(show_spinner="Loading prediction model …")
def cached_model(_unified):
    return load_or_train_model(_unified)

try:
    raw, unified_base = cached_data()
except Exception as e:
    st.error(f"**Data loading failed:** {e}")
    st.info("Run `python scripts/bootstrap_data.py` to initialise data.")
    st.stop()

try:
    model_bundle = cached_model(unified_base)
except Exception as e:
    st.error(f"**Model training failed:** {e}")
    st.stop()

# ── Session state ────────────────────────────────────────────────────────────

if "uploaded_rows" not in st.session_state:
    st.session_state["uploaded_rows"] = pd.DataFrame()
if "jarvis_rows" not in st.session_state:
    st.session_state["jarvis_rows"] = pd.DataFrame()

parts = [unified_base]
if not st.session_state["jarvis_rows"].empty:
    parts.append(st.session_state["jarvis_rows"])
if not st.session_state["uploaded_rows"].empty:
    parts.append(st.session_state["uploaded_rows"])
unified = pd.concat(parts, ignore_index=True) if len(parts) > 1 else unified_base.copy()

n_uploaded = len(st.session_state["uploaded_rows"])
n_jarvis = len(st.session_state["jarvis_rows"])
M = model_bundle["metrics"]
MODEL_ALGO = model_bundle.get("model_name", "RandomForestRegressor")
registry_df = detect_trainable_targets(unified)

source_datasets = unified["source_dataset"].nunique() if "source_dataset" in unified.columns else 1
ml_rows = int(
    (unified.get("used_for_ml_training", pd.Series(dtype=bool))
     .astype(str).str.lower().isin(["true", "1"])).sum()
)
avg_trust = round(unified["source_trust_score"].mean(), 1) if "source_trust_score" in unified.columns else 0

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fv(val, suffix="", fallback="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    return f"{esc(str(val))}{suffix}"


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
    <tr><td>Density</td><td>{_fv(row.get('density_g_cm3'), ' g/cm³')}</td></tr>
    <tr><td>Young's Modulus</td><td>{_fv(row.get('youngs_modulus_gpa'), ' GPa')}</td></tr>
    <tr><td>Elongation</td><td>{_fv(row.get('elongation_percent'), '%')}</td></tr>
    <tr><td>Bulk Modulus</td><td>{_fv(row.get('bulk_modulus_gpa'), ' GPa')}</td></tr>
    <tr><td>Band Gap</td><td>{_fv(row.get('band_gap_ev'), ' eV')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Sustainability</div>
    <table>
    <tr><td>Recycled Content</td><td>{_fv(row.get('recycled_content_percent'), '%')}</td></tr>
    <tr><td>CO2 Index</td><td>{_fv(row.get('co2_index'), '/100')}</td></tr>
    <tr><td>CO2 / kg</td><td>{_fv(row.get('co2_kg_per_kg'), ' kg')}</td></tr>
    <tr><td>Recyclability</td><td>{_fv(row.get('recyclability_score'), '/100')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    fatigue_v = row.get("fatigue_strength_mpa")
    fatigue_ok = fatigue_v is not None and not (isinstance(fatigue_v, float) and np.isnan(fatigue_v))
    corrosion_v = row.get("corrosion_resistance_score")
    corrosion_ok = corrosion_v is not None and not (isinstance(corrosion_v, float) and np.isnan(corrosion_v))

    st.markdown(f"""<div class="passport">
    <div class="passport-header">Risk & Governance</div>
    <table>
    <tr><td>Supplier Risk</td><td>{_fv(row.get('supplier_risk_score'), '/100')}</td></tr>
    <tr><td>Critical Material Risk</td><td>{_fv(row.get('critical_material_risk_score'), '/100')}</td></tr>
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
        label = field.replace("_", " ").replace("g cm3", "(g/cm³)").replace("mpa", "(MPa)").replace("gpa", "(GPa)").title()
        val = row.get(field)
        ok = val is not None and not (isinstance(val, float) and np.isnan(val))
        dot = "qg-ok" if ok else "qg-miss"
        status = "Available" if ok else "Validation needed"
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
        display = _fv(val) if ok else '<span class="pill pill-amber">Validation needed</span>'
        perf_rows += f"<tr><td>{esc(label)}</td><td>{display}</td></tr>"

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
    st.markdown(f"**Sources:** {source_datasets}  \n**Materials:** {len(unified)}  \n**ML rows:** {ml_rows}")
    st.divider()
    st.markdown(f"**Best model:** {MODEL_ALGO}  \nR² **{M['R2']}** · MAE **{M['MAE']}** MPa")
    st.divider()
    st.caption("Screening only. Not final engineering approval.")

# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""<div class="hero-block">
<h1>MatIntel</h1>
<p class="subtitle">Governed Material Intelligence for JLR Engineering</p>
<p class="tagline">MatIntel is property-aware, not one-score-fits-all. Structural materials are evaluated on strength/stiffness/fatigue/corrosion; electronics on thermal/electrical properties; interiors on VOC/fire/circularity; fluids on thermal/viscosity/toxicity. The model registry activates models only where eligible evidence exists.</p>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="metric-strip">
<div class="metric-chip"><p class="mv">{source_datasets}</p><p class="ml">Evidence Sources</p></div>
<div class="metric-chip"><p class="mv">{len(unified)}</p><p class="ml">Unified Materials</p></div>
<div class="metric-chip"><p class="mv">{ml_rows}</p><p class="ml">ML Eligible Rows</p></div>
<div class="metric-chip"><p class="mv">{M['R2']}</p><p class="ml">Model R²</p></div>
<div class="metric-chip"><p class="mv">{avg_trust}</p><p class="ml">Avg Source Trust</p></div>
</div>""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Evidence Intake", "Material Passport Library",
    "Predictive Model", "JLR Fit Scoring", "Validation Report",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EVIDENCE INTAKE
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="sec-header">Trusted Public Dataset Registry</div>', unsafe_allow_html=True)
    st.caption(
        "MatIntel does not assume one dataset covers every subsystem. "
        "Each source enters with domain, trust, and model eligibility."
    )

    reg_cols = st.columns(2)
    reg_items = list(PUBLIC_DATASET_REGISTRY.items())
    for idx, (key, info) in enumerate(reg_items):
        with reg_cols[idx % 2]:
            if key == "matminer_steel_strength":
                _render_source_card(key, {**info, "status": f"active — {len(unified_base)} rows loaded"}, "active")
            elif key == "jarvis_dft_3d_sample" and n_jarvis > 0:
                _render_source_card(key, {**info, "status": f"loaded — {n_jarvis} reference rows"}, "pending")
            elif key == "engineer_upload" and n_uploaded > 0:
                _render_source_card(key, {**info, "status": f"active — {n_uploaded} uploaded rows"}, "pending")
            else:
                card = "empty" if key not in ("matminer_steel_strength",) else "active"
                _render_source_card(key, info, card)

    st.markdown('<div class="sec-header">Evidence Coverage by Subsystem</div>', unsafe_allow_html=True)
    coverage_dash = subsystem_evidence_dashboard(unified, registry_df)
    st.dataframe(
        coverage_dash[[
            "subsystem", "materials", "fields_available", "fields_missing",
            "active_models", "waiting_models", "readiness",
        ]],
        use_container_width=True, hide_index=True,
    )

    with st.expander("Property Coverage Matrix"):
        matrix = property_coverage_matrix(unified)
        st.dataframe(matrix, use_container_width=True)
        st.caption("Cell values = count of non-null property values for materials tagged to that subsystem.")

    # JARVIS loader
    if n_jarvis == 0:
        if st.button("Load JARVIS-DFT sample (optional)"):
            with st.spinner("Loading JARVIS-DFT data..."):
                jarvis_df, jarvis_msg = load_jarvis_dft_sample(max_rows=500)
            if jarvis_df.empty:
                st.warning(f"JARVIS not available: {jarvis_msg}")
            else:
                st.session_state["jarvis_rows"] = jarvis_df
                st.success(jarvis_msg)
                st.rerun()
    else:
        st.info(f"JARVIS-DFT: {n_jarvis} reference rows loaded (computed, not for experimental ML).")

    with st.expander("Raw matminer data preview"):
        st.dataframe(raw.head(30), use_container_width=True)

    with st.expander("Unified schema preview"):
        preview_cols = [
            "material_id", "material_name", "material_family", "application_subsystem",
            "yield_strength_mpa", "density_g_cm3", "source_type", "source_trust_score",
            "used_for_ml_training", "data_completeness_score",
        ]
        available_preview = [c for c in preview_cols if c in unified.columns]
        st.dataframe(unified[available_preview].head(20), use_container_width=True)

    with st.expander("Demo enrichment fields"):
        demo_present = [c for c in DEMO_ENRICHMENT_COLS if c in unified.columns]
        st.caption("These fields use typical steel defaults for demonstration. Replace with measured values before engineering decisions.")
        st.code(", ".join(demo_present))

    # ── Evidence Upload ──
    st.markdown('<div class="sec-header">Evidence Source Upload</div>', unsafe_allow_html=True)
    st.caption("Deterministic, auditable schema mapping. No LLM used.")

    uploaded_file = st.file_uploader("Upload material evidence (CSV)", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as ex:
            st.error(f"Could not read CSV: {ex}")
            upload_df = None

        if upload_df is not None and not upload_df.empty:
            if len(upload_df) > MAX_UPLOAD_ROWS:
                st.warning(f"CSV has {len(upload_df)} rows — truncated to {MAX_UPLOAD_ROWS} for performance.")
                upload_df = upload_df.head(MAX_UPLOAD_ROWS)

            with st.expander("Raw uploaded data", expanded=True):
                st.dataframe(upload_df.head(15), use_container_width=True)

            src_col, info_col = st.columns([1, 2])
            with src_col:
                chosen_source_type = st.selectbox(
                    "Source type", ALL_SOURCE_TYPES,
                    index=ALL_SOURCE_TYPES.index("supplier_sheet"),
                    key="upload_src_type",
                )
            with info_col:
                profile = SOURCE_TYPE_PROFILES[chosen_source_type]
                st.caption(f"{profile['description']} — Trust: {profile['trust']}/100 · ML: {'Yes' if profile['ml_eligible'] else 'No'}")

            mapping_df = suggest_schema_mapping(upload_df)

            with st.expander("Schema mapping detail", expanded=True):
                st.dataframe(mapping_df, use_container_width=True, hide_index=True)
                options = ["ignore"] + STANDARD_FIELDS
                overrides: dict[str, str] = {}
                upload_cols = list(upload_df.columns)
                for i in range(0, len(upload_cols), 3):
                    row_cols = st.columns(3)
                    for j, col_name in enumerate(upload_cols[i:i+3]):
                        suggested_row = mapping_df[mapping_df["uploaded_column"] == col_name]
                        default = suggested_row["suggested_field"].values[0] if len(suggested_row) > 0 else "ignore"
                        default_idx = options.index(default) if default in options else 0
                        with row_cols[j]:
                            overrides[col_name] = st.selectbox(col_name, options, index=default_idx, key=f"map_{col_name}")

            temp_mapped = apply_schema_mapping(upload_df, overrides, source_type=chosen_source_type)
            ml_assessment = assess_ml_eligibility(temp_mapped, chosen_source_type)

            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Source Trust", f"{profile['trust']}/100")
            ac2.metric("Mapping Confidence", f"{mapping_df['confidence'].mean():.0f}%")
            ac3.metric("ML Eligible", "Yes" if ml_assessment["eligible"] else "No")
            ac4.metric("Engineer Reviewed", "Pending")

            if not ml_assessment["eligible"]:
                st.caption(f"ML: {ml_assessment['reason']}")

            if st.button("Ingest Evidence", type="primary"):
                mapped = apply_schema_mapping(upload_df, overrides, source_type=chosen_source_type)
                st.session_state["uploaded_rows"] = mapped
                st.session_state["upload_source_type"] = chosen_source_type
                st.success(f"Ingested {len(mapped)} rows (session only).")
                st.rerun()


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
    st.markdown('<div class="sec-header">Model Registry</div>', unsafe_allow_html=True)

    active_df = registry_df[registry_df["status"] == "Active"]
    trainable_df = registry_df[registry_df["status"] == "Trainable"]
    waiting_df = registry_df[registry_df["status"].isin(["Waiting for evidence", "Insufficient data"])]

    m1, m2, m3 = st.columns(3)
    m1.metric("Active", len(active_df))
    m2.metric("Trainable", len(trainable_df))
    m3.metric("Waiting", len(waiting_df))

    st.dataframe(
        registry_df[["model_name", "subsystem", "target", "eligible_rows", "status", "reason"]],
        use_container_width=True, hide_index=True,
    )

    model_options = list(MODEL_SPECS.keys())
    status_map = dict(zip(registry_df["model_name"], registry_df["status"]))
    model_labels = [f"{m} — {status_map.get(m, 'Unknown')}" for m in model_options]
    selected_model_idx = st.selectbox("Select model", range(len(model_options)),
                                        format_func=lambda i: model_labels[i], key="pred_model_select")
    selected_model = model_options[selected_model_idx]
    model_status = status_map.get(selected_model, "Unknown")
    activation = get_model_activation_info(selected_model, registry_df)

    st.caption(
        "Current active experimental model is trained on structural steel data. "
        "Waiting models need eligible labelled evidence — not errors."
    )

    if model_status == "Active" and selected_model == "structural_yield_strength":
        st.markdown('<div class="sec-header">Property Prediction — Yield Strength</div>', unsafe_allow_html=True)
        interval_label = model_bundle.get("interval_method", "tree_spread")
        st.caption(
            f"Best selected model: **{MODEL_ALGO}** · R² {M['R2']} · MAE {M['MAE']} MPa · "
            f"{M['train_rows']} training rows · Interval: {interval_label}"
        )

        pred_candidates = unified[unified["yield_strength_mpa"].notna()]["material_id"].tolist()
        if not pred_candidates:
            st.warning("No materials with yield strength data available for prediction.")
        else:
            material_id = st.selectbox("Select material", pred_candidates, key="pred_material")
            row = unified[unified["material_id"] == material_id].iloc[0]

            comp_cols = [c for c in row.index if c.startswith("wt_percent_") and pd.notna(row[c])]
            if comp_cols:
                with st.expander("Composition (wt%)", expanded=True):
                    comp_data = {c.replace("wt_percent_", "").upper(): round(row[c], 3) for c in comp_cols}
                    st.dataframe(pd.DataFrame([comp_data]), use_container_width=True, hide_index=True)

            pred_info = predict_with_interval(model_bundle, row)
            conf = confidence_score(row, pred_info, model_bundle)

            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Predicted", f"{pred_info['prediction']} MPa")
            if pred_info["actual"] is not None:
                delta = round(pred_info["prediction"] - pred_info["actual"], 1)
                pm2.metric("Actual", f"{pred_info['actual']} MPa", delta=f"{delta:+} error")
            else:
                pm2.metric("Actual", "—")
            pm3.metric("Prediction Confidence", f"{conf['confidence']}%")
            pm4.metric("Risk", conf["risk"])

            fig_iv = go.Figure()
            fig_iv.add_trace(go.Bar(
                x=[""], y=[pred_info["upper"] - pred_info["lower"]],
                base=[pred_info["lower"]],
                name="80% interval", marker_color="rgba(27,67,50,0.15)", width=0.3,
            ))
            fig_iv.add_trace(go.Scatter(
                x=[""], y=[pred_info["prediction"]],
                mode="markers", marker=dict(size=14, color="#1B4332"), name="Predicted",
            ))
            if pred_info["actual"] is not None:
                fig_iv.add_trace(go.Scatter(
                    x=[""], y=[pred_info["actual"]],
                    mode="markers", marker=dict(size=12, color="#B08D57", symbol="diamond"), name="Actual",
                ))
            fig_iv.update_layout(
                yaxis_title="Yield Strength (MPa)", height=260,
                showlegend=True, margin=dict(l=40, r=40, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_iv, use_container_width=True)

            st.markdown('<div class="sec-header">Confidence Breakdown</div>', unsafe_allow_html=True)
            for label, val, mx in [
                ("Uncertainty", conf["uncertainty_penalty"], 30),
                ("Missing Data", conf["missing_data_penalty"], 18),
                ("Source Trust", conf["source_risk_penalty"], 25),
                ("Out-of-Distribution", conf["out_of_distribution_penalty"], 25),
            ]:
                st.progress(min(val / max(mx, 1), 1.0), text=f"{label}: −{val} pts")

            with st.expander("Material Passport"):
                render_passport_for_subsystem(row, "Structural / Chassis")

            with st.expander("Model and uncertainty method"):
                st.markdown(
                    f"- **Best selected model:** {MODEL_ALGO} (chosen by lowest test RMSE)\n"
                    f"- **Interval method:** {pred_info.get('interval_method', interval_label)}\n"
                    "- **Prediction confidence** (`prediction_confidence_score`) starts at 100% "
                    "and is reduced by uncertainty, missing fields, source trust, and OOD risk.\n"
                    "- Source trust uses `source_trust_score` separately from prediction confidence."
                )
    else:
        st.markdown('<div class="sec-header">Waiting for Eligible Evidence</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="empty-state">
            <h3>{esc(selected_model)} — {esc(model_status)}</h3>
            <p><strong>Target property:</strong> {esc(activation['target'])}</p>
            <p><strong>Subsystem:</strong> {esc(activation['subsystem'])}</p>
            <p><strong>Required features:</strong> {esc(activation['required_features'])}</p>
            <p><strong>Eligible source types:</strong> {esc(activation['eligible_source_types'])}</p>
            <p><strong>Current eligible rows:</strong> {activation['eligible_rows']}</p>
            <p><strong>Why not active:</strong> {esc(activation['reason'])}</p>
            <p><strong>How to activate:</strong> {esc(activation['how_to_activate'])}</p>
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
