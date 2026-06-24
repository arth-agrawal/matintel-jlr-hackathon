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
from src.recommender import recommend_materials, USE_CASE_PRESETS
from src.report_generator import build_report, VALIDATION_GATES
from src.schema_mapper import (
    suggest_schema_mapping,
    apply_schema_mapping,
    assess_ml_eligibility,
    STANDARD_FIELDS,
    SOURCE_TYPE_PROFILES,
    ALL_SOURCE_TYPES,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="MatIntel", page_icon="◆", layout="wide")

# ── Premium CSS ──────────────────────────────────────────────────────────────

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --racing-green: #1B4332;
    --green-mid: #2D6A4F;
    --green-light: #40916C;
    --graphite: #2B2D30;
    --charcoal: #3D3F43;
    --ivory: #FAF9F6;
    --warm-white: #F5F3EF;
    --gold: #B08D57;
    --gold-muted: #C4A97D;
    --text-primary: #1A1A1A;
    --text-secondary: #5A5A5A;
    --border-light: #E0DCD4;
    --border-mid: #C8C3B8;
    --red-muted: #C0392B;
    --amber-muted: #D4A017;
    --green-status: #27AE60;
}

.stApp { font-family: 'Inter', -apple-system, sans-serif; }

/* Hero */
.hero-block {
    background: linear-gradient(135deg, var(--racing-green) 0%, var(--green-mid) 100%);
    border-radius: 10px;
    padding: 28px 32px 22px;
    margin-bottom: 20px;
    color: white;
}
.hero-block h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 2px;
    letter-spacing: -0.5px;
    color: white;
}
.hero-block .subtitle {
    font-size: 0.95rem;
    color: var(--gold-muted);
    font-weight: 500;
    margin: 0 0 8px;
}
.hero-block .tagline {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
}

/* Metric strip */
.metric-strip {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.metric-chip {
    background: white;
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    min-width: 130px;
    text-align: center;
}
.metric-chip .mv {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--racing-green);
    margin: 0;
}
.metric-chip .ml {
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0;
}

/* Source card */
.src-card {
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 16px;
    background: white;
}
.src-card h4 {
    margin: 0 0 8px;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--racing-green);
}
.src-card table { width: 100%; font-size: 0.82rem; border-collapse: collapse; }
.src-card td { padding: 3px 0; color: var(--text-secondary); }
.src-card td:last-child { font-weight: 600; color: var(--text-primary); text-align: right; }
.src-card.active { border-left: 3px solid var(--green-light); }
.src-card.pending { border-left: 3px solid var(--gold); }
.src-card.empty { border-left: 3px solid var(--border-mid); opacity: 0.55; }

/* Status pills */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.pill-green { background: #E8F5E9; color: #2E7D32; }
.pill-amber { background: #FFF8E1; color: #F57F17; }
.pill-red { background: #FFEBEE; color: #C62828; }
.pill-grey { background: #ECEFF1; color: #546E7A; }

/* Passport card */
.passport {
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 14px 16px;
    background: white;
    margin-bottom: 10px;
}
.passport-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--gold);
    margin: 0 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border-light);
}
.passport table { width: 100%; font-size: 0.82rem; border-collapse: collapse; }
.passport td { padding: 3px 0; }
.passport td:first-child { color: var(--text-secondary); width: 50%; }
.passport td:last-child { font-weight: 500; color: var(--text-primary); text-align: right; }

/* Recommendation card */
.rec-card {
    border: 1px solid var(--border-light);
    border-radius: 10px;
    padding: 18px;
    background: white;
    position: relative;
}
.rec-card.rank-1 { border-top: 3px solid var(--gold); }
.rec-card.rank-2 { border-top: 3px solid #9E9E9E; }
.rec-card.rank-3 { border-top: 3px solid #8D6E63; }
.rec-rank {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--gold);
    margin: 0 0 4px;
}
.rec-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px;
    line-height: 1.2;
}
.rec-score {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--racing-green);
    margin: 0;
}
.rec-score-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin: 0 0 8px;
}
.rec-meta { font-size: 0.78rem; color: var(--text-secondary); margin: 2px 0; }
.rec-meta strong { color: var(--text-primary); }

/* Quality gate */
.qg-row {
    display: flex;
    align-items: center;
    padding: 4px 0;
    font-size: 0.82rem;
}
.qg-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    flex-shrink: 0;
}
.qg-ok { background: var(--green-status); }
.qg-miss { background: var(--amber-muted); }
.qg-label { color: var(--text-primary); }
.qg-status { margin-left: auto; font-size: 0.75rem; color: var(--text-secondary); }

/* Section header */
.sec-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--green-mid);
    margin: 20px 0 10px;
    padding-bottom: 4px;
    border-bottom: 2px solid var(--green-light);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--graphite) !important;
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}
section[data-testid="stSidebar"] .stMarkdown strong {
    color: var(--gold-muted) !important;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading material data …")
def cached_data():
    return load_or_create_unified()

@st.cache_resource(show_spinner="Training yield-strength model …")
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

if st.session_state["uploaded_rows"].empty:
    unified = unified_base.copy()
else:
    unified = pd.concat([unified_base, st.session_state["uploaded_rows"]], ignore_index=True)

n_uploaded = len(st.session_state["uploaded_rows"])
M = model_bundle["metrics"]
n_sources = 1 + (1 if n_uploaded > 0 else 0)
ml_rows = int((unified.get("used_for_ml_training", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1"])).sum())
avg_trust = round(unified["source_trust_score"].mean(), 1) if "source_trust_score" in unified.columns else 0

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fv(val, suffix="", fallback="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    return f"{val}{suffix}"


def _decision_status(row):
    trust = row.get("source_trust_score", 0)
    if trust is None or (isinstance(trust, float) and np.isnan(trust)):
        trust = 0
    completeness = row.get("data_completeness_score", 0)
    if completeness is None or (isinstance(completeness, float) and np.isnan(completeness)):
        completeness = 0
    fatigue = row.get("fatigue_strength_mpa")
    corrosion = row.get("corrosion_resistance_score")
    has_critical = (fatigue is not None and not (isinstance(fatigue, float) and np.isnan(fatigue))) and \
                   (corrosion is not None and not (isinstance(corrosion, float) and np.isnan(corrosion)))
    if trust >= 85 and completeness >= 70 and has_critical:
        return "Ready for validation", "pill-green"
    elif trust >= 50 and completeness >= 40:
        return "Needs review", "pill-amber"
    return "Insufficient evidence", "pill-red"


def render_material_passport(row, pred=None, conf=None):
    """Render grouped Material Passport cards."""
    status_text, status_class = _decision_status(row)
    ml_flag = str(row.get("used_for_ml_training", False)).lower() in ("true", "1")

    # Identity
    st.markdown(f"""<div class="passport">
    <div class="passport-header">Identity</div>
    <table>
    <tr><td>Material Name</td><td>{_fv(row.get('material_name'))}</td></tr>
    <tr><td>Family</td><td>{_fv(row.get('material_family', row.get('family')))}</td></tr>
    <tr><td>Source</td><td>{_fv(row.get('source_dataset'))}</td></tr>
    <tr><td>Source Type</td><td>{_fv(row.get('source_type'))}</td></tr>
    <tr><td>Source Trust</td><td>{_fv(row.get('source_trust_score'), '/100')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    # Performance
    st.markdown(f"""<div class="passport">
    <div class="passport-header">Performance</div>
    <table>
    <tr><td>Yield Strength</td><td>{_fv(row.get('yield_strength_mpa'), ' MPa')}</td></tr>
    <tr><td>Tensile Strength</td><td>{_fv(row.get('ultimate_tensile_strength_mpa'), ' MPa')}</td></tr>
    <tr><td>Density</td><td>{_fv(row.get('density_g_cm3'), ' g/cm³')}</td></tr>
    <tr><td>Young's Modulus</td><td>{_fv(row.get('youngs_modulus_gpa'), ' GPa')}</td></tr>
    <tr><td>Elongation</td><td>{_fv(row.get('elongation_percent'), '%')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    # Sustainability
    st.markdown(f"""<div class="passport">
    <div class="passport-header">Sustainability</div>
    <table>
    <tr><td>Recycled Content</td><td>{_fv(row.get('recycled_content_percent'), '%')}</td></tr>
    <tr><td>CO2 Index</td><td>{_fv(row.get('co2_index'), '/100')}</td></tr>
    <tr><td>CO2 / kg</td><td>{_fv(row.get('co2_kg_per_kg'), ' kg')}</td></tr>
    <tr><td>Recyclability</td><td>{_fv(row.get('recyclability_score'), '/100')}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    # Risk & Governance
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

    # Decision status
    st.markdown(f"""<div class="passport">
    <div class="passport-header">Decision Status</div>
    <p style="margin:4px 0;"><span class="pill {status_class}">{status_text}</span></p>
    </div>""", unsafe_allow_html=True)


def render_quality_gate(row):
    """Compact data quality gate checklist."""
    checks = [
        ("Yield Strength", "yield_strength_mpa"),
        ("Density", "density_g_cm3"),
        ("Source Trust", "source_trust_score"),
        ("Stiffness", "youngs_modulus_gpa"),
        ("Fatigue", "fatigue_strength_mpa"),
        ("Corrosion", "corrosion_resistance_score"),
        ("Supplier Risk", "supplier_risk_score"),
    ]
    html = ""
    for label, field in checks:
        val = row.get(field)
        ok = val is not None and not (isinstance(val, float) and np.isnan(val))
        dot = "qg-ok" if ok else "qg-miss"
        status = "Available" if ok else "Validation needed"
        html += f'<div class="qg-row"><span class="qg-dot {dot}"></span><span class="qg-label">{label}</span><span class="qg-status">{status}</span></div>'
    st.markdown(f'<div class="passport"><div class="passport-header">Data Quality Gate</div>{html}</div>', unsafe_allow_html=True)


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
    st.markdown(f"**Sources:** {n_sources}  \n**Materials:** {len(unified)}  \n**ML rows:** {ml_rows}")
    st.divider()
    st.markdown(f"**Model:** RF ensemble  \nR² **{M['R2']}** · MAE **{M['MAE']}** MPa")
    st.divider()
    st.caption("Screening only. Not final engineering approval.")

# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""<div class="hero-block">
<h1>MatIntel</h1>
<p class="subtitle">Governed Material Intelligence for JLR Engineering</p>
<p class="tagline">Unify fragmented material evidence, predict properties with uncertainty, and recommend what to validate next.</p>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="metric-strip">
<div class="metric-chip"><p class="mv">{n_sources}</p><p class="ml">Evidence Sources</p></div>
<div class="metric-chip"><p class="mv">{len(unified)}</p><p class="ml">Unified Materials</p></div>
<div class="metric-chip"><p class="mv">{ml_rows}</p><p class="ml">ML Eligible Rows</p></div>
<div class="metric-chip"><p class="mv">{M['R2']}</p><p class="ml">Model R²</p></div>
<div class="metric-chip"><p class="mv">{avg_trust}</p><p class="ml">Avg Source Trust</p></div>
</div>""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Evidence Intake",
    "Material Passport Library",
    "Predictive Model",
    "JLR Fit Scoring",
    "Validation Report",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EVIDENCE INTAKE
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="sec-header">Public Reference Data</div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            f"""<div class="src-card active">
            <h4>matminer_steel_strength</h4>
            <table>
            <tr><td>Source Type</td><td>public_experimental</td></tr>
            <tr><td>Rows</td><td>{len(unified_base)}</td></tr>
            <tr><td>Trust Score</td><td>95 / 100</td></tr>
            <tr><td>ML Eligible</td><td>Yes</td></tr>
            <tr><td>Status</td><td>Curated dataset</td></tr>
            </table></div>""",
            unsafe_allow_html=True,
        )
    with s2:
        if n_uploaded > 0:
            up_type = st.session_state.get("upload_source_type", "supplier_sheet")
            profile = SOURCE_TYPE_PROFILES.get(up_type, SOURCE_TYPE_PROFILES["unknown"])
            st.markdown(
                f"""<div class="src-card pending">
                <h4>Uploaded Evidence</h4>
                <table>
                <tr><td>Source Type</td><td>{up_type}</td></tr>
                <tr><td>Rows</td><td>{n_uploaded}</td></tr>
                <tr><td>Trust Score</td><td>{profile['trust']} / 100</td></tr>
                <tr><td>ML Eligible</td><td>{'Yes' if profile['ml_eligible'] else 'No'}</td></tr>
                <tr><td>Status</td><td>Pending review</td></tr>
                </table></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="src-card empty">
                <h4>No evidence uploaded</h4>
                <table><tr><td>Upload a CSV below to add material evidence.</td></tr></table>
                </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("Raw dataset preview"):
        st.dataframe(raw.head(30), use_container_width=True)

    with st.expander("Unified schema preview"):
        preview_cols = [
            "material_id", "material_name", "material_family", "yield_strength_mpa",
            "density_g_cm3", "source_type", "source_trust_score", "used_for_ml_training",
            "data_completeness_score",
        ]
        available_preview = [c for c in preview_cols if c in unified.columns]
        st.dataframe(unified[available_preview].head(20), use_container_width=True)

    with st.expander("Missing values"):
        nulls = unified.isnull().sum()
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        if nulls.empty:
            st.success("No missing values.")
        else:
            st.dataframe(
                pd.DataFrame({"Column": nulls.index, "Missing": nulls.values,
                               "%": (nulls.values / len(unified) * 100).round(1)}),
                use_container_width=True, hide_index=True,
            )

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

    max_ys = int(unified["yield_strength_mpa"].max()) if not unified["yield_strength_mpa"].isna().all() else 3000
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        min_yield = st.slider("Min yield strength (MPa)", 0, max_ys, 250, key="lib_slider")
    filtered = unified[unified["yield_strength_mpa"].fillna(0) >= min_yield].copy()

    lib_cols = [
        "material_id", "material_name", "material_family", "yield_strength_mpa",
        "ultimate_tensile_strength_mpa", "density_g_cm3", "youngs_modulus_gpa",
        "source_type", "source_trust_score", "data_completeness_score",
    ]
    available_lib = [c for c in lib_cols if c in filtered.columns]
    st.dataframe(filtered[available_lib].head(200), use_container_width=True, hide_index=True)

    # Passport for selected material
    st.markdown('<div class="sec-header">Material Passport</div>', unsafe_allow_html=True)
    passport_id = st.selectbox("Select material", filtered["material_id"].tolist(), key="passport_select")
    if passport_id:
        p_row = filtered[filtered["material_id"] == passport_id].iloc[0]
        p1, p2 = st.columns(2)
        with p1:
            render_material_passport(p_row)
        with p2:
            fig = px.histogram(
                filtered, x="yield_strength_mpa", nbins=30,
                labels={"yield_strength_mpa": "Yield Strength (MPa)"},
            )
            fig.update_layout(
                bargap=0.05, height=400, margin=dict(l=20, r=20, t=30, b=20),
                title_text="Yield Strength Distribution", title_font_size=13,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE MODEL
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="sec-header">Predictive Model</div>', unsafe_allow_html=True)
    st.caption(f"Random Forest ensemble · R² {M['R2']} · MAE {M['MAE']} MPa · {M['train_rows']} training rows")

    material_id = st.selectbox("Select material", unified["material_id"].tolist(), key="pred_material")
    row = unified[unified["material_id"] == material_id].iloc[0]

    # Composition
    comp_cols = [c for c in row.index if c.startswith("wt_percent_") and pd.notna(row[c])]
    if comp_cols:
        with st.expander("Composition (wt%)", expanded=True):
            comp_data = {c.replace("wt_percent_", "").upper(): round(row[c], 3) for c in comp_cols}
            st.dataframe(pd.DataFrame([comp_data]), use_container_width=True, hide_index=True)

    pred_info = predict_with_interval(model_bundle, row)
    conf = confidence_score(row, pred_info, model_bundle)

    # Metrics
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Predicted", f"{pred_info['prediction']} MPa")
    if pred_info["actual"] is not None:
        delta = round(pred_info["prediction"] - pred_info["actual"], 1)
        pm2.metric("Actual", f"{pred_info['actual']} MPa", delta=f"{delta:+} error")
    else:
        pm2.metric("Actual", "—")
    pm3.metric("Confidence", f"{conf['confidence']}%")
    risk_col_map = {"Green": "normal", "Amber": "off", "Red": "inverse"}
    pm4.metric("Risk", conf["risk"])

    # Interval chart
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Bar(
        x=[""], y=[pred_info["upper"] - pred_info["lower"]],
        base=[pred_info["lower"]],
        name="80% interval", marker_color="rgba(27,67,50,0.15)", width=0.3,
    ))
    fig_iv.add_trace(go.Scatter(
        x=[""], y=[pred_info["prediction"]],
        mode="markers", marker=dict(size=14, color="#1B4332"),
        name="Predicted",
    ))
    if pred_info["actual"] is not None:
        fig_iv.add_trace(go.Scatter(
            x=[""], y=[pred_info["actual"]],
            mode="markers", marker=dict(size=12, color="#B08D57", symbol="diamond"),
            name="Actual",
        ))
    fig_iv.update_layout(
        yaxis_title="Yield Strength (MPa)", height=260,
        showlegend=True, margin=dict(l=40, r=40, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_iv, use_container_width=True)

    # Confidence breakdown
    st.markdown('<div class="sec-header">Confidence Breakdown</div>', unsafe_allow_html=True)
    penalties = [
        ("Uncertainty", conf["uncertainty_penalty"], 30),
        ("Missing Data", conf["missing_data_penalty"], 18),
        ("Source Trust", conf["source_risk_penalty"], 25),
        ("Out-of-Distribution", conf["out_of_distribution_penalty"], 25),
    ]
    for label, val, mx in penalties:
        pct = min(val / max(mx, 1), 1.0)
        st.progress(pct, text=f"{label}:  −{val} pts")

    # Passport
    with st.expander("Material Passport"):
        render_material_passport(row, pred_info, conf)

    with st.expander("Model and uncertainty method"):
        st.markdown(
            "- **250-tree Random Forest** trained on matminer experimental steel compositions.\n"
            "- **Point estimate**: mean prediction across all trees.\n"
            "- **Uncertainty interval**: 10th–90th percentile of individual tree predictions. "
            "Wide spread = low agreement = lower confidence.\n"
            "- **Confidence** starts at 100% and is reduced by: model uncertainty, "
            "missing engineering fields, source trust level, and out-of-distribution risk "
            "(composition far from training data)."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — JLR FIT SCORING
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="sec-header">JLR Fit Scoring</div>', unsafe_allow_html=True)
    st.caption("Weighted engineering score, not a black-box ML output.")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        use_case = st.selectbox("Use case", list(USE_CASE_PRESETS.keys()), key="rec_usecase")
    with r2:
        min_strength = st.number_input("Min yield (MPa)", value=250, min_value=0, key="rec_strength")
    with r3:
        baseline_density = st.number_input("Baseline density", value=7.85, min_value=0.1, key="rec_density")
    with r4:
        min_trust = st.number_input("Min trust", value=0, min_value=0, max_value=100, key="rec_trust")

    ranked = recommend_materials(unified, min_strength=min_strength, baseline_density=baseline_density, use_case=use_case, min_trust=min_trust)

    if ranked.empty:
        st.warning("No materials match the current filters.")
    else:
        # Data quality gate for top material
        st.markdown('<div class="sec-header">Data Quality Gate — Top Candidate</div>', unsafe_allow_html=True)
        top_row = ranked.iloc[0]
        gq1, gq2 = st.columns([1, 2])
        with gq1:
            render_quality_gate(top_row)
        with gq2:
            with st.expander("Scoring logic"):
                weights = USE_CASE_PRESETS.get(use_case, {}).get("weights", {})
                w_df = pd.DataFrame([
                    {"Component": k.replace("_", " ").title(), "Weight": f"{v:.0%}"}
                    for k, v in weights.items()
                ])
                st.dataframe(w_df, use_container_width=True, hide_index=True)
                st.caption(USE_CASE_PRESETS.get(use_case, {}).get("description", ""))

        # Top 3 cards
        st.markdown('<div class="sec-header">Top Candidates</div>', unsafe_allow_html=True)
        top3 = ranked.head(3)
        card_cols = st.columns(3)

        sub_score_cols = [
            ("Strength", "strength_score"),
            ("Weight Saving", "weight_saving_score"),
            ("Stiffness", "stiffness_score"),
            ("Confidence", "confidence_score"),
            ("Sustainability", "sustainability_score"),
            ("Mfg", "manufacturability_score"),
            ("Cost", "cost_score"),
            ("Supply Risk", "supply_risk_score"),
        ]

        for idx, (card_col, (_, mat)) in enumerate(zip(card_cols, top3.iterrows())):
            with card_col:
                rank_labels = ["1st", "2nd", "3rd"]
                rank_class = f"rank-{idx+1}"
                mat_name = str(mat.get("material_name", "N/A"))
                if len(mat_name) > 30:
                    mat_name = mat_name[:27] + "..."
                status_text, status_class = _decision_status(mat)
                score_val = mat.get("suitability_score", 0)

                # Missing validation items
                missing_items = []
                for field, label in [("fatigue_strength_mpa", "Fatigue"), ("corrosion_resistance_score", "Corrosion"), ("hardness_hv", "Hardness")]:
                    v = mat.get(field)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        missing_items.append(label)
                missing_str = ", ".join(missing_items) if missing_items else "None"

                st.markdown(
                    f"""<div class="rec-card {rank_class}">
                    <p class="rec-rank">{rank_labels[idx]} Candidate</p>
                    <p class="rec-name">{mat_name}</p>
                    <p class="rec-score">{score_val}</p>
                    <p class="rec-score-label">Screening Score / 100</p>
                    <p style="margin:4px 0;"><span class="pill {status_class}">{status_text}</span></p>
                    <p class="rec-meta"><strong>Yield:</strong> {mat.get('yield_strength_mpa', '—')} MPa</p>
                    <p class="rec-meta"><strong>Density:</strong> {mat.get('density_g_cm3', '—')} g/cm³</p>
                    <p class="rec-meta"><strong>Wt save:</strong> {mat.get('weight_saving_percent', '—')}%</p>
                    <p class="rec-meta"><strong>Source:</strong> {mat.get('source_type', '—')} · Trust {_fv(mat.get('source_trust_score'))}</p>
                    <p class="rec-meta"><strong>Missing:</strong> {missing_str}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Reason codes
                reasons = str(mat.get("reason_codes", ""))
                for r_code in reasons.split("|"):
                    r_code = r_code.strip()
                    if r_code.startswith("+"):
                        st.success(r_code, icon="✓")
                    elif r_code.startswith("-"):
                        st.warning(r_code, icon="!")

                # Sub-scores
                with st.expander("Sub-scores"):
                    for label, col in sub_score_cols:
                        val = mat.get(col, 50)
                        if isinstance(val, float) and np.isnan(val):
                            val = 50
                        st.progress(min(val / 100, 1.0), text=f"{label}: {round(val, 1)}")

        # Full ranking
        with st.expander("Full ranking table"):
            rank_cols = [
                "material_id", "material_name", "yield_strength_mpa", "density_g_cm3",
                "weight_saving_percent", "source_trust_score", "suitability_score",
            ]
            available_rank = [c for c in rank_cols if c in ranked.columns]
            st.dataframe(ranked[available_rank].head(30), use_container_width=True, hide_index=True)

        fig = px.bar(
            ranked.head(10), x="material_id", y="suitability_score",
            hover_data=["yield_strength_mpa", "weight_saving_percent"],
            labels={"suitability_score": "Score", "material_id": ""},
        )
        fig.update_layout(
            bargap=0.15, height=320, margin=dict(l=40, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title_text="Top 10 by Screening Score", title_font_size=13,
        )
        fig.update_traces(marker_color="#2D6A4F")
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — VALIDATION REPORT
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="sec-header">Validation Report</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        report_usecase = st.selectbox("Use case", list(USE_CASE_PRESETS.keys()), key="rpt_uc")
    with rc2:
        report_strength = st.number_input("Min yield (MPa)", value=250, min_value=0, key="rpt_str")
    with rc3:
        report_density = st.number_input("Baseline density", value=7.85, min_value=0.1, key="rpt_den")

    ranked_report = recommend_materials(unified, min_strength=report_strength, baseline_density=report_density, use_case=report_usecase)

    if ranked_report.empty:
        st.warning("No materials match the current filters.")
    else:
        report_options = ranked_report.head(20)
        report_labels = [
            f"#{i+1} {r['material_id']} — Score {r.get('suitability_score', '—')}"
            for i, (_, r) in enumerate(report_options.iterrows())
        ]
        selected_label = st.selectbox("Select material", report_labels, key="rpt_pick")
        selected_idx = report_labels.index(selected_label)
        selected = report_options.iloc[selected_idx]

        report = build_report(
            selected, min_strength=report_strength, use_case=report_usecase,
            model_metrics=model_bundle["metrics"],
            n_reference_rows=len(unified_base), n_uploaded_rows=n_uploaded,
        )

        st.markdown(report)

        st.markdown('<div class="sec-header">Validation Gate Tracker</div>', unsafe_allow_html=True)
        for g in VALIDATION_GATES:
            st.checkbox(g["gate"], value=False, key=f"gate_{g['gate']}", help=g["detail"])

        st.download_button("Download Report (.md)", data=report, file_name="matintel_report.md", mime="text/markdown")
