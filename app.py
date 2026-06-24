"""MatIntel — Material Intelligence Platform for JLR.

Single-command entry: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_pipeline import load_or_create_unified, build_unified_schema, DEMO_ENRICHMENT_COLS
from src.modeling import load_or_train_model, predict_with_interval
from src.confidence import confidence_score
from src.recommender import recommend_materials, USE_CASE_PRESETS
from src.report_generator import build_report
from src.schema_mapper import (
    suggest_schema_mapping,
    apply_schema_mapping,
    STANDARD_FIELDS,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MatIntel — JLR Material Intelligence",
    page_icon="🔩",
    layout="wide",
)

# ── Data loading with error handling ─────────────────────────────────────────

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
    st.info("Ensure `data/raw/steel_strength_raw.csv` exists, or run `python scripts/bootstrap_data.py`.")
    st.stop()

try:
    model_bundle = cached_model(unified_base)
except Exception as e:
    st.error(f"**Model training failed:** {e}")
    st.stop()

# ── Session state for uploaded data ──────────────────────────────────────────

if "uploaded_rows" not in st.session_state:
    st.session_state["uploaded_rows"] = pd.DataFrame()

if st.session_state["uploaded_rows"].empty:
    unified = unified_base.copy()
else:
    unified = pd.concat([unified_base, st.session_state["uploaded_rows"]], ignore_index=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## MatIntel")
    st.caption("Material Intelligence Platform for JLR")
    st.divider()
    st.markdown("### Data Sources")
    st.markdown(f"**matminer steel_strength**  \n{len(unified_base)} experimental alloys  \nTrust: 95 / 100  \nUsed for ML: Yes")
    n_uploaded = len(st.session_state["uploaded_rows"])
    if n_uploaded > 0:
        st.markdown(f"**Uploaded supplier sheet**  \n{n_uploaded} rows  \nTrust: 70 / 100  \nUsed for ML: No")
    st.divider()
    st.markdown("### Model Performance")
    m = model_bundle["metrics"]
    st.markdown(
        f"R² = **{m['R2']}**  \n"
        f"MAE = **{m['MAE']} MPa**  \n"
        f"RMSE = **{m['RMSE']} MPa**  \n"
        f"Train / Test = {m['train_rows']} / {m['test_rows']}"
    )
    st.divider()
    st.caption("Hackathon MVP — screening only, not final engineering approval.")

# ── Header ───────────────────────────────────────────────────────────────────

st.title("MatIntel — Material Intelligence Platform")
st.caption(
    "Multi-source ingestion → schema mapping → unified library "
    "→ yield-strength prediction → confidence / risk → JLR use-case scoring "
    "→ recommendation → decision report"
)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Data Ingestion",
    "2 · Material Library",
    "3 · Property Predictor",
    "4 · Alternative Recommender",
    "5 · Decision Report",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Multi-Source Data Ingestion")
    st.info(
        "MatIntel separates **data integration** from **model training**. "
        "Not every source has the same fields or reliability. "
        "Experimental steel data trains the yield-strength model; "
        "uploaded supplier / internal sheets expand the searchable material library after schema review.",
        icon="ℹ️",
    )

    # ── Source cards ──
    st.markdown("#### Connected Data Sources")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            f"""<div style="border:1px solid #444; border-radius:8px; padding:16px;">
            <strong>matminer_steel_strength</strong><br>
            Type: <code>experimental</code><br>
            Rows: <strong>{len(unified_base)}</strong><br>
            Trust Score: <strong>95 / 100</strong><br>
            Used for ML Training: <strong>Yes</strong>
            </div>""",
            unsafe_allow_html=True,
        )
    with s2:
        up_count = len(st.session_state["uploaded_rows"])
        if up_count > 0:
            st.markdown(
                f"""<div style="border:1px solid #444; border-radius:8px; padding:16px;">
                <strong>uploaded_supplier_sheet</strong><br>
                Type: <code>supplier_sheet</code><br>
                Rows: <strong>{up_count}</strong><br>
                Trust Score: <strong>70 / 100</strong><br>
                Used for ML Training: <strong>No</strong>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style="border:1px solid #555; border-radius:8px; padding:16px; opacity:0.6;">
                <strong>No uploaded sheet yet</strong><br>
                Upload a CSV below to add supplier or internal data.
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Raw data preview ──
    st.markdown("#### Raw Dataset Preview (matminer)")
    with st.expander("Show raw data", expanded=False):
        st.dataframe(raw.head(30), use_container_width=True)

    # ── Unified schema preview ──
    st.markdown("#### Unified Schema Preview")
    preview_cols = [
        "material_id", "material_name", "material_family", "yield_strength_mpa",
        "ultimate_tensile_strength_mpa", "density_g_cm3", "source_dataset",
        "source_type", "source_trust_score", "data_completeness_score",
    ]
    available_preview = [c for c in preview_cols if c in unified.columns]
    st.dataframe(unified[available_preview].head(20), use_container_width=True)

    # ── Missing values ──
    with st.expander("Missing values per column"):
        nulls = unified.isnull().sum()
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        if nulls.empty:
            st.success("No missing values in unified library.")
        else:
            st.dataframe(
                pd.DataFrame({"column": nulls.index, "missing_count": nulls.values,
                               "percent": (nulls.values / len(unified) * 100).round(1)}),
                use_container_width=True,
            )

    # ── Demo enrichment warning ──
    with st.expander("Demo enrichment fields"):
        st.warning(
            "The following columns are **demo enrichment — not from the source dataset**. "
            "They use typical steel defaults for demonstration only and should be replaced "
            "with real measured or supplier-provided values before engineering decisions.",
            icon="⚠️",
        )
        demo_present = [c for c in DEMO_ENRICHMENT_COLS if c in unified.columns]
        st.code(", ".join(demo_present))

    # ── CSV Uploader ──
    st.markdown("---")
    st.markdown("#### Upload Additional Material Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as ex:
            st.error(f"Could not read CSV: {ex}")
            upload_df = None

        if upload_df is not None and not upload_df.empty:
            st.markdown("**Uploaded raw data preview**")
            st.dataframe(upload_df.head(15), use_container_width=True)

            mapping_df = suggest_schema_mapping(upload_df)
            st.markdown("**Suggested Schema Mapping**")
            st.dataframe(mapping_df, use_container_width=True)

            st.markdown("**Override Mapping** (adjust if auto-mapping is wrong)")
            options = ["ignore"] + STANDARD_FIELDS
            overrides: dict[str, str] = {}
            cols_per_row = 3
            upload_cols = list(upload_df.columns)
            for i in range(0, len(upload_cols), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col_name in enumerate(upload_cols[i : i + cols_per_row]):
                    suggested_row = mapping_df[mapping_df["uploaded_column"] == col_name]
                    default = suggested_row["suggested_field"].values[0] if len(suggested_row) > 0 else "ignore"
                    default_idx = options.index(default) if default in options else 0
                    with row_cols[j]:
                        overrides[col_name] = st.selectbox(
                            col_name,
                            options,
                            index=default_idx,
                            key=f"map_{col_name}",
                        )

            if st.button("Ingest Uploaded Sheet", type="primary"):
                mapped = apply_schema_mapping(upload_df, overrides)
                st.session_state["uploaded_rows"] = mapped
                st.success(f"Ingested {len(mapped)} rows into the unified material library (session only).")
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATERIAL LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Unified Material Library")

    max_ys = int(unified["yield_strength_mpa"].max()) if not unified["yield_strength_mpa"].isna().all() else 3000
    min_yield = st.slider("Minimum yield strength (MPa)", 0, max_ys, 250, key="lib_slider")
    filtered = unified[unified["yield_strength_mpa"].fillna(0) >= min_yield].copy()

    lib_cols = [
        "material_id", "material_name", "material_family", "yield_strength_mpa",
        "ultimate_tensile_strength_mpa", "density_g_cm3", "youngs_modulus_gpa",
        "source_dataset", "source_type", "source_trust_score", "data_completeness_score",
    ]
    available_lib = [c for c in lib_cols if c in filtered.columns]
    st.dataframe(filtered[available_lib].head(200), use_container_width=True)

    fig = px.histogram(
        filtered, x="yield_strength_mpa", nbins=30,
        title="Yield Strength Distribution",
        labels={"yield_strength_mpa": "Yield Strength (MPa)"},
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROPERTY PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Yield Strength Predictor")

    material_id = st.selectbox(
        "Select material",
        unified["material_id"].tolist(),
        key="pred_material",
    )
    row = unified[unified["material_id"] == material_id].iloc[0]

    # Show composition features
    st.markdown("**Composition (wt%)**")
    comp_cols = [c for c in row.index if c.startswith("wt_percent_") and pd.notna(row[c])]
    if comp_cols:
        comp_data = {c.replace("wt_percent_", "").upper(): round(row[c], 3) for c in comp_cols}
        st.dataframe(pd.DataFrame([comp_data]), use_container_width=True)
    else:
        st.caption("No composition data available for this material.")

    # Prediction
    pred_info = predict_with_interval(model_bundle, row)
    conf = confidence_score(row, pred_info, model_bundle)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted", f"{pred_info['prediction']} MPa")
    if pred_info["actual"] is not None:
        delta = round(pred_info["prediction"] - pred_info["actual"], 1)
        m2.metric("Actual (measured)", f"{pred_info['actual']} MPa", delta=f"{delta:+} MPa error")
    else:
        m2.metric("Actual", "Not available")
    m3.metric("Confidence", f"{conf['confidence']}%")
    m4.metric("Risk Level", conf["risk"])

    # Prediction interval chart
    st.markdown("**Prediction Interval**")
    fig_interval = go.Figure()
    fig_interval.add_trace(go.Bar(
        x=["Prediction Range"],
        y=[pred_info["upper"] - pred_info["lower"]],
        base=[pred_info["lower"]],
        name="80% prediction interval",
        marker_color="rgba(99,110,250,0.4)",
        width=0.4,
    ))
    fig_interval.add_trace(go.Scatter(
        x=["Prediction Range"],
        y=[pred_info["prediction"]],
        mode="markers+text",
        marker=dict(size=14, color="rgb(99,110,250)"),
        text=[f"{pred_info['prediction']} MPa"],
        textposition="middle right",
        name="Predicted",
    ))
    if pred_info["actual"] is not None:
        fig_interval.add_trace(go.Scatter(
            x=["Prediction Range"],
            y=[pred_info["actual"]],
            mode="markers+text",
            marker=dict(size=14, color="rgb(239,85,59)", symbol="diamond"),
            text=[f"{pred_info['actual']} MPa (actual)"],
            textposition="middle right",
            name="Actual",
        ))
    fig_interval.update_layout(
        yaxis_title="Yield Strength (MPa)",
        height=300,
        showlegend=True,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(fig_interval, use_container_width=True)

    # Confidence breakdown
    st.markdown("**Confidence Breakdown**")
    penalties = [
        ("Model Uncertainty", conf["uncertainty_penalty"], 30),
        ("Missing Data", conf["missing_data_penalty"], 18),
        ("Source Risk", conf["source_risk_penalty"], 25),
        ("Out-of-Distribution", conf["out_of_distribution_penalty"], 25),
    ]
    for label, val, max_val in penalties:
        pct = min(val / max(max_val, 1) * 100, 100)
        col_l, col_r = st.columns([3, 1])
        col_l.progress(pct / 100, text=f"{label}: −{val} pts")
        severity = "low" if val < max_val * 0.3 else ("medium" if val < max_val * 0.6 else "high")
        col_r.caption(severity)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALTERNATIVE RECOMMENDER
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Alternative Material Recommender")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        use_case = st.selectbox("Use-case preset", list(USE_CASE_PRESETS.keys()), key="rec_usecase")
    with r2:
        min_strength = st.number_input("Min yield strength (MPa)", value=250, min_value=0, key="rec_strength")
    with r3:
        baseline_density = st.number_input("Baseline density (g/cm³)", value=7.85, min_value=0.1, key="rec_density")
    with r4:
        min_trust = st.number_input("Min source trust", value=0, min_value=0, max_value=100, key="rec_trust")

    preset_info = USE_CASE_PRESETS.get(use_case, {})
    st.caption(preset_info.get("description", ""))

    ranked = recommend_materials(
        unified,
        min_strength=min_strength,
        baseline_density=baseline_density,
        use_case=use_case,
        min_trust=min_trust,
    )

    if ranked.empty:
        st.warning("No materials match the current filters.")
    else:
        # Top 3 cards
        st.markdown("#### Top 3 Candidates")
        top3 = ranked.head(3)
        card_cols = st.columns(3)
        for idx, (card_col, (_, mat)) in enumerate(zip(card_cols, top3.iterrows())):
            with card_col:
                rank_label = ["🥇", "🥈", "🥉"][idx]
                st.markdown(
                    f"""<div style="border:1px solid #555; border-radius:8px; padding:14px;">
                    <h4>{rank_label} {mat.get('material_name', 'N/A')}</h4>
                    <p style="margin:2px 0;"><strong>Score:</strong> {mat.get('suitability_score', 'N/A')} / 100</p>
                    <p style="margin:2px 0;"><strong>Yield:</strong> {mat.get('yield_strength_mpa', 'N/A')} MPa</p>
                    <p style="margin:2px 0;"><strong>Density:</strong> {mat.get('density_g_cm3', 'N/A')} g/cm³</p>
                    <p style="margin:2px 0;"><strong>Weight saving:</strong> {mat.get('weight_saving_percent', 'N/A')}%</p>
                    <p style="margin:2px 0;"><strong>Source:</strong> {mat.get('source_type', 'N/A')}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                reasons = str(mat.get("reason_codes", ""))
                for r in reasons.split("|"):
                    r = r.strip()
                    if r.startswith("+"):
                        st.success(r, icon="✅")
                    elif r.startswith("-"):
                        st.warning(r, icon="⚠️")

        st.caption("Scores are screening scores, not final engineering approval.")

        # Full ranking table
        st.markdown("#### Full Ranking")
        rank_cols = [
            "material_id", "material_name", "yield_strength_mpa", "density_g_cm3",
            "weight_saving_percent", "source_trust_score", "suitability_score", "reason_codes",
        ]
        available_rank = [c for c in rank_cols if c in ranked.columns]
        st.dataframe(ranked[available_rank].head(30), use_container_width=True)

        # Bar chart
        fig = px.bar(
            ranked.head(10),
            x="material_id",
            y="suitability_score",
            hover_data=["yield_strength_mpa", "weight_saving_percent", "material_name"],
            title="Top 10 Candidates by Suitability Score",
            labels={"suitability_score": "Score", "material_id": "Material"},
        )
        fig.update_layout(bargap=0.15)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — DECISION REPORT
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("Decision Report")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        report_usecase = st.selectbox("Use case", list(USE_CASE_PRESETS.keys()), key="rpt_uc")
    with rc2:
        report_strength = st.number_input("Min yield strength (MPa)", value=250, min_value=0, key="rpt_str")
    with rc3:
        report_density = st.number_input("Baseline density (g/cm³)", value=7.85, min_value=0.1, key="rpt_den")

    ranked_report = recommend_materials(
        unified,
        min_strength=report_strength,
        baseline_density=report_density,
        use_case=report_usecase,
    )

    if ranked_report.empty:
        st.warning("No materials match the current filters.")
    else:
        # Let user choose which material to report on
        report_options = ranked_report.head(20)
        report_labels = [
            f"#{i+1} — {r['material_id']} ({r.get('material_name', '')[:40]}) — Score: {r.get('suitability_score', 'N/A')}"
            for i, (_, r) in enumerate(report_options.iterrows())
        ]
        selected_label = st.selectbox("Select material for report", report_labels, key="rpt_pick")
        selected_idx = report_labels.index(selected_label)
        selected = report_options.iloc[selected_idx]

        report = build_report(
            selected,
            min_strength=report_strength,
            use_case=report_usecase,
            model_metrics=model_bundle["metrics"],
        )

        st.markdown(report)
        st.download_button(
            label="Download Decision Report (.md)",
            data=report,
            file_name="matintel_decision_report.md",
            mime="text/markdown",
        )
