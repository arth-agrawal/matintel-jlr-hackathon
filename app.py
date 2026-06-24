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

st.set_page_config(
    page_title="MatIntel — JLR Material Intelligence",
    page_icon="🔩",
    layout="wide",
)

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
    st.info("Ensure `data/raw/steel_strength_raw.csv` exists, or run `python scripts/bootstrap_data.py`.")
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
metrics = model_bundle["metrics"]

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## MatIntel")
    st.caption("Material Intelligence Platform for JLR")
    st.divider()

    st.markdown("### Platform Layers")
    st.markdown(
        "**L1** Public reference data  \n"
        "**L2** Uploaded evidence + review  \n"
        "**L3** Prediction + uncertainty  \n"
        "**L4** JLR suitability scoring  \n"
        "**L5** Validation-aware report"
    )
    st.divider()

    st.markdown("### Data Sources")
    st.markdown(
        f"**matminer steel_strength**  \n"
        f"{len(unified_base)} experimental alloys  \n"
        f"Trust: 95/100 · ML: Yes"
    )
    if n_uploaded > 0:
        st.markdown(
            f"**Uploaded evidence**  \n"
            f"{n_uploaded} rows  \n"
            f"Trust: 70/100 · ML: No"
        )
    st.divider()

    st.markdown("### Model")
    st.markdown(
        f"RandomForest ensemble  \n"
        f"R² = **{metrics['R2']}** · MAE = **{metrics['MAE']}** MPa  \n"
        f"Train / Test = {metrics['train_rows']} / {metrics['test_rows']}"
    )
    st.divider()
    st.caption("Hackathon MVP — screening only, not final engineering approval.")

# ── Header ───────────────────────────────────────────────────────────────────

st.title("MatIntel — Material Intelligence Platform")
st.caption(
    "Layer 1: Public reference data → Layer 2: Evidence upload + review → "
    "Layer 3: Ensemble prediction + confidence → Layer 4: JLR suitability scoring → "
    "Layer 5: Validation-aware decision report"
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

    # ── Layer 1: Public Reference Data ──
    st.subheader("Layer 1 — Public Reference Data")
    st.info(
        "Public datasets bootstrap the predictive layer before private JLR data is connected. "
        "The matminer **steel_strength** dataset provides 312 experimentally measured steel alloys "
        "with composition and mechanical properties — used directly as ML training data.",
        icon="📊",
    )

    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            f"""<div style="border:1px solid #4a9; border-radius:8px; padding:16px;">
            <h4 style="margin-top:0;">matminer_steel_strength</h4>
            <table style="width:100%; font-size:0.9em;">
            <tr><td>Source Type</td><td><strong>public_experimental</strong></td></tr>
            <tr><td>Rows</td><td><strong>{len(unified_base)}</strong></td></tr>
            <tr><td>Trust Score</td><td><strong>95 / 100</strong></td></tr>
            <tr><td>ML Eligible</td><td><strong>Yes</strong></td></tr>
            <tr><td>Engineer Reviewed</td><td><strong>N/A (curated dataset)</strong></td></tr>
            </table>
            </div>""",
            unsafe_allow_html=True,
        )
    with s2:
        if n_uploaded > 0:
            up_type = st.session_state.get("upload_source_type", "supplier_sheet")
            profile = SOURCE_TYPE_PROFILES.get(up_type, SOURCE_TYPE_PROFILES["unknown"])
            st.markdown(
                f"""<div style="border:1px solid #a94; border-radius:8px; padding:16px;">
                <h4 style="margin-top:0;">uploaded_evidence</h4>
                <table style="width:100%; font-size:0.9em;">
                <tr><td>Source Type</td><td><strong>{up_type}</strong></td></tr>
                <tr><td>Rows</td><td><strong>{n_uploaded}</strong></td></tr>
                <tr><td>Trust Score</td><td><strong>{profile['trust']} / 100</strong></td></tr>
                <tr><td>ML Eligible</td><td><strong>{'Yes' if profile['ml_eligible'] else 'No'}</strong></td></tr>
                <tr><td>Engineer Reviewed</td><td><strong>No (pending)</strong></td></tr>
                </table>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style="border:1px solid #666; border-radius:8px; padding:16px; opacity:0.5;">
                <h4 style="margin-top:0;">No evidence uploaded yet</h4>
                <p>Upload a CSV below to add supplier or internal data as Layer 2 evidence.</p>
                </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("Raw dataset preview (matminer)", expanded=False):
        st.dataframe(raw.head(30), use_container_width=True)

    st.markdown("#### Unified Schema Preview")
    preview_cols = [
        "material_id", "material_name", "material_family", "yield_strength_mpa",
        "ultimate_tensile_strength_mpa", "density_g_cm3", "source_dataset",
        "source_type", "source_trust_score", "used_for_ml_training", "data_completeness_score",
    ]
    available_preview = [c for c in preview_cols if c in unified.columns]
    st.dataframe(unified[available_preview].head(20), use_container_width=True)

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

    with st.expander("Demo enrichment fields"):
        st.warning(
            "The following columns are **demo enrichment — not from the source dataset**. "
            "They use typical steel defaults for demonstration and must be replaced "
            "with measured or supplier-provided values before engineering decisions.",
            icon="⚠️",
        )
        demo_present = [c for c in DEMO_ENRICHMENT_COLS if c in unified.columns]
        st.code(", ".join(demo_present))

    # ── Layer 2: Evidence Source Upload ──
    st.markdown("---")
    st.subheader("Layer 2 — Evidence Source Upload")
    st.info(
        "Uploaded sheets are **not blindly ingested**. They go through deterministic schema mapping, "
        "confidence scoring, and engineer review before influencing any recommendation. "
        "ML training eligibility depends on source type, data completeness, and review status.",
        icon="🔬",
    )

    uploaded_file = st.file_uploader("Upload material evidence (CSV)", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as ex:
            st.error(f"Could not read CSV: {ex}")
            upload_df = None

        if upload_df is not None and not upload_df.empty:
            st.markdown("**Raw uploaded data**")
            st.dataframe(upload_df.head(15), use_container_width=True)

            # Source type selection
            src_col, info_col = st.columns([1, 2])
            with src_col:
                chosen_source_type = st.selectbox(
                    "Classify source type",
                    ALL_SOURCE_TYPES,
                    index=ALL_SOURCE_TYPES.index("supplier_sheet"),
                    key="upload_src_type",
                )
            with info_col:
                profile = SOURCE_TYPE_PROFILES[chosen_source_type]
                st.markdown(
                    f"**{chosen_source_type}** — {profile['description']}  \n"
                    f"Trust: **{profile['trust']}/100** · "
                    f"ML eligible: **{'Yes' if profile['ml_eligible'] else 'No'}** · "
                    f"Usage: {profile['usage']}"
                )

            # Schema mapping
            mapping_df = suggest_schema_mapping(upload_df)
            st.markdown("**Schema Mapping (deterministic)**")
            st.dataframe(mapping_df, use_container_width=True)

            st.markdown("**Override Mapping** — adjust if auto-mapping is incorrect")
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
                            col_name, options, index=default_idx, key=f"map_{col_name}",
                        )

            # ML eligibility assessment
            temp_mapped = apply_schema_mapping(upload_df, overrides, source_type=chosen_source_type)
            ml_assessment = assess_ml_eligibility(temp_mapped, chosen_source_type)

            st.markdown("**Ingestion Assessment**")
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Source Type", chosen_source_type)
            ac2.metric("Source Trust", f"{profile['trust']}/100")
            ac3.metric("Mapping Confidence", f"{mapping_df['confidence'].mean():.0f}%")
            ac4.metric("ML Eligible", "Yes" if ml_assessment["eligible"] else "No")

            if not ml_assessment["eligible"]:
                st.caption(f"ML reason: {ml_assessment['reason']}")
            st.caption(f"Usage: {ml_assessment['usage']}")
            st.caption("Engineer reviewed: **No** (pending review)")

            if st.button("Ingest Evidence Source", type="primary"):
                mapped = apply_schema_mapping(
                    upload_df, overrides,
                    source_type=chosen_source_type,
                )
                st.session_state["uploaded_rows"] = mapped
                st.session_state["upload_source_type"] = chosen_source_type
                st.success(f"Ingested {len(mapped)} rows as **{chosen_source_type}** (session only, not persisted).")
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
        "source_dataset", "source_type", "source_trust_score", "used_for_ml_training",
        "data_completeness_score",
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
    st.subheader("Layer 3 — Yield Strength Prediction")

    st.info(
        "**Model:** RandomForest ensemble (250 trees) trained on matminer experimental data.  \n"
        "**Prediction interval:** derived from the spread of individual tree predictions "
        "(10th–90th percentile). A wide interval means the trees disagree — lower confidence.  \n"
        f"**Performance:** R² = {metrics['R2']}, MAE = {metrics['MAE']} MPa "
        f"(trained on {metrics['train_rows']} rows, tested on {metrics['test_rows']}).",
        icon="🤖",
    )

    material_id = st.selectbox(
        "Select material",
        unified["material_id"].tolist(),
        key="pred_material",
    )
    row = unified[unified["material_id"] == material_id].iloc[0]

    # Composition features
    st.markdown("**Composition (wt%)**")
    comp_cols = [c for c in row.index if c.startswith("wt_percent_") and pd.notna(row[c])]
    if comp_cols:
        comp_data = {c.replace("wt_percent_", "").upper(): round(row[c], 3) for c in comp_cols}
        st.dataframe(pd.DataFrame([comp_data]), use_container_width=True)
    else:
        st.caption("No composition data available — prediction relies on imputed medians.")

    # Prediction
    pred_info = predict_with_interval(model_bundle, row)
    conf = confidence_score(row, pred_info, model_bundle)

    # Metrics
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
        x=["Prediction"],
        y=[pred_info["upper"] - pred_info["lower"]],
        base=[pred_info["lower"]],
        name="80% interval (10th–90th %ile of 250 trees)",
        marker_color="rgba(99,110,250,0.3)",
        width=0.35,
    ))
    fig_interval.add_trace(go.Scatter(
        x=["Prediction"],
        y=[pred_info["prediction"]],
        mode="markers+text",
        marker=dict(size=14, color="rgb(99,110,250)"),
        text=[f"{pred_info['prediction']} MPa"],
        textposition="middle right",
        name="Predicted (ensemble mean)",
    ))
    if pred_info["actual"] is not None:
        fig_interval.add_trace(go.Scatter(
            x=["Prediction"],
            y=[pred_info["actual"]],
            mode="markers+text",
            marker=dict(size=14, color="rgb(239,85,59)", symbol="diamond"),
            text=[f"{pred_info['actual']} MPa (actual)"],
            textposition="middle right",
            name="Actual (measured)",
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
    st.caption("Confidence starts at 100% and penalties are subtracted for each risk factor.")
    penalties = [
        ("Uncertainty Penalty", conf["uncertainty_penalty"], 30,
         "Wide tree spread = low agreement among estimators"),
        ("Missing Data Penalty", conf["missing_data_penalty"], 18,
         "Key engineering fields not present in source"),
        ("Source Trust Penalty", conf["source_risk_penalty"], 25,
         "Lower-trust source types receive higher penalty"),
        ("Out-of-Distribution Penalty", conf["out_of_distribution_penalty"], 25,
         "Composition far from training data distribution"),
    ]
    for label, val, max_val, explanation in penalties:
        pct = min(val / max(max_val, 1) * 100, 100)
        col_l, col_r = st.columns([3, 1])
        col_l.progress(pct / 100, text=f"{label}: −{val} pts")
        col_r.caption(explanation)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALTERNATIVE RECOMMENDER
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Layer 4 — JLR Use-Case Recommender")

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
        # Explanation
        st.info(
            "Final screening score is a **weighted engineering score**, not a black-box ML output. "
            "Sub-scores are computed from measurable properties and weighted per JLR use-case preset. "
            "Missing data defaults to a neutral score of 50.",
            icon="⚖️",
        )

        # Weight breakdown for current preset
        with st.expander("Score weight breakdown for this use case"):
            weights = preset_info.get("weights", {})
            w_df = pd.DataFrame([
                {"Sub-Score": k.replace("_", " ").title(), "Weight": f"{v:.0%}"}
                for k, v in weights.items()
            ])
            st.dataframe(w_df, use_container_width=True, hide_index=True)

        # Top 3 cards with sub-score breakdown
        st.markdown("#### Top 3 Candidates")
        top3 = ranked.head(3)
        card_cols = st.columns(3)

        sub_score_cols = [
            ("Strength", "strength_score"),
            ("Weight Saving", "weight_saving_score"),
            ("Stiffness", "stiffness_score"),
            ("Confidence", "confidence_score"),
            ("Sustainability", "sustainability_score"),
            ("Manufacturability", "manufacturability_score"),
            ("Cost", "cost_score"),
            ("Supply Risk", "supply_risk_score"),
        ]

        for idx, (card_col, (_, mat)) in enumerate(zip(card_cols, top3.iterrows())):
            with card_col:
                rank_label = ["🥇", "🥈", "🥉"][idx]
                mat_name = str(mat.get("material_name", "N/A"))
                if len(mat_name) > 35:
                    mat_name = mat_name[:32] + "…"
                st.markdown(
                    f"""<div style="border:1px solid #555; border-radius:8px; padding:14px;">
                    <h4 style="margin-top:0;">{rank_label} {mat_name}</h4>
                    <p style="margin:2px 0;"><strong>Score:</strong> {mat.get('suitability_score', 'N/A')} / 100</p>
                    <p style="margin:2px 0;"><strong>Yield:</strong> {mat.get('yield_strength_mpa', 'N/A')} MPa</p>
                    <p style="margin:2px 0;"><strong>Density:</strong> {mat.get('density_g_cm3', 'N/A')} g/cm³</p>
                    <p style="margin:2px 0;"><strong>Wt save:</strong> {mat.get('weight_saving_percent', 'N/A')}%</p>
                    <p style="margin:2px 0;"><strong>Source:</strong> {mat.get('source_type', 'N/A')}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Sub-score breakdown
                sub_rows = []
                for label, col in sub_score_cols:
                    val = mat.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        sub_rows.append({"Sub-Score": label, "Value": round(val, 1)})
                if sub_rows:
                    st.dataframe(
                        pd.DataFrame(sub_rows), use_container_width=True,
                        hide_index=True, height=min(35 * len(sub_rows) + 38, 340),
                    )

                # Reason codes
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
            "weight_saving_percent", "source_trust_score", "suitability_score",
        ]
        available_rank = [c for c in rank_cols if c in ranked.columns]
        st.dataframe(ranked[available_rank].head(30), use_container_width=True)

        # Sub-score breakdown table for top 10
        with st.expander("Sub-score breakdown (top 10)"):
            breakdown_cols = ["material_id", "suitability_score"] + [c for _, c in sub_score_cols if c in ranked.columns]
            st.dataframe(ranked[breakdown_cols].head(10), use_container_width=True)

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
    st.subheader("Layer 5 — Validation-Aware Decision Report")

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
        report_options = ranked_report.head(20)
        report_labels = [
            f"#{i+1} — {r['material_id']} ({str(r.get('material_name', ''))[:35]}) — Score: {r.get('suitability_score', 'N/A')}"
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
            n_reference_rows=len(unified_base),
            n_uploaded_rows=n_uploaded,
        )

        st.markdown(report)

        # Validation gates status
        st.markdown("---")
        st.markdown("#### Validation Gate Tracker")
        st.caption("All gates must be passed before material can be approved for engineering release.")
        for g in VALIDATION_GATES:
            st.checkbox(g["gate"], value=False, key=f"gate_{g['gate']}", help=g["detail"])

        st.download_button(
            label="Download Decision Report (.md)",
            data=report,
            file_name="matintel_decision_report.md",
            mime="text/markdown",
        )
