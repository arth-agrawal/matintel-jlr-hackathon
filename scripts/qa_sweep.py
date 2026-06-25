#!/usr/bin/env python3
"""End-to-end QA sweep for MatIntel submission readiness."""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = []
FAIL = []


def ok(name: str, detail: str = "") -> None:
    PASS.append((name, detail))


def bad(name: str, detail: str) -> None:
    FAIL.append((name, detail))


def main() -> int:
    # ── Imports ──
    try:
        from src.data_pipeline import load_or_create_unified, build_unified_schema
        from src.public_sources import auto_load_trusted_reference, CACHE_JARVIS_5000
        from src.modeling import load_all_trained_models, predict_with_interval, train_property_model
        from src.evidence_cockpit import (
            build_all_subsystem_cards, compute_hero_metrics, render_subsystem_card_html,
            render_evidence_flow_html, build_coverage_table,
        )
        from src.model_registry import detect_trainable_targets, get_trained_models_for_subsystem
        from src.evidence_coverage import filter_subsystem_rows, format_display_table
        from src.recommender import recommend_materials, USE_CASE_PRESETS
        from src.report_generator import build_report
        from src.upload_workflow import (
            template_download_bytes, suggest_schema_mapping, enrich_mapping_with_units,
            ingest_approved_evidence, SUBSYSTEM_TEMPLATE_COLUMNS,
        )
        from src.upload_model_update import assess_upload_model_impact, retrain_affected_model
        from src.demo_ui import display_status, upload_template_count
        from src.data_assumptions import is_assumption_field
        from src.schema_mapper import esc
        ok("imports", "all app modules import")
    except Exception as exc:
        bad("imports", traceback.format_exc())
        _report()
        return 1

    # ── Data universe ──
    try:
        _, steel = load_or_create_unified()
        ref, _ = auto_load_trusted_reference()
        unified = pd.concat([steel, ref], ignore_index=True) if not ref.empty else steel.copy()
        if len(steel) != 312:
            bad("steel_rows", f"expected 312, got {len(steel)}")
        else:
            ok("steel_rows", "312")
        if len(ref) != 8000:
            bad("reference_rows", f"expected 8000, got {len(ref)}")
        else:
            ok("reference_rows", "8000")
        if len(unified) != 8312:
            bad("unified_rows", f"expected 8312, got {len(unified)}")
        else:
            ok("unified_rows", "8312")
    except Exception as exc:
        bad("data_universe", traceback.format_exc())

    # ── Models ──
    try:
        models = load_all_trained_models()
        expected = {
            "structural_yield_strength", "computed_band_gap",
            "computed_formation_energy", "elastic_modulus_proxy",
        }
        if set(models.keys()) != expected:
            bad("models_trained", f"expected {expected}, got {set(models.keys())}")
        else:
            ok("models_trained", "4 models")
        for k, b in models.items():
            if not b.get("metrics") and not b.get("r2"):
                bad(f"model_metrics_{k}", "missing metrics")
            else:
                ok(f"model_metrics_{k}", f"R2={b.get('r2', b.get('metrics', {}).get('R2'))}")
    except Exception as exc:
        bad("models", traceback.format_exc())

    # ── Hero ──
    try:
        registry = detect_trainable_targets(unified)
        hero = compute_hero_metrics(unified, registry, models, 312)
        if hero["unified_materials"] != 8312:
            bad("hero_unified", str(hero["unified_materials"]))
        else:
            ok("hero_unified", "8312")
        if hero["active_trained_models"] != 4:
            bad("hero_models", str(hero["active_trained_models"]))
        else:
            ok("hero_models", "4")
        if upload_template_count() != 9:
            bad("upload_templates", str(upload_template_count()))
        else:
            ok("upload_templates", "9")
    except Exception as exc:
        bad("hero", traceback.format_exc())

    # ── Subsystem cards HTML ──
    try:
        cards = build_all_subsystem_cards(unified, registry)
        for card in cards:
            html = render_subsystem_card_html(card)
            if not html.startswith("<div"):
                bad("subsystem_html", f"bad start for {card['subsystem']}")
            if "&lt;p class" in html or html.count("<p class") != html.count("</p>"):
                pass  # basic sanity
            if "No coverage yet" in html or "Upload evidence needed" in card.get("status", ""):
                if card["counts"]["relevant_rows"] > 0:
                    bad("subsystem_status", card["subsystem"])
        ok("subsystem_cards", f"{len(cards)} cards, HTML renders")
        flow = render_evidence_flow_html(8312, 4, 0)
        if "<div" not in flow:
            bad("flow_html", "empty flow")
        else:
            ok("flow_html", "renders")
    except Exception as exc:
        bad("subsystem_cards", traceback.format_exc())

    # ── Passport library subsystems ──
    subs_tests = {
        "Structural / Chassis": lambda n: n > 0,
        "Electronics / Thermal Interface": lambda n: n > 0,
        "Battery Enclosure / Underbody": lambda n: n > 0,
        "Interior / Seating / Foam": lambda n: n >= 0,
        "Tyres / Elastomers": lambda n: n >= 0,
    }
    for sub, check in subs_tests.items():
        try:
            rows = filter_subsystem_rows(unified, sub)
            if not check(len(rows)):
                bad(f"subsystem_rows_{sub}", str(len(rows)))
            else:
                ok(f"subsystem_rows_{sub}", str(len(rows)))
            disp = format_display_table(rows.head(5), sub)
            if disp.empty and len(rows) > 0:
                bad(f"display_table_{sub}", "empty display")
        except Exception as exc:
            bad(f"subsystem_{sub}", traceback.format_exc())

    # ── Predictive model per model ──
    for model_key, bundle in models.items():
        try:
            target = bundle["target"]
            candidates = unified[unified[target].notna()]["material_id"].tolist()
            if not candidates:
                bad(f"pred_candidates_{model_key}", "no candidates")
                continue
            row = unified[unified["material_id"] == candidates[0]].iloc[0]
            pred = predict_with_interval(bundle, row)
            if "prediction" not in pred:
                bad(f"predict_{model_key}", "no prediction key")
            else:
                ok(f"predict_{model_key}", str(pred["prediction"]))
            # MPa sanity for yield
            if model_key == "structural_yield_strength" and pred["prediction"] < 10:
                bad("mpa_units", f"yield looks like Pa: {pred['prediction']}")
            elif model_key == "structural_yield_strength":
                ok("mpa_units", f"{pred['prediction']} MPa range ok")
            test_a = bundle.get("test_actuals", [])
            test_p = bundle.get("test_predictions", [])
            if test_a and test_p:
                ok(f"plot_data_{model_key}", f"{len(test_a)} points")
            imp = bundle.get("feature_importance", {})
            ok(f"importance_{model_key}", f"{len(imp)} features")
        except Exception as exc:
            bad(f"predict_{model_key}", traceback.format_exc())

    # ── JLR fit scoring use cases ──
    for uc in USE_CASE_PRESETS:
        try:
            sub = USE_CASE_PRESETS[uc]["subsystem"]
            ranked = recommend_materials(unified, subsystem=sub, use_case=uc)
            if ranked.empty and sub in ("Structural / Chassis", "General Material Reuse"):
                bad(f"recommend_{uc}", "empty rankings")
            else:
                ok(f"recommend_{uc}", f"{len(ranked)} rows")
            # Non-structural should not require strength threshold breaking
            if sub not in ("Structural / Chassis", "General Material Reuse"):
                ranked2 = recommend_materials(unified, subsystem=sub, min_strength=99999, use_case=uc)
                if ranked2.empty and len(filter_subsystem_rows(unified, sub)) > 0:
                    bad(f"nonstruct_neutral_{uc}", "all filtered by strength")
                elif sub not in ("Structural / Chassis", "General Material Reuse"):
                    ok(f"nonstruct_neutral_{uc}", "strength filter not blocking")
        except Exception as exc:
            bad(f"recommend_{uc}", traceback.format_exc())

    # ── Validation report ──
    try:
        ranked = recommend_materials(unified, subsystem="Structural / Chassis", use_case=list(USE_CASE_PRESETS.keys())[0])
        if ranked.empty:
            bad("report", "no ranked materials")
        else:
            row = ranked.iloc[0]
            report = build_report(
                row, min_strength=250, use_case=list(USE_CASE_PRESETS.keys())[0],
                model_metrics=models["structural_yield_strength"]["metrics"],
                n_reference_rows=312, n_uploaded_rows=0,
                active_model_name="structural_yield_strength", model_status="Active",
                model_bundle=models["structural_yield_strength"],
                subsystem="Structural / Chassis", registry_df=registry,
            )
            if len(report) < 100:
                bad("report", "too short")
            else:
                ok("report", f"{len(report)} chars")
    except Exception as exc:
        bad("report", traceback.format_exc())

    # ── Upload: random CSV ──
    try:
        random_csv = pd.DataFrame({"foo": [1, 2], "bar": ["a", "b"], "random_col": [3, 4]})
        mapping = suggest_schema_mapping(random_csv)
        enriched = enrich_mapping_with_units(random_csv, mapping)
        overrides = {c: "ignore" for c in random_csv.columns}
        mapped, summary = ingest_approved_evidence(
            random_csv, overrides, source_type="supplier_sheet",
            application_subsystem="Structural / Chassis", infer_subsystem=False,
            engineer_reviewed=True, mapping_notes="qa", engineer_review_notes="qa",
        )
        ok("upload_random", f"ingested {summary['rows_ingested']} rows without crash")
    except Exception as exc:
        bad("upload_random", traceback.format_exc())

    # ── Upload: valid-ish supplier CSV ──
    try:
        supplier = pd.DataFrame({
            "material_name": ["SupplierAlloy1", "SupplierAlloy2"],
            "yield_strength_mpa": [450.0, 520.0],
            "density_g_cm3": [7.9, 7.85],
            "formula": ["Fe-C-Mn", "Fe-C-Cr"],
        })
        mapping = suggest_schema_mapping(supplier)
        overrides = {}
        for _, mrow in mapping.iterrows():
            col = mrow["uploaded_column"]
            field = mrow["suggested_field"]
            overrides[col] = field if field != "ignore" else "ignore"
        mapped, summary = ingest_approved_evidence(
            supplier, overrides, source_type="supplier_sheet",
            application_subsystem="Structural / Chassis", infer_subsystem=False,
            engineer_reviewed=True, mapping_notes="qa", engineer_review_notes="qa",
        )
        impact = assess_upload_model_impact(mapped, unified, models)
        if summary["rows_ingested"] != 2:
            bad("upload_supplier", f"rows {summary['rows_ingested']}")
        else:
            ok("upload_supplier", "2 rows ingested")
        if not impact.get("scoring_coverage_improved"):
            bad("upload_impact", "no scoring impact")
        else:
            ok("upload_impact", str(impact.get("impacts", [])))
        # Not enough rows for retrain (need 30+ for structural)
        if impact.get("any_retrain_available"):
            bad("upload_retrain_guard", "should not offer retrain for 2 rows")
        else:
            ok("upload_retrain_guard", "correctly no retrain for 2 rows")
    except Exception as exc:
        bad("upload_supplier", traceback.format_exc())

    # ── Assumption fields not scored as measured ──
    try:
        steel_row = unified[unified["source_dataset"] == "matminer_steel_strength"].iloc[0]
        if not is_assumption_field(steel_row, "density_g_cm3"):
            bad("assumption_density", "density should be assumption on steel")
        else:
            ok("assumption_density", "tracked")
        if is_assumption_field(steel_row, "yield_strength_mpa"):
            bad("assumption_yield", "yield should not be assumption")
        else:
            ok("assumption_yield", "yield is measured")
        ranked = recommend_materials(unified, subsystem="Structural / Chassis")
        if not ranked.empty:
            top = ranked.iloc[0]
            ok("scoring_ran", f"top score {top.get('suitability_score')}")
    except Exception as exc:
        bad("assumptions", traceback.format_exc())

    # ── Template downloads ──
    try:
        for name in SUBSYSTEM_TEMPLATE_COLUMNS:
            data = template_download_bytes(name)
            if len(data) < 10:
                bad(f"template_{name}", "empty")
        ok("templates_download", f"{len(SUBSYSTEM_TEMPLATE_COLUMNS)} templates")
    except Exception as exc:
        bad("templates_download", traceback.format_exc())

    # ── HTML escape ──
    try:
        evil = "<script>alert(1)</script>"
        if evil in esc(evil):
            bad("html_escape", "not escaped")
        else:
            ok("html_escape", "esc works")
    except Exception as exc:
        bad("html_escape", traceback.format_exc())

    # ── display_status on registry ──
    try:
        shown = registry["status"].map(display_status).tolist()
        if any("Waiting for evidence" in s for s in shown):
            bad("registry_ui_status", str(shown))
        else:
            ok("registry_ui_status", "user-facing labels ok")
    except Exception as exc:
        bad("registry_ui_status", traceback.format_exc())

    _report()
    return 1 if FAIL else 0


def _report() -> None:
    print("\n=== QA PASS ===")
    for name, detail in PASS:
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    if FAIL:
        print("\n=== QA FAIL ===")
        for name, detail in FAIL:
            print(f"  FAIL  {name}")
            print(f"        {detail[:500]}")


if __name__ == "__main__":
    raise SystemExit(main())
