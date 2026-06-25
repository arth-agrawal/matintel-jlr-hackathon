#!/usr/bin/env python3
"""Offline bootstrap: steel data, public reference bundle, unified schema, train models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    load_real_steel_data, build_unified_schema, PROJECT_ROOT, PUBLIC_REF_PATH,
)
from src.modeling import train_all_models
from src.public_sources import build_default_reference_cache, summarize_sources

ROOT = PROJECT_ROOT
(ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(parents=True, exist_ok=True)

print("Loading steel...")
raw = load_real_steel_data()
raw.to_csv(ROOT / "data" / "raw" / "steel_strength_raw.csv", index=False)

steel_unified = build_unified_schema(raw)
steel_unified.to_csv(ROOT / "data" / "processed" / "unified_material_library.csv", index=False)
print(f"  Steel unified: {len(steel_unified)} rows")

print("Building trusted reference cache (JARVIS 5000 + matbench if cached)...")
ref_df, messages = build_default_reference_cache(jarvis_rows=5000, include_matbench_cache=True)
for msg in messages:
    print(f"  {msg}")

if not ref_df.empty:
    ref_df.to_csv(PUBLIC_REF_PATH, index=False)
    summary = summarize_sources(ref_df)
    print(f"Public reference saved: {summary['total']} rows")
    print(f"  By source: {summary['by_source']}")
else:
    print("No public reference rows loaded (JARVIS may be unavailable offline).")

import pandas as pd

training_unified = steel_unified if ref_df.empty else pd.concat(
    [steel_unified, ref_df], ignore_index=True,
)
print(f"Training property-specific models on {len(training_unified)} unified records...")
trained = train_all_models(training_unified)
for key, bundle in trained.items():
    print(
        f"  {key}: {bundle['selected_algorithm']} · "
        f"R² {bundle['r2']} · MAE {bundle['mae']} · {bundle['rows']} rows"
    )
if not trained:
    print("  No models trained (reference cache may be missing).")
print("Bootstrap complete.")
