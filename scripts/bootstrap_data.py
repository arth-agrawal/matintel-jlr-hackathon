#!/usr/bin/env python3
"""Offline bootstrap: steel data, public reference bundle, unified schema, train model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    load_real_steel_data, build_unified_schema, PROJECT_ROOT, PUBLIC_REF_PATH,
)
from src.modeling import train_model
from src.public_sources import load_trusted_public_bundle, summarize_sources

ROOT = PROJECT_ROOT
(ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(parents=True, exist_ok=True)

print("Loading steel...")
raw = load_real_steel_data()
raw.to_csv(ROOT / "data" / "raw" / "steel_strength_raw.csv", index=False)

unified = build_unified_schema(raw)
unified.to_csv(ROOT / "data" / "processed" / "unified_material_library.csv", index=False)
print(f"  Steel unified: {len(unified)} rows")

print("Loading JARVIS...")
ref_df, messages = load_trusted_public_bundle(
    mode="fast",
    include_jarvis=True,
    include_matbench=False,
    matbench_light_only=True,
)
for msg in messages:
    print(f"  {msg}")

if not ref_df.empty:
    ref_df.to_csv(PUBLIC_REF_PATH, index=False)
    summary = summarize_sources(ref_df)
    print(f"Public reference saved: {summary['total']} rows")
    print(f"  By source: {summary['by_source']}")
else:
    print("No public reference rows loaded (JARVIS may be unavailable offline).")

print("Training structural yield model (experimental steel only)...")
model_bundle = train_model(unified)
print("Model:", model_bundle.get("model_name"), "Metrics:", model_bundle["metrics"])
print("Bootstrap complete.")
