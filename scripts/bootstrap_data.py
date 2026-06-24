#!/usr/bin/env python3
"""Offline bootstrap: download matminer data, build unified schema, train model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import load_real_steel_data, build_unified_schema
from src.modeling import train_model

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

raw = load_real_steel_data()
raw.to_csv("data/raw/steel_strength_raw.csv", index=False)

unified = build_unified_schema(raw)
unified.to_csv("data/processed/unified_material_library.csv", index=False)

model_bundle = train_model(unified)
print("Saved raw dataset, unified dataset, and trained model.")
print("Model metrics:", model_bundle["metrics"])
