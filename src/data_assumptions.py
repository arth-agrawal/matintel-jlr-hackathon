"""Track engineering assumptions vs measured source values."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

ASSUMPTION_NOTE = "Default engineering assumption, not measured source value."
STEEL_DEFAULT_DENSITY_G_CM3 = 7.85

# Fields that were previously demo-filled on matminer steel — now NaN unless uploaded/sourced
QUARANTINED_STEEL_DEMO_FIELDS = [
    "recycled_content_percent", "recyclability_score", "co2_kg_per_kg",
    "supplier_risk_score", "cost_index", "co2_index", "bio_based_content_percent",
    "traceability_score", "critical_material_risk_score", "youngs_modulus_gpa",
]

# Only density kept as labelled engineering default for structural steel
STEEL_ASSUMPTION_FIELDS = ["density_g_cm3"]


def parse_assumption_fields(row: Any) -> set[str]:
    raw = row.get("assumption_fields") if hasattr(row, "get") else None
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or raw == "":
        return set()
    return {f.strip() for f in str(raw).split(",") if f.strip()}


def is_assumption_field(row: Any, field: str) -> bool:
    return field in parse_assumption_fields(row)


def is_assumption_field_series(df: pd.DataFrame, field: str) -> pd.Series:
    if "assumption_fields" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["assumption_fields"].fillna("").astype(str).apply(
        lambda s: field in {x.strip() for x in s.split(",") if x.strip()}
    )


def field_has_measured_data(df: pd.DataFrame, field: str, min_rows: int = 1) -> bool:
    """True if at least min_rows have non-null, non-assumption values."""
    if field not in df.columns:
        return False
    mask = df[field].notna() & ~is_assumption_field_series(df, field)
    return int(mask.sum()) >= min_rows


def field_has_measured_variance(df: pd.DataFrame, field: str, min_std: float = 0.01) -> bool:
    if field not in df.columns:
        return False
    mask = df[field].notna() & ~is_assumption_field_series(df, field)
    series = pd.to_numeric(df.loc[mask, field], errors="coerce").dropna()
    if len(series) < 2:
        return False
    return float(series.std()) >= min_std


def format_field_display(row: Any, field: str, val: Any = None, suffix: str = "") -> str:
    """HTML-safe display value with assumption / validation pill."""
    from src.schema_mapper import esc

    if val is None:
        val = row.get(field) if hasattr(row, "get") else None

    if is_assumption_field(row, field):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            text = "—"
        else:
            text = f"{esc(str(val))}{suffix}"
        return (
            f'{text} <span class="pill pill-amber">Assumption / validation needed</span>'
        )

    if val is None or (isinstance(val, float) and math.isnan(val)):
        return '<span class="pill pill-amber">Validation needed</span>'

    return f"{esc(str(val))}{suffix}"
