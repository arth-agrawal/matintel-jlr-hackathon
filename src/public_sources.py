"""Public dataset registry and trusted loaders for MatIntel.

Loads real experimental, computed, and benchmark datasets — no fabricated rows.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_JARVIS = RAW_DIR / "jarvis_dft_3d.csv"
CACHE_MATBENCH_COMBINED = RAW_DIR / "matminer_extra_combined.csv"
CACHE_MATBENCH_DIR = RAW_DIR / "matbench_cache"
CACHE_MATMINER_META = RAW_DIR / "matminer_extra_meta.json"

MATBENCH_SKIP_FAST_MSG = (
    "Matbench extra dataset skipped in fast mode; load from UI if needed."
)

# Datasets that trigger slow pymatgen Structure decoding via matminer
MATBENCH_HEAVY_DATASETS = {
    "matbench_dielectric",
    "matbench_log_gvrh",
    "matbench_log_kvrh",
    "matbench_mp_gap",
    "matbench_mp_e_form",
    "matbench_expt_gap",
}

RECOMMENDATION_COMPUTED = "computed-reference based screening"
RECOMMENDATION_BENCHMARK = "benchmark/reference screening"
RECOMMENDATION_EXPERIMENTAL = "observed-data based screening"

PUBLIC_DATASET_REGISTRY: dict[str, dict] = {
    "matminer_steel_strength": {
        "name": "Matminer steel_strength / matbench_steels",
        "source_type": "public_experimental",
        "application_subsystem": "Structural / Chassis",
        "trust_score": 95,
        "role": "experimental training",
        "used_for_ml_training": True,
        "trainable_targets": ["yield_strength_mpa", "ultimate_tensile_strength_mpa"],
        "status": "active",
        "description": "Experimental steel alloy strength — trains active structural yield model.",
    },
    "jarvis_dft_3d": {
        "name": "JARVIS-DFT dft_3d",
        "source_type": "computed_database",
        "application_subsystem": (
            "General Material Reuse · Battery Enclosure / Underbody · "
            "Electronics / Thermal Interface · Structural reference"
        ),
        "trust_score": 85,
        "role": "computed reference + computed model training",
        "used_for_ml_training": False,
        "trainable_targets": [
            "formation_energy_per_atom", "band_gap_ev",
            "bulk_modulus_gpa", "shear_modulus_gpa", "density_g_cm3",
        ],
        "status": "optional",
        "description": "Broad DFT-computed inorganic materials reference from NIST JARVIS.",
    },
    "matminer_matbench_extra": {
        "name": "Matminer / Matbench property benchmarks",
        "source_type": "public_benchmark",
        "application_subsystem": "Multi-subsystem (property-dependent)",
        "trust_score": 85,
        "role": "benchmark / property reference models",
        "used_for_ml_training": False,
        "trainable_targets": [
            "band_gap_ev", "formation_energy_per_atom",
            "dielectric_constant", "bulk_modulus_gpa", "shear_modulus_gpa",
        ],
        "status": "optional",
        "description": "Matbench dielectric, modulus, band gap, formation energy datasets.",
    },
    "nasa_tpsx_reference": {
        "name": "NASA TPSX reference / upload pathway",
        "source_type": "public_reference",
        "application_subsystem": (
            "Thermal Fluids / Coolants · Coatings / Corrosion Protection · "
            "Interior / Insulation / Composites"
        ),
        "trust_score": 80,
        "role": "reference / upload pathway",
        "used_for_ml_training": False,
        "trainable_targets": ["thermal_conductivity_w_mk", "specific_heat", "density_g_cm3"],
        "status": "pathway",
        "description": "Import NASA TPSX CSV exports for thermal, coatings, composites screening.",
    },
    "engineer_reviewed_upload": {
        "name": "Engineer-reviewed evidence upload",
        "source_type": "experimental_test",
        "application_subsystem": "All subsystems (mapped by engineer)",
        "trust_score": 90,
        "role": "JLR / supplier / internal evidence",
        "used_for_ml_training": False,
        "trainable_targets": ["subsystem-specific labelled properties"],
        "status": "pathway",
        "description": "Upload supplier sheets, test reports, sustainability or procurement data.",
    },
}

MATBENCH_DATASETS: list[dict[str, Any]] = [
    {
        "name": "matbench_dielectric",
        "source_dataset": "matbench_dielectric",
        "source_type": "public_benchmark",
        "target_col": ["n", "dielectric", "dielectric_const"],
        "map_to": "dielectric_constant",
        "subsystem": "Electronics / Thermal Interface",
    },
    {
        "name": "matbench_log_gvrh",
        "source_dataset": "matbench_log_gvrh",
        "source_type": "public_benchmark",
        "target_col": ["g_vrh", "gv", "shear_modulus"],
        "map_to": "shear_modulus_gpa",
        "log10_transform": True,
        "subsystem": "General Material Reuse",
    },
    {
        "name": "matbench_log_kvrh",
        "source_dataset": "matbench_log_kvrh",
        "source_type": "public_benchmark",
        "target_col": ["k_vrh", "kv", "bulk_modulus"],
        "map_to": "bulk_modulus_gpa",
        "log10_transform": True,
        "subsystem": "General Material Reuse",
    },
    {
        "name": "matbench_mp_gap",
        "source_dataset": "matbench_mp_gap",
        "source_type": "computed_database",
        "target_col": ["gap pbe", "gap_pbe", "bandgap", "gap"],
        "map_to": "band_gap_ev",
        "subsystem": "Electronics / Thermal Interface",
    },
    {
        "name": "matbench_mp_e_form",
        "source_dataset": "matbench_mp_e_form",
        "source_type": "computed_database",
        "target_col": ["e_form", "formation_energy", "formation energy"],
        "map_to": "formation_energy_per_atom",
        "subsystem": "Battery Enclosure / Underbody",
    },
    {
        "name": "matbench_expt_gap",
        "source_dataset": "matbench_expt_gap",
        "source_type": "public_benchmark",
        "target_col": ["gap expt", "gap_expt", "bandgap"],
        "map_to": "band_gap_ev",
        "subsystem": "Electronics / Thermal Interface",
    },
    {
        "name": "matbench_steels",
        "source_dataset": "matbench_steels",
        "source_type": "public_experimental",
        "target_col": ["yield strength", "yield_strength", "ys"],
        "map_to": "yield_strength_mpa",
        "subsystem": "Structural / Chassis",
        "skip_if_duplicate": True,
    },
]


def _safe_float(val) -> float:
    if val is None:
        return np.nan
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _normalize_key(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {_normalize_key(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_key(cand)
        if key in norm_map:
            return norm_map[key]
    for key, original in norm_map.items():
        for cand in candidates:
            if _normalize_key(cand) in key:
                return original
    return None


def infer_jarvis_subsystem(
    band_gap: float,
    formation_energy: float,
    bulk: float,
    shear: float,
    density: float,
) -> str:
    """Logical subsystem tag from available computed properties — not automotive validation."""
    bg = _safe_float(band_gap)
    fe = _safe_float(formation_energy)
    b = _safe_float(bulk)
    s = _safe_float(shear)
    d = _safe_float(density)

    if not np.isnan(bg) and bg > 0:
        return "Electronics / Thermal Interface"
    if not np.isnan(fe):
        return "Battery Enclosure / Underbody"
    if (not np.isnan(b) or not np.isnan(s)) and not np.isnan(d):
        return "General Material Reuse"
    return "General Material Reuse"


def _map_jarvis_entry(entry: dict, idx: int) -> dict:
    jid = str(entry.get("jid", f"JARVIS_{idx:06d}"))
    formula = str(entry.get("formula", entry.get("reduced_formula", jid)))
    density = _safe_float(entry.get("density"))
    bulk = _safe_float(entry.get("bulk_modulus_kv", entry.get("kv")))
    shear = _safe_float(entry.get("shear_modulus_gv", entry.get("gv")))
    form_e = _safe_float(entry.get("formation_energy_peratom", entry.get("form_enp")))
    bandgap = _safe_float(entry.get("optb88vdw_bandgap", entry.get("mbj_bandgap")))
    dielectric = _safe_float(entry.get("epsx", entry.get("dielectric")))

    subsystem = infer_jarvis_subsystem(bandgap, form_e, bulk, shear, density)

    row: dict[str, Any] = {
        "material_id": jid,
        "material_name": formula,
        "formula": formula,
        "material_family": "Inorganic compound",
        "material_subfamily": "DFT computed",
        "family": "Inorganic compound",
        "source_dataset": "jarvis_dft_3d",
        "source_type": "computed_database",
        "source_trust_score": 85,
        "used_for_ml_training": False,
        "model_registry_eligible": True,
        "application_subsystem": subsystem,
        "recommendation_basis": RECOMMENDATION_COMPUTED,
        "engineer_reviewed": False,
        "density_g_cm3": density,
        "bulk_modulus_gpa": bulk,
        "shear_modulus_gpa": shear,
        "formation_energy_per_atom": form_e,
        "band_gap_ev": bandgap,
        "dielectric_constant": dielectric,
        "yield_strength_mpa": np.nan,
        "ultimate_tensile_strength_mpa": np.nan,
        "elongation_percent": np.nan,
        "youngs_modulus_gpa": np.nan,
        "thermal_conductivity_w_mk": np.nan,
        "specific_heat": np.nan,
        "notes": "JARVIS DFT computed reference; not experimental validation. Not used for experimental strength model.",
    }
    return row


def load_jarvis_dft(max_rows: int | None = 500, cache: bool = True) -> tuple[pd.DataFrame, str]:
    """Load JARVIS-DFT dft_3d with optional disk cache."""
    if cache and CACHE_JARVIS.exists() and max_rows is not None:
        try:
            cached = pd.read_csv(CACHE_JARVIS)
            if len(cached) >= max_rows:
                return cached.head(max_rows).copy(), f"Loaded {min(max_rows, len(cached))} JARVIS rows from cache."
        except Exception:
            pass

    try:
        from jarvis.db.figshare import data as jdata
    except ImportError:
        if cache and CACHE_JARVIS.exists():
            cached = pd.read_csv(CACHE_JARVIS)
            n = len(cached) if max_rows is None else min(max_rows, len(cached))
            return cached.head(n).copy(), f"jarvis-tools unavailable — loaded {n} rows from cache."
        return pd.DataFrame(), "jarvis-tools not installed. Run: pip install jarvis-tools"

    try:
        raw = jdata("dft_3d")
    except Exception as e:
        if cache and CACHE_JARVIS.exists():
            cached = pd.read_csv(CACHE_JARVIS)
            n = len(cached) if max_rows is None else min(max_rows, len(cached))
            return cached.head(n).copy(), f"JARVIS download failed ({e}) — using {n} cached rows."
        return pd.DataFrame(), f"JARVIS data download failed: {e}"

    if not raw:
        return pd.DataFrame(), "JARVIS returned empty data."

    if max_rows is not None:
        raw = raw[:max_rows]

    records = [_map_jarvis_entry(entry, i) for i, entry in enumerate(raw)]
    df = pd.DataFrame(records)
    df = _add_completeness(df)

    if cache and not df.empty:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_JARVIS, index=False)

    return df, f"Loaded {len(df)} JARVIS-DFT rows (computed reference)."


def load_jarvis_dft_sample(max_rows: int = 500) -> tuple[pd.DataFrame, str]:
    """Backward-compatible alias."""
    return load_jarvis_dft(max_rows=max_rows, cache=True)


def _composition_to_formula(comp) -> str:
    if comp is None or (isinstance(comp, float) and np.isnan(comp)):
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, dict):
        parts = [f"{k}{v}" for k, v in comp.items()]
        return "".join(parts)
    return str(comp)


def _map_matbench_dataset(raw: pd.DataFrame, cfg: dict, max_rows: int) -> pd.DataFrame:
    df = raw.head(max_rows).copy()
    target_col = _find_column(df, cfg["target_col"])
    if target_col is None:
        return pd.DataFrame()

    comp_col = _find_column(df, ["composition", "formula", "cif", "material_id"])
    records = []
    map_field = cfg["map_to"]
    prefix = cfg["source_dataset"]

    for i, row in df.iterrows():
        val = _safe_float(row[target_col])
        if np.isnan(val):
            continue
        if cfg.get("log10_transform"):
            val = 10 ** val

        formula = ""
        if comp_col:
            formula = _composition_to_formula(row[comp_col])
        if not formula:
            formula = f"{prefix}_{i}"

        rec: dict[str, Any] = {
            "material_id": f"{prefix}_{i:05d}",
            "material_name": formula,
            "formula": formula,
            "material_family": "Matbench material",
            "material_subfamily": prefix,
            "family": "Matbench material",
            "source_dataset": prefix,
            "source_type": cfg["source_type"],
            "source_trust_score": 85,
            "used_for_ml_training": False,
            "model_registry_eligible": True,
            "application_subsystem": cfg.get("subsystem", "General Material Reuse"),
            "recommendation_basis": (
                RECOMMENDATION_BENCHMARK if cfg["source_type"] == "public_benchmark"
                else RECOMMENDATION_COMPUTED
            ),
            "engineer_reviewed": False,
            map_field: val,
            "notes": f"Matbench reference row from {prefix}; benchmark/reference screening.",
        }
        records.append(rec)

    return pd.DataFrame(records) if records else pd.DataFrame()


def _load_matminer_dataset_live(name: str, timeout_seconds: int) -> pd.DataFrame | None:
    """Load one matminer dataset with a wall-clock timeout (structure decode can be very slow)."""

    def _work() -> pd.DataFrame:
        from matminer.datasets import load_dataset

        return load_dataset(name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_work)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return None
        except Exception:
            return None


def _load_matbench_from_per_dataset_cache(
    cfg: dict,
    max_rows: int,
) -> tuple[pd.DataFrame, str | None]:
    cache_path = CACHE_MATBENCH_DIR / f"{cfg['name']}.csv"
    if not cache_path.exists():
        return pd.DataFrame(), None
    try:
        cached = pd.read_csv(cache_path)
        if max_rows and len(cached) > max_rows:
            cached = cached.head(max_rows).copy()
        if cached.empty:
            return pd.DataFrame(), None
        return cached, f"{cfg['name']}({len(cached)}) from cache"
    except Exception:
        return pd.DataFrame(), None


def _load_matbench_from_combined_cache(max_rows_per_dataset: int) -> tuple[pd.DataFrame, str]:
    if not CACHE_MATBENCH_COMBINED.exists():
        return pd.DataFrame(), MATBENCH_SKIP_FAST_MSG
    try:
        combined = pd.read_csv(CACHE_MATBENCH_COMBINED)
    except Exception as e:
        return pd.DataFrame(), f"Matbench combined cache unreadable: {e}"

    if combined.empty or "source_dataset" not in combined.columns:
        return pd.DataFrame(), MATBENCH_SKIP_FAST_MSG

    parts: list[pd.DataFrame] = []
    loaded_names: list[str] = []
    for cfg in MATBENCH_DATASETS:
        if cfg.get("skip_if_duplicate"):
            continue
        subset = combined[combined["source_dataset"] == cfg["source_dataset"]]
        if subset.empty:
            continue
        if max_rows_per_dataset and len(subset) > max_rows_per_dataset:
            subset = subset.head(max_rows_per_dataset).copy()
        parts.append(subset)
        loaded_names.append(f"{cfg['name']}({len(subset)})")

    if not parts:
        return pd.DataFrame(), MATBENCH_SKIP_FAST_MSG

    out = pd.concat(parts, ignore_index=True)
    return out, f"Loaded matbench extras from cache: {', '.join(loaded_names)}"


def load_matminer_extra_datasets(
    max_rows_per_dataset: int = 500,
    *,
    require_cache: bool = False,
    matbench_light_only: bool = True,
    timeout_seconds: int = 120,
) -> tuple[pd.DataFrame, str]:
    """Load matbench property datasets — cache-first; live matminer is optional and bounded."""
    if require_cache or matbench_light_only:
        combined_cached, cache_msg = _load_matbench_from_combined_cache(max_rows_per_dataset)
        if not combined_cached.empty:
            return combined_cached, cache_msg

    frames: list[pd.DataFrame] = []
    loaded_names: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    allow_live = not require_cache and not matbench_light_only

    for cfg in MATBENCH_DATASETS:
        if cfg.get("skip_if_duplicate"):
            continue

        mapped, cache_label = _load_matbench_from_per_dataset_cache(cfg, max_rows_per_dataset)
        if cache_label:
            frames.append(mapped)
            loaded_names.append(cache_label)
            continue

        if not allow_live:
            skipped.append(cfg["name"])
            continue

        raw = _load_matminer_dataset_live(cfg["name"], timeout_seconds=timeout_seconds)
        if raw is None:
            if cfg["name"] in MATBENCH_HEAVY_DATASETS:
                errors.append(f"{cfg['name']}: timed out or failed (heavy Structure decode)")
            else:
                errors.append(f"{cfg['name']}: timed out or failed")
            continue

        mapped = _map_matbench_dataset(raw, cfg, max_rows_per_dataset)
        if mapped.empty:
            errors.append(f"{cfg['name']}: no mappable rows")
            continue

        CACHE_MATBENCH_DIR.mkdir(parents=True, exist_ok=True)
        mapped.to_csv(CACHE_MATBENCH_DIR / f"{cfg['name']}.csv", index=False)
        frames.append(mapped)
        loaded_names.append(f"{cfg['name']}({len(mapped)})")

    if not frames:
        if skipped and not allow_live:
            return pd.DataFrame(), MATBENCH_SKIP_FAST_MSG
        msg = "No matbench extras loaded."
        if errors:
            msg += " " + "; ".join(errors[:4])
        return pd.DataFrame(), msg

    combined = pd.concat(frames, ignore_index=True)
    combined = _add_completeness(combined)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CACHE_MATBENCH_COMBINED, index=False)
    CACHE_MATMINER_META.write_text(
        json.dumps({"loaded": loaded_names, "skipped": skipped, "errors": errors[:10]})
    )

    msg = f"Loaded matbench extras: {', '.join(loaded_names)}"
    if errors:
        msg += f" (skipped/failed: {'; '.join(errors[:3])})"
    return combined, msg


def load_trusted_public_bundle(
    mode: str = "fast",
    *,
    include_jarvis: bool = True,
    include_matbench: bool = False,
    matbench_light_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load reference datasets bundle. Returns (dataframe, status_messages)."""
    messages: list[str] = []
    frames: list[pd.DataFrame] = []

    if mode == "fast":
        jarvis_n = 500
        matbench_n = 500
    elif mode == "broad":
        jarvis_n = 5000
        matbench_n = 2000
    else:  # full
        jarvis_n = None
        matbench_n = 2000

    if include_jarvis:
        jarvis_df, jarvis_msg = load_jarvis_dft(max_rows=jarvis_n, cache=True)
        messages.append(jarvis_msg)
        if not jarvis_df.empty:
            frames.append(jarvis_df)
    else:
        messages.append("JARVIS skipped.")

    if include_matbench:
        if matbench_light_only:
            matbench_df, matbench_msg = _load_matbench_from_combined_cache(matbench_n)
            if matbench_df.empty:
                matbench_df, matbench_msg = load_matminer_extra_datasets(
                    max_rows_per_dataset=matbench_n,
                    require_cache=True,
                    matbench_light_only=True,
                )
        else:
            matbench_df, matbench_msg = load_matminer_extra_datasets(
                max_rows_per_dataset=matbench_n,
                require_cache=False,
                matbench_light_only=False,
                timeout_seconds=180 if mode == "broad" else 300,
            )
        messages.append(matbench_msg)
        if not matbench_df.empty:
            frames.append(matbench_df)
    else:
        messages.append("Skipping Matbench extras in fast bootstrap.")

    if not frames:
        return pd.DataFrame(), messages

    combined = pd.concat(frames, ignore_index=True)
    return combined, messages


def load_nasa_tpsx_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Import NASA TPSX-style CSV export into unified schema."""
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        return pd.DataFrame(), f"Could not read TPSX CSV: {e}"

    records = []
    for i, row in raw.iterrows():
        name = str(row.get("material_name", row.get("name", f"TPSX_{i:04d}")))
        rec: dict[str, Any] = {
            "material_id": f"TPSX_{i:04d}",
            "material_name": name,
            "formula": str(row.get("formula", name)),
            "source_dataset": "nasa_tpsx_reference",
            "source_type": "public_reference",
            "source_trust_score": 80,
            "used_for_ml_training": False,
            "model_registry_eligible": True,
            "application_subsystem": "Thermal Fluids / Coolants",
            "recommendation_basis": "reference-only screening",
            "engineer_reviewed": False,
            "notes": "NASA TPSX reference import; engineer review recommended.",
        }
        for col in raw.columns:
            key = _normalize_key(col)
            val = row[col]
            if key in ("thermal_conductivity", "thermal_conductivity_w_mk", "k"):
                rec["thermal_conductivity_w_mk"] = _safe_float(val)
            elif key in ("density", "density_g_cm3"):
                rec["density_g_cm3"] = _safe_float(val)
            elif key in ("specific_heat", "cp"):
                rec["specific_heat"] = _safe_float(val)
        records.append(rec)

    df = pd.DataFrame(records)
    return _add_completeness(df), f"Imported {len(df)} NASA TPSX reference rows."


def _add_completeness(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        c for c in [
            "density_g_cm3", "bulk_modulus_gpa", "shear_modulus_gpa",
            "formation_energy_per_atom", "band_gap_ev", "dielectric_constant",
            "thermal_conductivity_w_mk", "yield_strength_mpa",
        ] if c in df.columns
    ]
    if key_cols:
        df = df.copy()
        df["data_completeness_score"] = (
            100 * (1 - df[key_cols].isna().mean(axis=1))
        ).round(1)
    else:
        df = df.copy()
        df["data_completeness_score"] = 0.0
    return df


def summarize_sources(df: pd.DataFrame) -> dict[str, Any]:
    """Summary stats for UI metrics."""
    if df.empty:
        return {"total": 0, "by_source": {}, "by_subsystem": {}, "computed": 0, "experimental": 0}

    by_source = df["source_dataset"].value_counts().to_dict() if "source_dataset" in df.columns else {}
    by_subsystem = df["application_subsystem"].value_counts().to_dict() if "application_subsystem" in df.columns else {}

    exp_types = {"public_experimental", "experimental_test"}
    comp_types = {"computed_database", "public_benchmark"}

    experimental = 0
    computed = 0
    if "source_type" in df.columns:
        experimental = int(df["source_type"].isin(exp_types).sum())
        computed = int(df["source_type"].isin(comp_types).sum())

    ml_rows = 0
    if "used_for_ml_training" in df.columns:
        ml_rows = int(df["used_for_ml_training"].astype(str).str.lower().isin(["true", "1"]).sum())

    return {
        "total": len(df),
        "by_source": by_source,
        "by_subsystem": by_subsystem,
        "computed": computed,
        "experimental": experimental,
        "ml_training_rows": ml_rows,
    }


def jarvis_property_coverage(df: pd.DataFrame) -> dict[str, int]:
    """Count non-null JARVIS property fields."""
    jarvis = df[df["source_dataset"].astype(str).str.startswith("jarvis")] if "source_dataset" in df.columns else df
    fields = [
        "density_g_cm3", "band_gap_ev", "formation_energy_per_atom",
        "bulk_modulus_gpa", "shear_modulus_gpa", "dielectric_constant",
    ]
    return {f: int(jarvis[f].notna().sum()) for f in fields if f in jarvis.columns}
