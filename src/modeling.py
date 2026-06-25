"""Property-specific prediction models with automatic algorithm selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.formula_features import add_formula_descriptor_columns, get_formula_feature_cols


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
# Legacy path kept for backward compatibility
MODEL_PATH = MODELS_DIR / "yield_strength_model.pkl"

EXPERIMENTAL_SOURCE_TYPES = {"public_experimental", "experimental_test"}
COMPUTED_SOURCE_TYPES = {"computed_database", "public_benchmark"}

MIN_ROWS_EXPERIMENTAL = 30
MIN_ROWS_COMPUTED = 100

MODEL_TRAINING_SPECS: dict[str, dict[str, Any]] = {
    "structural_yield_strength": {
        "target": "yield_strength_mpa",
        "model_family": "experimental",
        "feature_mode": "steel_composition",
        "eligible_source_types": list(EXPERIMENTAL_SOURCE_TYPES),
        "min_rows": MIN_ROWS_EXPERIMENTAL,
        "use_ml_training_flag": True,
        "subsystem": "Structural / Chassis",
        "recommendation_basis": "observed-data based screening",
        "limitation_label": "Measured-data prediction",
        "unit": "MPa",
    },
    "computed_band_gap": {
        "target": "band_gap_ev",
        "model_family": "computed_reference",
        "feature_mode": "formula_descriptor",
        "eligible_source_types": list(COMPUTED_SOURCE_TYPES),
        "min_rows": MIN_ROWS_COMPUTED,
        "subsystem": "Electronics / Thermal Interface",
        "recommendation_basis": "computed-reference based screening",
        "limitation_label": "Computed-reference screening model — validation required before engineering release",
        "unit": "eV",
    },
    "computed_formation_energy": {
        "target": "formation_energy_per_atom",
        "model_family": "computed_reference",
        "feature_mode": "formula_descriptor",
        "eligible_source_types": list(COMPUTED_SOURCE_TYPES),
        "min_rows": MIN_ROWS_COMPUTED,
        "subsystem": "Battery Enclosure / Underbody / General Material Reuse",
        "recommendation_basis": "computed-reference based screening",
        "limitation_label": "Computed-reference screening model — validation required before engineering release",
        "unit": "eV/atom",
    },
    "elastic_modulus_proxy": {
        "target": "bulk_modulus_gpa",
        "alt_target": "shear_modulus_gpa",
        "model_family": "computed_reference",
        "feature_mode": "formula_descriptor",
        "eligible_source_types": list(COMPUTED_SOURCE_TYPES),
        "min_rows": MIN_ROWS_COMPUTED,
        "subsystem": "Structural reference / General Material Reuse",
        "recommendation_basis": "computed-reference based screening",
        "limitation_label": "Computed-reference screening model — validation required before engineering release",
        "unit": "GPa",
    },
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("wt_percent_")]


def _model_path(model_key: str) -> Path:
    return MODELS_DIR / f"{model_key}.pkl"


def _build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _candidate_estimators() -> list[tuple[str, object]]:
    return [
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1,
            ),
        ),
        (
            "ExtraTreesRegressor",
            ExtraTreesRegressor(
                n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1,
            ),
        ),
        (
            "HistGradientBoostingRegressor",
            HistGradientBoostingRegressor(random_state=42, max_iter=200),
        ),
    ]


def _eligible_training_frame(
    unified: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Return rows eligible for a specific model target."""
    target = spec["target"]
    alt_target = spec.get("alt_target")
    eligible_types = set(spec["eligible_source_types"])
    min_rows = int(spec.get("min_rows", MIN_ROWS_COMPUTED))

    df = unified.copy()
    if target not in df.columns:
        return pd.DataFrame(), target

    mask = df[target].notna()
    if alt_target and alt_target in df.columns:
        mask = mask | df[alt_target].notna()

    if "source_type" in df.columns:
        mask = mask & df["source_type"].isin(eligible_types)

    if spec.get("use_ml_training_flag") and "used_for_ml_training" in df.columns:
        ml_flag = df["used_for_ml_training"].astype(str).str.lower().isin(["true", "1", "yes"])
        mask = mask & ml_flag

    if "model_registry_eligible" in df.columns:
        eligible_flag = df["model_registry_eligible"].astype(str).str.lower().isin(["true", "1", "yes"])
        mask = mask & eligible_flag

    train_df = df[mask].copy()
    if train_df.empty:
        return train_df, target

    # Unified target for elastic modulus proxy
    if alt_target and alt_target in train_df.columns:
        primary = pd.to_numeric(train_df[target], errors="coerce")
        secondary = pd.to_numeric(train_df[alt_target], errors="coerce")
        train_df["_training_target"] = primary.where(primary.notna(), secondary)
        effective_target = "_training_target"
    else:
        train_df["_training_target"] = pd.to_numeric(train_df[target], errors="coerce")
        effective_target = "_training_target"

    train_df = train_df[train_df["_training_target"].notna()].copy()
    if len(train_df) < min_rows:
        return pd.DataFrame(), effective_target

    return train_df, effective_target


def _prepare_features(
    train_df: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    mode = spec["feature_mode"]
    if mode == "steel_composition":
        feature_cols = get_feature_cols(train_df)
        if not feature_cols:
            raise ValueError("No composition feature columns (wt_percent_*) found.")
        return train_df[feature_cols], feature_cols

    enriched = add_formula_descriptor_columns(train_df)
    feature_cols = get_formula_feature_cols(enriched)
    if len(feature_cols) < 3:
        raise ValueError("Insufficient formula descriptor features.")
    return enriched[feature_cols], feature_cols


def _select_best_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[str, Pipeline, dict, np.ndarray, np.ndarray]:
    best_name = "RandomForestRegressor"
    best_pipe: Pipeline | None = None
    best_metrics: dict | None = None
    best_pred: np.ndarray | None = None

    for name, estimator in _candidate_estimators():
        try:
            pipe = _build_pipeline(estimator)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            rmse = float(root_mean_squared_error(y_test, pred))
            r2 = float(r2_score(y_test, pred))
            mae = float(mean_absolute_error(y_test, pred))
            metrics = {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }
            if best_metrics is None or rmse < best_metrics["RMSE"] or (
                rmse == best_metrics["RMSE"] and r2 > best_metrics["R2"]
            ):
                best_name = name
                best_pipe = pipe
                best_metrics = metrics
                best_pred = pred
        except Exception:
            continue

    if best_pipe is None or best_metrics is None or best_pred is None:
        pipe = _build_pipeline(RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1))
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        best_pipe = pipe
        best_name = "RandomForestRegressor"
        best_metrics = {
            "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
            "RMSE": round(float(root_mean_squared_error(y_test, pred)), 4),
            "R2": round(float(r2_score(y_test, pred)), 4),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        best_pred = pred

    return best_name, best_pipe, best_metrics, y_test.to_numpy(), best_pred


def _feature_importance(pipe: Pipeline, feature_cols: list[str]) -> dict[str, float]:
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
        return {feature_cols[i]: round(float(vals[i]), 4) for i in range(len(feature_cols))}
    return {}


def train_property_model(model_key: str, unified: pd.DataFrame) -> dict | None:
    """Train one property-specific model. Returns None if insufficient data."""
    if model_key not in MODEL_TRAINING_SPECS:
        raise ValueError(f"Unknown model key: {model_key}")

    spec = MODEL_TRAINING_SPECS[model_key]
    train_df, effective_target = _eligible_training_frame(unified, spec)
    if train_df.empty or len(train_df) < int(spec.get("min_rows", MIN_ROWS_COMPUTED)):
        return None

    try:
        X, feature_cols = _prepare_features(train_df, spec)
    except ValueError:
        return None

    y = train_df[effective_target]
    valid = y.notna() & X.notna().any(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]
    if len(X) < int(spec.get("min_rows", MIN_ROWS_COMPUTED)):
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42,
    )

    best_name, best_pipe, best_metrics, y_test_arr, y_pred_arr = _select_best_model(
        X_train, X_test, y_train, y_test,
    )

    est = best_pipe.named_steps["model"]
    interval_method = "tree_spread" if hasattr(est, "estimators_") else "residual_rmse"
    source_datasets = (
        sorted(train_df["source_dataset"].dropna().astype(str).unique().tolist())
        if "source_dataset" in train_df.columns else []
    )

    bundle: dict[str, Any] = {
        "model_key": model_key,
        "model": best_pipe,
        "model_name": best_name,
        "selected_algorithm": best_name,
        "target": spec["target"],
        "effective_target": effective_target,
        "feature_cols": feature_cols,
        "features": feature_cols,
        "metrics": best_metrics,
        "r2": best_metrics["R2"],
        "mae": best_metrics["MAE"],
        "rmse": best_metrics["RMSE"],
        "rows": int(len(train_df)),
        "model_family": spec["model_family"],
        "subsystem": spec.get("subsystem", ""),
        "source_datasets": source_datasets,
        "recommendation_basis": spec.get("recommendation_basis", ""),
        "limitation_label": spec.get("limitation_label", ""),
        "unit": spec.get("unit", ""),
        "train_feature_mean": X_train.mean(numeric_only=True).to_dict(),
        "train_feature_std": X_train.std(numeric_only=True).replace(0, 1).to_dict(),
        "interval_method": interval_method,
        "feature_importance": _feature_importance(best_pipe, feature_cols),
        "test_actuals": y_test_arr.tolist(),
        "test_predictions": y_pred_arr.tolist(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, _model_path(model_key))
    if model_key == "structural_yield_strength":
        joblib.dump(bundle, MODEL_PATH)
    return bundle


def train_model(unified: pd.DataFrame) -> dict:
    """Backward-compatible: train structural yield strength only."""
    bundle = train_property_model("structural_yield_strength", unified)
    if bundle is None:
        raise ValueError("Insufficient data to train structural yield strength model.")
    return bundle


def train_all_models(unified: pd.DataFrame) -> dict[str, dict]:
    """Train all property-specific models where valid targets exist."""
    trained: dict[str, dict] = {}
    metadata: list[dict[str, Any]] = []

    for model_key in MODEL_TRAINING_SPECS:
        try:
            bundle = train_property_model(model_key, unified)
        except Exception:
            bundle = None
        if bundle is not None:
            trained[model_key] = bundle
            metadata.append(_bundle_metadata(bundle))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(metadata, indent=2))
    return trained


def _bundle_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_key": bundle["model_key"],
        "target": bundle["target"],
        "model_family": bundle["model_family"],
        "rows": bundle["rows"],
        "features": bundle["features"],
        "selected_algorithm": bundle["selected_algorithm"],
        "r2": bundle["r2"],
        "mae": bundle["mae"],
        "rmse": bundle["rmse"],
        "source_datasets": bundle.get("source_datasets", []),
        "recommendation_basis": bundle.get("recommendation_basis", ""),
        "subsystem": bundle.get("subsystem", ""),
        "limitation_label": bundle.get("limitation_label", ""),
        "timestamp": bundle.get("timestamp", ""),
    }


def load_trained_model(model_key: str) -> dict | None:
    path = _model_path(model_key)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        bundle.setdefault("model_key", model_key)
        bundle.setdefault("selected_algorithm", bundle.get("model_name", "RandomForestRegressor"))
        return bundle
    except Exception:
        return None


def load_model_registry() -> list[dict[str, Any]]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            pass
    return []


def load_all_trained_models() -> dict[str, dict]:
    """Load all saved property-specific models."""
    models: dict[str, dict] = {}
    for model_key in MODEL_TRAINING_SPECS:
        bundle = load_trained_model(model_key)
        if bundle is not None and bundle.get("metrics"):
            models[model_key] = bundle
    if not models and MODEL_PATH.exists():
        try:
            legacy = joblib.load(MODEL_PATH)
            legacy["model_key"] = "structural_yield_strength"
            models["structural_yield_strength"] = legacy
        except Exception:
            pass
    return models


def load_or_train_models(unified: pd.DataFrame) -> dict[str, dict]:
    models = load_all_trained_models()
    if models:
        return models
    return train_all_models(unified)


def load_or_train_model(unified: pd.DataFrame) -> dict:
    models = load_or_train_models(unified)
    if "structural_yield_strength" in models:
        return models["structural_yield_strength"]
    return train_model(unified)


def count_trained_models(models: dict[str, dict] | None = None) -> dict[str, int]:
    if models is None:
        models = load_all_trained_models()
    experimental = sum(
        1 for b in models.values() if b.get("model_family") == "experimental"
    )
    computed = sum(
        1 for b in models.values() if b.get("model_family") == "computed_reference"
    )
    return {
        "active_trained_models": len(models),
        "experimental_trained_models": experimental,
        "computed_reference_trained_models": computed,
    }


def predict_with_interval(bundle: dict, row: pd.Series) -> dict:
    """Predict with tree spread or residual RMSE fallback interval."""
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    target = bundle.get("target", "yield_strength_mpa")

    if bundle.get("model_family") == "computed_reference":
        enriched = add_formula_descriptor_columns(pd.DataFrame([row]))
        X = enriched.reindex(columns=feature_cols)
    else:
        X = pd.DataFrame([row.reindex(feature_cols)])

    pred = float(model.predict(X)[0])
    interval_method = bundle.get("interval_method", "residual_rmse")
    rmse = float(bundle.get("metrics", {}).get("RMSE", bundle.get("rmse", 50)))

    imputed = model.named_steps["imputer"].transform(X)
    scaled = model.named_steps["scaler"].transform(imputed)
    est = model.named_steps.get("model") or model.named_steps.get("rf")

    if interval_method == "tree_spread" and hasattr(est, "estimators_"):
        tree_preds = np.array([tree.predict(scaled)[0] for tree in est.estimators_])
        lower = float(np.percentile(tree_preds, 10))
        upper = float(np.percentile(tree_preds, 90))
    else:
        lower = pred - 1.28 * rmse
        upper = pred + 1.28 * rmse

    actual = row.get(target)
    if actual is None or (isinstance(actual, float) and np.isnan(actual)):
        alt = bundle.get("alt_target")
        if alt:
            actual = row.get(alt)
    actual_val = float(actual) if actual is not None and pd.notna(actual) else None

    return {
        "prediction": round(pred, 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "uncertainty_width": round(upper - lower, 4),
        "actual": round(actual_val, 4) if actual_val is not None else None,
        "interval_method": interval_method,
    }
