"""Property prediction models with automatic algorithm selection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yield_strength_model.pkl"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("wt_percent_")]


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


def train_model(unified: pd.DataFrame) -> dict:
    """Train yield strength model; pick best tabular algorithm by test RMSE."""
    df = unified.dropna(subset=["yield_strength_mpa"]).copy()

    if "used_for_ml_training" in df.columns:
        ml_rows = df[df["used_for_ml_training"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if len(ml_rows) > 20:
            df = ml_rows

    feature_cols = get_feature_cols(df)
    if len(feature_cols) == 0:
        raise ValueError("No composition feature columns (wt_percent_*) found.")

    X = df[feature_cols]
    y = df["yield_strength_mpa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42,
    )

    best_name = "RandomForestRegressor"
    best_pipe: Pipeline | None = None
    best_metrics: dict | None = None

    for name, estimator in _candidate_estimators():
        try:
            pipe = _build_pipeline(estimator)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            rmse = float(root_mean_squared_error(y_test, pred))
            r2 = float(r2_score(y_test, pred))
            mae = float(mean_absolute_error(y_test, pred))
            metrics = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 3),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }
            if best_metrics is None or rmse < best_metrics["RMSE"] or (
                rmse == best_metrics["RMSE"] and r2 > best_metrics["R2"]
            ):
                best_name = name
                best_pipe = pipe
                best_metrics = metrics
        except Exception:
            continue

    if best_pipe is None or best_metrics is None:
        pipe = _build_pipeline(RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1))
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        best_pipe = pipe
        best_name = "RandomForestRegressor"
        best_metrics = {
            "MAE": round(float(mean_absolute_error(y_test, pred)), 2),
            "RMSE": round(float(root_mean_squared_error(y_test, pred)), 2),
            "R2": round(float(r2_score(y_test, pred)), 3),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

    est = best_pipe.named_steps["model"]
    interval_method = "tree_spread" if hasattr(est, "estimators_") else "residual_rmse"

    bundle = {
        "model": best_pipe,
        "model_name": best_name,
        "target": "yield_strength_mpa",
        "feature_cols": feature_cols,
        "metrics": best_metrics,
        "train_feature_mean": X_train.mean(numeric_only=True).to_dict(),
        "train_feature_std": X_train.std(numeric_only=True).replace(0, 1).to_dict(),
        "interval_method": interval_method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model(unified: pd.DataFrame) -> dict:
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if "model_name" not in bundle:
                bundle["model_name"] = "RandomForestRegressor"
            if "interval_method" not in bundle:
                est = bundle["model"].named_steps.get("model") or bundle["model"].named_steps.get("rf")
                bundle["interval_method"] = "tree_spread" if hasattr(est, "estimators_") else "residual_rmse"
            return bundle
        except Exception:
            pass
    return train_model(unified)


def predict_with_interval(bundle: dict, row: pd.Series) -> dict:
    """Predict with tree spread or residual RMSE fallback interval."""
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    X = pd.DataFrame([row.reindex(feature_cols)])

    pred = float(model.predict(X)[0])
    interval_method = bundle.get("interval_method", "residual_rmse")
    rmse = float(bundle.get("metrics", {}).get("RMSE", 50))

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

    actual = row.get(bundle.get("target", "yield_strength_mpa"))
    actual_val = float(actual) if pd.notna(actual) else None

    return {
        "prediction": round(pred, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "uncertainty_width": round(upper - lower, 2),
        "actual": round(actual_val, 2) if actual_val is not None else None,
        "interval_method": interval_method,
    }
