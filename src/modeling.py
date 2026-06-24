"""Yield strength prediction model using RandomForest on composition features."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("models/yield_strength_model.pkl")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("wt_percent_")]


def train_model(unified: pd.DataFrame) -> dict:
    """Train only on rows that have yield_strength_mpa and wt_percent_ composition columns."""
    df = unified.dropna(subset=["yield_strength_mpa"]).copy()

    # Only train on experimental matminer rows (not uploaded supplier sheets)
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
        X, y, test_size=0.22, random_state=42
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=250, random_state=42,
                min_samples_leaf=2, n_jobs=-1,
            )),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE": round(float(mean_absolute_error(y_test, pred)), 2),
        "RMSE": round(float(root_mean_squared_error(y_test, pred)), 2),
        "R2": round(float(r2_score(y_test, pred)), 3),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "train_feature_mean": X_train.mean(numeric_only=True).to_dict(),
        "train_feature_std": X_train.std(numeric_only=True).replace(0, 1).to_dict(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model(unified: pd.DataFrame) -> dict:
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return train_model(unified)


def predict_with_interval(bundle: dict, row: pd.Series) -> dict:
    """Predict yield strength with tree-based prediction interval."""
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    X = pd.DataFrame([row.reindex(feature_cols)])

    pred = float(model.predict(X)[0])

    imputed = model.named_steps["imputer"].transform(X)
    scaled = model.named_steps["scaler"].transform(imputed)
    rf = model.named_steps["rf"]

    tree_preds = np.array([tree.predict(scaled)[0] for tree in rf.estimators_])
    lower = float(np.percentile(tree_preds, 10))
    upper = float(np.percentile(tree_preds, 90))

    actual = row.get("yield_strength_mpa")
    actual_val = float(actual) if pd.notna(actual) else None

    return {
        "prediction": round(pred, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "uncertainty_width": round(upper - lower, 2),
        "actual": round(actual_val, 2) if actual_val is not None else None,
    }
