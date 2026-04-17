from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .features import TARGET_COLUMN, get_feature_columns


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape(y_true.to_numpy(), y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=42,
                        max_iter=300,
                        learning_rate=0.05,
                        max_depth=10,
                        min_samples_leaf=20,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=18,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(df: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    metrics_dir = output_path / "metrics"
    predictions_dir = output_path / "predictions"
    models_dir = output_path / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = split_time_series(df)
    feature_columns = get_feature_columns(df, target_column=TARGET_COLUMN)

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    best_name = None
    best_model = None
    best_val_rmse = float("inf")
    validation_results: dict[str, dict[str, float]] = {}

    for name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_pred)
        validation_results[name] = val_metrics
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_name = name
            best_model = model

    if best_model is None or best_name is None:
        raise RuntimeError("No model was successfully trained.")

    train_val_df = pd.concat([train_df, val_df], axis=0)
    X_train_val = train_val_df[feature_columns]
    y_train_val = train_val_df[TARGET_COLUMN]
    best_model.fit(X_train_val, y_train_val)

    test_pred = best_model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_pred)

    prediction_frame = pd.DataFrame(
        {
            "datetime": test_df["datetime"],
            "actual_load": y_test,
            "predicted_load": test_pred,
            "absolute_error": np.abs(y_test.to_numpy() - test_pred),
        }
    )
    prediction_frame.to_csv(predictions_dir / "test_predictions.csv", index=False)

    payload = {
        "selected_model": best_name,
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "features": feature_columns,
    }

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    joblib.dump(best_model, models_dir / "load_forecast_pipeline.joblib")

    return payload
