from __future__ import annotations
import numpy as np
import pandas as pd


TARGET_COLUMN = "target_load"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["datetime"]

    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["day_of_year"] = dt.dt.dayofyear
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_month_start"] = dt.dt.is_month_start.astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


def add_lag_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    out = df.copy()

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        out[f"load_lag_{lag}"] = out[target_column].shift(lag)

    return out


def add_rolling_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    out = df.copy()
    shifted = out[target_column].shift(1)

    out["rolling_mean_24"] = shifted.rolling(window=24).mean()
    out["rolling_std_24"] = shifted.rolling(window=24).std()
    out["rolling_min_24"] = shifted.rolling(window=24).min()
    out["rolling_max_24"] = shifted.rolling(window=24).max()
    out["rolling_mean_168"] = shifted.rolling(window=168).mean()
    out["rolling_std_168"] = shifted.rolling(window=168).std()

    return out


def engineer_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_lag_features(out, target_column=target_column)
    out = add_rolling_features(out, target_column=target_column)
    out = out.dropna().reset_index(drop=True)
    return out


def get_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    excluded = {"datetime", target_column}
    return [col for col in df.columns if col not in excluded]
