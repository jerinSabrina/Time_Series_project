from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def run_eda(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["hour"] = plot_df["datetime"].dt.hour
    plot_df["day_name"] = plot_df["datetime"].dt.day_name()

    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "date_min": str(df["datetime"].min()),
        "date_max": str(df["datetime"].max()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "numeric_summary": _json_safe(df.describe().round(4).to_dict()),
    }

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    corr = df.select_dtypes(include="number").corr(numeric_only=True)
    corr["target_correlation"] = corr["target_load"]
    corr[["target_correlation"]].sort_values("target_correlation", ascending=False).to_csv(
        output_path / "correlations.csv"
    )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(14, 5))
    plt.plot(df["datetime"], df["target_load"], linewidth=0.8, color="#1f77b4")
    plt.title("Electricity Load Over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Load")
    plt.tight_layout()
    plt.savefig(output_path / "load_over_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x="hour", y="target_load", data=plot_df, color="#6baed6")
    plt.title("Load Distribution by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Load")
    plt.tight_layout()
    plt.savefig(output_path / "load_by_hour.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x="day_name", y="target_load", data=plot_df)
    plt.title("Load Distribution by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Load")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path / "load_by_day_of_week.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x="T2M_toc", y="target_load", data=df.sample(min(len(df), 5000), random_state=42), s=18, alpha=0.5)
    plt.title("Load vs Temperature at TOC")
    plt.xlabel("T2M_toc")
    plt.ylabel("Load")
    plt.tight_layout()
    plt.savefig(output_path / "load_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 10))
    heatmap_cols = [
        "target_load",
        "load_forecast",
        "T2M_toc",
        "T2M_san",
        "T2M_dav",
        "W2M_toc",
        "W2M_san",
        "W2M_dav",
        "holiday",
        "school",
    ]
    available_cols = [col for col in heatmap_cols if col in df.columns]
    sns.heatmap(df[available_cols].corr(numeric_only=True), cmap="Blues", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path / "correlation_heatmap.png", dpi=160)
    plt.close()
