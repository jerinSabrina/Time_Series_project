from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_and_merge_data(continuous_csv: str | Path, forecast_csv: str | Path) -> pd.DataFrame:
    """Load the raw demand/weather data and merge the pre-dispatch forecast."""
    continuous_path = Path(continuous_csv)
    forecast_path = Path(forecast_csv)

    demand_df = pd.read_csv(continuous_path, parse_dates=["datetime"])
    forecast_df = pd.read_csv(forecast_path, parse_dates=["datetime"])

    demand_df = demand_df.sort_values("datetime").reset_index(drop=True)
    forecast_df = forecast_df.sort_values("datetime").reset_index(drop=True)

    merged = demand_df.merge(forecast_df, on="datetime", how="left")
    merged = merged.rename(columns={"nat_demand": "target_load"})

    return merged
