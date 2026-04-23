from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_forecasting.data import load_and_merge_data
from load_forecasting.eda import run_eda
from load_forecasting.features import engineer_features
from load_forecasting.modeling import train_and_evaluate


DEFAULT_CONTINUOUS_CSV = "/Users/sabrina/Downloads/archive (2)/continuous dataset.csv"
DEFAULT_FORECAST_CSV = "/Users/sabrina/Downloads/archive (2)/weekly pre-dispatch forecast.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run electricity load forecasting pipeline.")
    parser.add_argument("--continuous-csv", default=DEFAULT_CONTINUOUS_CSV, help="Path to the continuous dataset CSV.")
    parser.add_argument("--forecast-csv", default=DEFAULT_FORECAST_CSV, help="Path to the weekly forecast CSV.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"), help="Directory for generated artifacts.")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_and_merge_data(args.continuous_csv, args.forecast_csv)
    run_eda(raw_df, output_dir / "eda")

    model_df = engineer_features(raw_df)
    results = train_and_evaluate(model_df, output_dir)

    print("Pipeline completed successfully.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
