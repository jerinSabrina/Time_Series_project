# Electricity Load Forecasting Project

This project builds an end-to-end electricity load forecasting pipeline from the provided hourly datasets.

It includes:
- exploratory data analysis (EDA)
- feature engineering for time series forecasting
- preprocessing and model training
- evaluation on a chronological holdout set
- saved plots, metrics, predictions, and the trained model

## Project Structure

- `scripts/run_pipeline.py`: main entry point
- `src/load_forecasting/data.py`: dataset loading and merging
- `src/load_forecasting/features.py`: feature engineering
- `src/load_forecasting/eda.py`: EDA reports and charts
- `src/load_forecasting/modeling.py`: training and evaluation
- `outputs/`: generated artifacts after a run

## Default Data Sources

The pipeline is configured to use:

- `/Users/sabrina/Downloads/archive (2)/continuous dataset.csv`
- `/Users/sabrina/Downloads/archive (2)/weekly pre-dispatch forecast.csv`

You can override them with command-line arguments.

## Run

```bash
./.venv/bin/python scripts/run_pipeline.py
```

## Optional Arguments

```bash
./.venv/bin/python scripts/run_pipeline.py \
  --continuous-csv "/path/to/continuous dataset.csv" \
  --forecast-csv "/path/to/weekly pre-dispatch forecast.csv" \
  --output-dir outputs
```

## Outputs

After running, the pipeline will create:

- `outputs/eda/summary.json`
- `outputs/eda/correlations.csv`
- `outputs/eda/*.png`
- `outputs/models/load_forecast_pipeline.joblib`
- `outputs/metrics/metrics.json`
- `outputs/predictions/test_predictions.csv`

## Notes

- The model uses only information available from the historical timeline when creating lag and rolling demand features.
- The merged `load_forecast` feature is treated as an available exogenous input.
- The weather variables are included as predictive features.
