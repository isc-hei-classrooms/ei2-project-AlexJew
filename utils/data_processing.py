"""Data processing pipeline for OIKEN and MeteoSwiss datasets.

Run this file directly to process raw data into a clean CSV:
    uv run python utils/data_processing.py

Functions can also be imported individually in notebooks:
    from utils.data_processing import rename_oiken, rename_weather, merge_datasets, clean_data
"""

from datetime import datetime
from glob import glob
from pathlib import Path
import json

import polars as pl
from utils.config import load_config, update_version

def write_train_test_parquets(df_train: pl.DataFrame, df_test: pl.DataFrame) -> str:
    """Save train and test DataFrames to timestamped Parquet files and update config."""
    cfg = load_config()
    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    train_path = data_dir / f"df_train_{timestamp}.parquet"
    test_path = data_dir / f"df_test_{timestamp}.parquet"

    print(f"Saving train data to {train_path}...")
    df_train.write_parquet(train_path)
    
    print(f"Saving test data to {test_path}...")
    df_test.write_parquet(test_path)

    update_version("dataset", "version", timestamp)
    print(f"Updated config.toml with dataset.version = {timestamp}")
    
    return timestamp


def write_model_features(features: list[str]) -> str:
    """Save model features to a timestamped JSON file and update config."""
    cfg = load_config()
    models_dir = Path(cfg.paths.models_dir)
    models_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    features_path = models_dir / f"model_features_{timestamp}.json"

    print(f"Saving model features to {features_path}...")
    with open(features_path, "w") as f:
        json.dump(features, f, indent=2)

    update_version("models", "features_version", timestamp)
    print(f"Updated config.toml with models.features_version = {timestamp}")
    
    return timestamp


def rename_oiken(df: pl.DataFrame) -> pl.DataFrame:
    """Rename OIKEN columns to snake_case."""
    return df.rename(
        {
            "standardised load [-]": "load",
            "standardised forecast load [-]": "forecast_load",
            "central valais solar production [kWh]": "solar_central_valais",
            "sion area solar production [kWh]": "solar_sion",
            "sierre area production [kWh]": "solar_sierre",
            "remote solar production [kWh]": "solar_remote",
        }
    )


def rename_weather(df: pl.DataFrame) -> pl.DataFrame:
    """Rename PRED_* columns to forecast_* and drop pivot artefacts."""
    df = df.rename(
        {
            "PRED_T_2M_ctrl": "forecast_temperature",
            "PRED_GLOB_ctrl": "forecast_global_radiation",
            "PRED_TOT_PREC_ctrl": "forecast_precipitation",
            "PRED_RELHUM_2M_ctrl": "forecast_humidity",
            "PRED_DURSUN_ctrl": "forecast_sunshine_duration",
        }
    )
    # Drop columns with _0 suffix if present (polars pivot artefacts)
    df = df.select([c for c in df.columns if not c.endswith("_0")])
    # Reorder columns
    return df.select(
        "timestamp",
        "forecast_temperature",
        "forecast_global_radiation",
        "forecast_precipitation",
        "forecast_humidity",
        "forecast_sunshine_duration",
    )


def merge_datasets(
    oiken_df: pl.DataFrame, weather_df: pl.DataFrame
) -> pl.DataFrame:
    """Merge OIKEN and weather datasets after converting weather UTC to Swiss local time."""
    weather_local = weather_df.with_columns(
        pl.col("timestamp")
        .dt.convert_time_zone("Europe/Zurich")
        .dt.replace_time_zone(None)
    )
    return oiken_df.join(
        weather_local,
        on="timestamp",
        how="full",
        coalesce=True,
    ).sort("timestamp")


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clip negative forecast values to 0, forward-fill nulls, and drop remaining nulls."""
    df = df.with_columns(
        [
            pl.col("forecast_global_radiation").clip(lower_bound=0),
            pl.col("forecast_precipitation").clip(lower_bound=0),
            pl.col("forecast_humidity").clip(lower_bound=0),
            pl.col("forecast_sunshine_duration").clip(lower_bound=0),
        ]
    )
    df = df.fill_null(strategy="forward")
    df = df.drop_nulls()
    return df


def load_oiken(path: str = "data/oiken_data.csv") -> pl.DataFrame:
    """Load and parse OIKEN CSV with mixed date formats."""
    df = pl.read_csv(
        path,
        null_values=["#N/A"],
        schema_overrides={
            "central valais solar production [kWh]": pl.Float64,
            "sion area solar production [kWh]": pl.Float64,
            "sierre area production [kWh]": pl.Float64,
            "remote solar production [kWh]": pl.Float64,
        },
    )
    return df.with_columns(
        pl.col("timestamp")
        .str.strptime(pl.Datetime, "%d/%m/%y %H:%M", strict=False)
        .fill_null(
            pl.col("timestamp").str.strptime(
                pl.Datetime, "%d/%m/%Y %H:%M", strict=False
            )
        )
        .alias("timestamp")
    )


def load_weather(path: str | None = None) -> pl.DataFrame:
    """Load the most recent weather forecast CSV."""
    if path is None:
        files = sorted(glob("data/sion_forecast_*.csv"))
        if not files:
            msg = "No sion_forecast_*.csv files found in data/"
            raise FileNotFoundError(msg)
        path = files[-1]
    return pl.read_csv(path, try_parse_dates=True)


if __name__ == "__main__":
    oiken_raw = load_oiken()
    weather_raw = load_weather()

    oiken = rename_oiken(oiken_raw)
    weather = rename_weather(weather_raw)
    merged = merge_datasets(oiken, weather)
    processed = clean_data(merged)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output = data_dir / f"processed_data_{timestamp}.csv"
    processed.write_csv(output)
    print(f"Processed data saved to {output} ({len(processed):,} rows)")
