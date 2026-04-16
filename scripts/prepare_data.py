"""Prepare training and test data for energy load forecasting models.

Run this script before any training script to generate the prepared parquet
files that train_ridge.py, train_lgbm_baseline.py, and train_lgbm_tuned.py
all depend on.

Usage:
    uv run python scripts/prepare_data.py
"""

import datetime
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import cast

import polars as pl

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.feature_engineering import (  # noqa: E402
    add_cyclical_features,
    add_dst_feature,
    add_holiday_features,
    add_lag_features,
    add_remote_yield_ratio,
    add_temporal_features,
    add_working_day_flag,
    compute_poa_irradiance,
    estimate_solar_capacity,
    save_featured_data,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPLIT_DATE = datetime.datetime(2024, 10, 1)
WARMUP_DAYS = 9

STATIONS = [
    "sion",
    "evionnaz",
    "evolene",
    "montana",
    "visp",
    "basel",
    "bern",
    "geneve",
    "pully",
    "zurich",
]

FORECAST_RAW_TO_CLEAN = {
    "PRED_T_2M_ctrl": "forecast_temperature",
    "PRED_GLOB_ctrl": "forecast_global_radiation",
    "PRED_TOT_PREC_ctrl": "forecast_precipitation",
    "PRED_RELHUM_2M_ctrl": "forecast_humidity",
    "PRED_DURSUN_ctrl": "forecast_sunshine_duration",
}

MEASUREMENT_RAW_TO_CLEAN = {
    "Air temperature 2m above ground (current value)": "measured_temperature",
    "Global radiation (ten minutes mean)": "measured_global_radiation",
    "Precipitation (ten minutes total)": "measured_precipitation",
    "Relative air humidity 2m above ground (current value)": "measured_humidity",
    "Sunshine duration (ten minutes total)": "measured_sunshine_duration",
}

# Solar panel defaults (match notebook slider defaults)
PANEL_TILT = 30
PANEL_AZIMUTH = 180

# ---------------------------------------------------------------------------
# Data splitting and saving (absorbed from utils/model_preparation.py)
# ---------------------------------------------------------------------------

def split_temporal(
    df: pl.DataFrame, split_date: datetime.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the dataframe into training and test sets at split_date."""
    df_train = df.filter(pl.col("utc_timestamp") < split_date)
    df_test = df.filter(pl.col("utc_timestamp") >= split_date)
    return df_train, df_test


def apply_warmup_clipping(
    df_train_full: pl.DataFrame,
    df_test_full: pl.DataFrame,
    split_date: datetime.datetime,
    warmup_days: int = 9,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop the first warmup_days rows of both splits to avoid undefined lags."""
    _min_train = df_train_full["utc_timestamp"].min()
    if _min_train is None:
        raise ValueError("Training dataframe is empty.")
    _train_clip_start = cast(datetime.datetime, _min_train) + datetime.timedelta(days=warmup_days)
    _test_clip_start = split_date + datetime.timedelta(days=warmup_days)
    df_train = df_train_full.filter(pl.col("utc_timestamp") >= _train_clip_start)
    df_test = df_test_full.filter(pl.col("utc_timestamp") >= _test_clip_start)
    return df_train, df_test


def exclude_incorrect_test_timestamps(
    df_test: pl.DataFrame,
    start_local: datetime.datetime = datetime.datetime(2025, 9, 13, 0, 15),
    end_local: datetime.datetime = datetime.datetime(2025, 9, 17, 0, 0),
) -> pl.DataFrame:
    """Exclude known-bad local timestamp interval from the test split."""
    return df_test.filter(
        (pl.col("local_timestamp") < start_local) | (pl.col("local_timestamp") > end_local)
    )


def fill_test_feature_gaps(df_test: pl.DataFrame) -> pl.DataFrame:
    """Fill known sparse test feature gaps for downstream model inference."""
    if "solar_remote_yield_ratio" not in df_test.columns:
        return df_test
    return df_test.with_columns(
        pl.col("solar_remote_yield_ratio").backward_fill().forward_fill()
    )


def save_prepared_data(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    model_features: list[str],
    data_dir: str = "data",
    models_dir: str = "models",
    timestamp: str | None = None,
) -> None:
    """Save train/test parquets and the feature list JSON."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df_train.write_parquet(os.path.join(data_dir, "df_train_latest.parquet"))
    df_test.write_parquet(os.path.join(data_dir, "df_test_latest.parquet"))
    if timestamp:
        df_train.write_parquet(os.path.join(data_dir, f"df_train_{timestamp}.parquet"))
        df_test.write_parquet(os.path.join(data_dir, f"df_test_{timestamp}.parquet"))

    feature_path = os.path.join(models_dir, "model_features_latest.json")
    with open(feature_path, "w") as f:
        json.dump(model_features, f, indent=2)
    if timestamp:
        with open(os.path.join(models_dir, f"model_features_{timestamp}.json"), "w") as f:
            json.dump(model_features, f, indent=2)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_oiken(path: str = "data/oiken_data.csv") -> pl.DataFrame:
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
            pl.col("timestamp").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False)
        )
        .alias("timestamp")
    )


def _latest_file(pattern: str) -> str:
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files[-1]


def _load_forecasts(data_dir: str = "data") -> pl.DataFrame:
    weather_df = None
    for station in STATIONS:
        path = _latest_file(os.path.join(data_dir, f"{station}_forecast_*.csv"))
        df = pl.read_csv(path, try_parse_dates=True)
        df = df.rename({k: f"{station}_{v}" for k, v in FORECAST_RAW_TO_CLEAN.items()})
        # Drop pivot artefact columns
        df = df.select([c for c in df.columns if not c.endswith("_0")])
        if weather_df is None:
            weather_df = df
        else:
            weather_df = weather_df.join(df, on="timestamp", how="full", coalesce=True)
    assert weather_df is not None
    return weather_df.sort("timestamp")


def _load_measurements(data_dir: str = "data") -> pl.DataFrame:
    measurement_df = None
    for station in STATIONS:
        path = _latest_file(os.path.join(data_dir, f"{station}_measurement_*.csv"))
        df = (
            pl.read_csv(path, try_parse_dates=True)
            .filter(pl.col("timestamp").dt.minute().is_in([0, 30]))
        )
        df = df.rename({k: f"{station}_{v}" for k, v in MEASUREMENT_RAW_TO_CLEAN.items()})
        if measurement_df is None:
            measurement_df = df
        else:
            measurement_df = measurement_df.join(df, on="timestamp", how="full", coalesce=True)
    assert measurement_df is not None
    return measurement_df.sort("timestamp")


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def _rename_oiken(df: pl.DataFrame) -> pl.DataFrame:
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


def _merge_and_clean(
    oiken_renamed: pl.DataFrame,
    weather_df: pl.DataFrame,
    measurement_df: pl.DataFrame,
) -> pl.DataFrame:
    # Convert OIKEN timestamps from Swiss local time to naive UTC
    oiken_utc = oiken_renamed.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone("Europe/Zurich", ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
    ).with_columns(pl.col("timestamp").forward_fill())

    # Strip timezone from weather (already UTC)
    weather_utc = weather_df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    measurement_utc = measurement_df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    # Merge all three datasets
    merged = (
        oiken_utc
        .join(weather_utc, on="timestamp", how="full", coalesce=True)
        .join(measurement_utc, on="timestamp", how="full", coalesce=True)
        .sort("timestamp")
        .rename({"timestamp": "utc_timestamp"})
        .with_columns(
            pl.col("utc_timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Europe/Zurich")
            .dt.replace_time_zone(None)
            .alias("local_timestamp")
        )
        .select("utc_timestamp", "local_timestamp", pl.exclude("utc_timestamp", "local_timestamp"))
    )

    # Clip negative forecast values
    clip_suffixes = [
        "forecast_global_radiation",
        "forecast_precipitation",
        "forecast_humidity",
        "forecast_sunshine_duration",
    ]
    merged = merged.with_columns(
        [
            pl.col(c).clip(lower_bound=0)
            for c in merged.columns
            if any(c.endswith(s) for s in clip_suffixes)
        ]
    )

    # Forward-fill then drop remaining nulls at the start
    return merged.fill_null(strategy="forward").drop_nulls()


def _build_model_features(df: pl.DataFrame) -> list[str]:
    """Return feature columns (excludes target, timestamps, baselines, raw measured/solar)."""
    weather_vars = [
        "temperature", "global_radiation", "precipitation", "humidity", "sunshine_duration"
    ]
    raw_measured = {f"{s}_measured_{v}" for s in STATIONS for v in weather_vars}
    raw_solar_prod = {"solar_central_valais", "solar_sion", "solar_sierre", "solar_remote"}
    exclude = {
        "utc_timestamp", "local_timestamp", "load", "forecast_load",
        *raw_measured, *raw_solar_prod,
    }
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full data preparation pipeline and save train/test parquet files."""
    print("=== prepare_data.py ===")

    # 1. Load raw data
    print("Loading OIKEN data...")
    oiken_raw = _load_oiken()

    print("Loading weather forecasts...")
    weather_df = _load_forecasts()

    print("Loading weather measurements...")
    measurement_df = _load_measurements()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # 2. Rename, merge, and clean
    print("Merging and cleaning datasets...")
    oiken_renamed = _rename_oiken(oiken_raw)
    df_clean = _merge_and_clean(oiken_renamed, weather_df, measurement_df)
    print(f"  Clean dataset: {df_clean.height:,} rows, {df_clean.width} columns")

    print("  Saving df_clean_latest.parquet...")
    df_clean.write_parquet(os.path.join("data", "df_clean_latest.parquet"))

    # 3. Feature engineering
    print("Running feature engineering pipeline...")
    df = add_temporal_features(df_clean)
    df = add_holiday_features(df)
    df = add_working_day_flag(df)
    df = add_dst_feature(df)
    df = add_cyclical_features(df)
    df = compute_poa_irradiance(df, tilt=PANEL_TILT, azimuth=PANEL_AZIMUTH)
    df = estimate_solar_capacity(df, threshold=200.0, window_days=30, min_periods=96)
    df = add_remote_yield_ratio(df, window_days=30)
    df = add_lag_features(df)
    print(f"  After feature engineering: {df.width} columns")

    print("  Saving feature_data_latest.parquet...")
    save_featured_data(df, timestamp=timestamp)

    # 4. Build feature list
    model_features = _build_model_features(df)
    print(f"  Model features: {len(model_features)}")

    # 5. Split and clip warmup
    print(f"Splitting at {SPLIT_DATE.date()} with {WARMUP_DAYS}-day warmup clipping...")
    df_train_full, df_test_full = split_temporal(df, SPLIT_DATE)
    df_train, df_test = apply_warmup_clipping(df_train_full, df_test_full, SPLIT_DATE, WARMUP_DAYS)

    # Add persistence baseline column (load from t-7d) using the full pre-split series
    _rows_per_day = 96  # 15-min intervals
    df_test = df_test.join(
        df.select(
            "utc_timestamp",
            pl.col("load").shift(_rows_per_day * 7).alias("load_persistence_7d"),
        ),
        on="utc_timestamp",
        how="left",
    )

    df_test = fill_test_feature_gaps(df_test)

    _test_before_drop = df_test.height
    df_test = exclude_incorrect_test_timestamps(df_test)
    _dropped = _test_before_drop - df_test.height
    print(f"  Dropped {_dropped} rows with known incorrect timestamps in test set")

    _tr_min = df_train["utc_timestamp"].min().date()
    _tr_max = df_train["utc_timestamp"].max().date()
    _te_min = df_test["utc_timestamp"].min().date()
    _te_max = df_test["utc_timestamp"].max().date()
    print(f"  Train: {df_train.height:,} rows  ({_tr_min} → {_tr_max})")
    print(f"  Test:  {df_test.height:,} rows  ({_te_min} → {_te_max})")

    # 6. Save artifacts
    print("Saving prepared data...")
    save_prepared_data(df_train, df_test, model_features, timestamp=timestamp)
    print("  Saved: data/df_clean_latest.parquet")
    print("  Saved: data/feature_data_latest.parquet")
    print("  Saved: data/df_train_latest.parquet")
    print("  Saved: data/df_test_latest.parquet")
    print("  Saved: models/model_features_latest.json")
    print("Done.")


if __name__ == "__main__":
    main()
