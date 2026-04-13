"""Utility functions for preparing data for model training."""

import datetime
import json
import os
from typing import cast

import polars as pl


def split_temporal(
    df: pl.DataFrame, split_date: datetime.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the dataframe into training and testing sets based on a split date."""
    df_train = df.filter(pl.col("utc_timestamp") < split_date)
    df_test = df.filter(pl.col("utc_timestamp") >= split_date)
    return df_train, df_test


def apply_warmup_clipping(
    df_train_full: pl.DataFrame,
    df_test_full: pl.DataFrame,
    split_date: datetime.datetime,
    warmup_days: int = 9,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Apply warmup period clipping to both train and test sets.

    Avoids data leakage and undefined lag features by dropping the initial
    rows of each set.
    """
    _min_train = df_train_full["utc_timestamp"].min()
    if _min_train is None:
        raise ValueError("The training dataframe is empty or 'utc_timestamp' contains only nulls.")

    _train_clip_start = cast(datetime.datetime, _min_train) + datetime.timedelta(days=warmup_days)
    _test_clip_start = split_date + datetime.timedelta(days=warmup_days)

    df_train = df_train_full.filter(pl.col("utc_timestamp") >= _train_clip_start)
    df_test = df_test_full.filter(pl.col("utc_timestamp") >= _test_clip_start)

    return df_train, df_test


def save_prepared_data(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    model_features: list[str],
    data_dir: str = "data",
    models_dir: str = "models",
    timestamp: str | None = None,
):
    """
    Save the training and testing dataframes as parquet files.

    Saves both a 'latest' version and an optional timestamped version for
    reproducibility. Also saves the feature list as a JSON file.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save Parquet data
    paths = [
        os.path.join(data_dir, "df_train_latest.parquet"),
        os.path.join(data_dir, "df_test_latest.parquet"),
    ]
    if timestamp:
        paths.extend(
            [
                os.path.join(data_dir, f"df_train_{timestamp}.parquet"),
                os.path.join(data_dir, f"df_test_{timestamp}.parquet"),
            ]
        )

    df_train.write_parquet(paths[0])
    df_test.write_parquet(paths[1])
    if timestamp:
        df_train.write_parquet(paths[2])
        df_test.write_parquet(paths[3])

    # Save feature list
    feature_paths = [os.path.join(models_dir, "model_features_latest.json")]
    if timestamp:
        feature_paths.append(os.path.join(models_dir, f"model_features_{timestamp}.json"))

    for p in feature_paths:
        with open(p, "w") as f:
            json.dump(model_features, f, indent=2)
