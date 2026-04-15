"""Utilities for preparing data for model training and tuning."""

import json

import pandas as pd
import polars as pl


def load_data_and_features(
    data_path: str, features_path: str
) -> tuple[pl.DataFrame, list[str]]:
    """Load the training data (parquet) and the list of features (JSON)."""
    df = pl.read_parquet(data_path)
    with open(features_path) as f:
        features = json.load(f)
    return df, features


def prepare_X_y(
    df: pl.DataFrame, features: list[str], target: str = "load"
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X and y by selecting features, converting to pandas, and filling gaps."""
    X = df.select(features).to_pandas()
    # Filling gaps for specific features if they exist
    if "solar_remote_yield_ratio" in X.columns:
        X["solar_remote_yield_ratio"] = (
            X["solar_remote_yield_ratio"].bfill().ffill()
        )
    y = df[target].to_pandas()
    return X, y
