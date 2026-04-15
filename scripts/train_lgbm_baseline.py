"""Train a baseline LightGBM model for energy load forecasting."""

import datetime
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

# Add project root to sys.path to import utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.metrics import mae, rmse  # noqa: E402


def train_lgbm_baseline(
    data_path: str = "data/df_train_latest.parquet",
    features_path: str = "models/model_features_latest.json",
    models_dir: str = "models",
):
    """Load data, train baseline LightGBM, and save the model."""
    print(f"Loading data from {data_path}...")
    df_train = pl.read_parquet(data_path)

    print(f"Loading features from {features_path}...")
    with open(features_path) as f:
        model_features = json.load(f)

    # Prepare data
    # Note: solar_remote_yield_ratio gaps are filled in the notebook.
    X_train = df_train.select(model_features).to_pandas()
    X_train["solar_remote_yield_ratio"] = X_train["solar_remote_yield_ratio"].bfill().ffill()
    y_train = df_train["load"].to_pandas()

    # --- Validation split: last 3 months of training = Jul-Sep 2024 -----------
    # We find the SPLIT_DATE by looking at the max timestamp in the training data
    # and subtracting 90 days. This assumes the training data is up to the split point.
    _max_train = df_train["utc_timestamp"].max()
    if _max_train is None:
        raise ValueError("Training data is empty.")

    _val_start = _max_train - datetime.timedelta(days=90)
    _mask_val = df_train["utc_timestamp"] >= _val_start
    _mask_fit = ~_mask_val

    X_fit = X_train.loc[_mask_fit.to_pandas().values]
    y_fit = y_train.loc[_mask_fit.to_pandas().values]
    X_val = X_train.loc[_mask_val.to_pandas().values]
    y_val = y_train.loc[_mask_val.to_pandas().values]

    print(f"Training on {len(X_fit)} rows, validating on {len(X_val)} rows...")

    # --- Train LightGBM -------------------------------------------------------
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        reg_lambda=0.1,
        objective="regression_l1",
        random_state=42,
        verbose=-1,
    )

    lgb_model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    print(f"Best iteration: {lgb_model.best_iteration_}")

    # Evaluate on validation set
    y_pred_val = lgb_model.predict(X_val)
    val_mae = mae(y_val.to_numpy(), y_pred_val)
    val_rmse = rmse(y_val.to_numpy(), y_pred_val)
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")

    # Save artifact
    os.makedirs(models_dir, exist_ok=True)
    lgb_path = os.path.join(models_dir, "lgb_default_latest.txt")

    print(f"Saving model to {lgb_path}...")
    lgb_model.booster_.save_model(lgb_path)

    print("Training complete.")


if __name__ == "__main__":
    train_lgbm_baseline()
