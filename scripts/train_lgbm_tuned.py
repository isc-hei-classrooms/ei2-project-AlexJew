"""Refit the LightGBM model using the best parameters from Optuna tuning."""

import datetime
import json
import os
import sys
from pathlib import Path
from typing import cast

import lightgbm as lgb
import numpy as np
import polars as pl

# Add project root to sys.path to import utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.metrics import mae, rmse  # noqa: E402


def train_lgbm_tuned(
    data_path: str = "data/df_train_latest.parquet",
    features_path: str = "models/model_features_latest.json",
    params_path: str = "tuning_results/best_params.json",
    models_dir: str = "models",
):
    """Load data, load best params, refit LightGBM, and save the model."""
    print(f"Loading data from {data_path}...")
    df_train = pl.read_parquet(data_path)

    print(f"Loading features from {features_path}...")
    with open(features_path) as f:
        model_features = json.load(f)

    print(f"Loading best parameters from {params_path}...")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Best parameters not found at {params_path}. Run tuning first.")
    with open(params_path) as f:
        best_params = json.load(f)

    # Prepare data
    X_train = df_train.select(model_features).to_pandas()
    X_train["solar_remote_yield_ratio"] = X_train["solar_remote_yield_ratio"].bfill().ffill()
    y_train = df_train["load"].to_pandas()

    # --- Validation split for early stopping: last 3 months of training -------
    _max_train = df_train["utc_timestamp"].max()
    if _max_train is None:
        raise ValueError("Training data is empty.")

    # Cast to ensure it's treated as a datetime for the subtraction
    _max_train_dt = cast(datetime.datetime, _max_train)
    _val_start = _max_train_dt - datetime.timedelta(days=90)
    
    # Create boolean masks as numpy arrays
    _mask_val = (df_train["utc_timestamp"] >= _val_start).to_numpy()
    _mask_fit = (~(df_train["utc_timestamp"] >= _val_start)).to_numpy()

    # Use boolean indexing directly on the dataframes
    X_fit = X_train[_mask_fit]
    y_fit = y_train[_mask_fit]
    X_val = X_train[_mask_val]
    y_val = y_train[_mask_val]

    print(f"Refitting on {len(X_fit)} rows, validating on {len(X_val)} rows...")

    # --- Refit LightGBM -------------------------------------------------------
    tuned_params = {
        **best_params,
        "objective": "regression_l1",
        "bagging_freq": 5,
        "n_estimators": 2000,
        "random_state": 42,
        "verbose": -1,
    }

    lgbm_reg = lgb.LGBMRegressor(**tuned_params)
    lgbm_reg.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # booster_ is only available after fit, Pylance might not know this.
    # We use cast(Any, ...) to avoid attribute access errors.
    from typing import Any
    reg_with_booster = cast(Any, lgbm_reg)
    print(f"Best iteration: {reg_with_booster.best_iteration_}")

    # Evaluate on validation set
    y_pred_val = cast(np.ndarray, lgbm_reg.predict(X_val))
    val_mae = mae(np.asarray(y_val), y_pred_val)
    val_rmse = rmse(np.asarray(y_val), y_pred_val)
    print(f"Validation MAE (tuned): {val_mae:.4f}")
    print(f"Validation RMSE (tuned): {val_rmse:.4f}")

    # Save artifact
    os.makedirs(models_dir, exist_ok=True)
    lgb_path = os.path.join(models_dir, "lgb_tuned_latest.txt")

    print(f"Saving tuned model to {lgb_path}...")
    reg_with_booster.booster_.save_model(lgb_path)

    print("Retraining complete.")


if __name__ == "__main__":
    train_lgbm_tuned()
