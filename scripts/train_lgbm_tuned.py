"""Refit the LightGBM model using the best parameters from Optuna tuning."""

import datetime
import json
import sys
from pathlib import Path
from typing import cast, Any

import lightgbm as lgb
import numpy as np
import polars as pl

# Add project root to sys.path to import utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config import load_config  # noqa: E402
from utils.metrics import mae, rmse  # noqa: E402


def train_lgbm_tuned():
    """Load data, load best params, refit LightGBM, and save the model."""
    cfg = load_config()

    data_path = cfg.train_parquet_path()
    features_path = cfg.features_json_path()
    # Note: tuning results still write to best_params.json for now (Step 8)
    params_path = Path(cfg.paths.tuning_dir) / "best_params.json"

    print(f"Loading data from {data_path}...")
    df_train = pl.read_parquet(data_path)

    print(f"Loading features from {features_path}...")
    with open(features_path) as f:
        model_features = json.load(f)

    print(f"Loading best parameters from {params_path}...")
    if not params_path.exists():
        raise FileNotFoundError(f"Best parameters not found at {params_path}. Run tuning first.")
    with open(params_path) as f:
        best_params = json.load(f)

    # Prepare data
    X_train = df_train.select(model_features).to_pandas()
    for col in cfg.dataset.fill_columns:
        if col in X_train.columns:
            X_train[col] = X_train[col].bfill().ffill()

    y_train = df_train[cfg.dataset.target].to_pandas()

    # --- Validation split for early stopping: from config -------
    _max_train = df_train["utc_timestamp"].max()
    if _max_train is None:
        raise ValueError("Training data is empty.")

    _val_start = cast(datetime.datetime, _max_train) - datetime.timedelta(days=cfg.training.validation_days)
    
    # Create boolean masks as numpy arrays
    _mask_val = (df_train["utc_timestamp"] >= _val_start).to_numpy()
    _mask_fit = (~_mask_val)

    # Use boolean indexing directly on the dataframes
    X_fit = X_train[_mask_fit]
    y_fit = y_train[_mask_fit]
    X_val = X_train[_mask_val]
    y_val = y_train[_mask_val]

    print(f"Refitting on {len(X_fit)} rows, validating on {len(X_val)} rows...")

    # --- Refit LightGBM with parameters from config and Optuna ----------------
    tc = cfg.training.lgbm_tuned
    tuned_params = {
        **best_params,
        "objective": tc.objective,
        "bagging_freq": tc.bagging_freq,
        "n_estimators": tc.n_estimators,
        "random_state": cfg.training.random_state,
        "verbose": -1,
    }

    lgbm_reg = lgb.LGBMRegressor(**tuned_params)
    lgbm_reg.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(cfg.training.early_stopping_rounds, verbose=False)],
    )

    reg_with_booster = cast(Any, lgbm_reg)
    print(f"Best iteration: {reg_with_booster.best_iteration_}")

    # Evaluate on validation set
    y_pred_val = cast(np.ndarray, lgbm_reg.predict(X_val))
    val_mae = mae(np.asarray(y_val), y_pred_val)
    val_rmse = rmse(np.asarray(y_val), y_pred_val)
    print(f"Validation MAE (tuned): {val_mae:.4f}")
    print(f"Validation RMSE (tuned): {val_rmse:.4f}")

    # Save artifact using path helpers
    lgb_path = cfg.lgbm_tuned_path()
    lgb_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving tuned model to {lgb_path}...")
    reg_with_booster.booster_.save_model(str(lgb_path))

    print("Retraining complete.")


if __name__ == "__main__":
    train_lgbm_tuned()
