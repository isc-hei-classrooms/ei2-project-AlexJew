"""Train a baseline LightGBM model for energy load forecasting."""

import datetime
import json
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

# Add project root to sys.path to import utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config import load_config  # noqa: E402
from utils.metrics import mae, rmse  # noqa: E402


def train_lgbm_baseline():
    """Load data, train baseline LightGBM, and save the model."""
    cfg = load_config()

    data_path = cfg.train_parquet_path()
    features_path = cfg.features_json_path()

    print(f"Loading data from {data_path}...")
    df_train = pl.read_parquet(data_path)

    print(f"Loading features from {features_path}...")
    with open(features_path) as f:
        model_features = json.load(f)

    # Prepare data
    X_train = df_train.select(model_features).to_pandas()
    for col in cfg.dataset.fill_columns:
        if col in X_train.columns:
            X_train[col] = X_train[col].bfill().ffill()

    y_train = df_train[cfg.dataset.target].to_pandas()

    # --- Validation split: from config -----------
    _max_train = df_train["utc_timestamp"].max()
    if _max_train is None:
        raise ValueError("Training data is empty.")

    _val_start = _max_train - datetime.timedelta(days=cfg.training.validation_days)
    _mask_val = df_train["utc_timestamp"] >= _val_start
    _mask_fit = ~_mask_val

    X_fit = X_train.loc[_mask_fit.to_pandas().values]
    y_fit = y_train.loc[_mask_fit.to_pandas().values]
    X_val = X_train.loc[_mask_val.to_pandas().values]
    y_val = y_train.loc[_mask_val.to_pandas().values]

    print(f"Training on {len(X_fit)} rows, validating on {len(X_val)} rows...")

    # --- Train LightGBM using config parameters -------------------------------
    params = cfg.training.lgbm_baseline
    lgb_model = lgb.LGBMRegressor(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples,
        reg_lambda=params.reg_lambda,
        objective=params.objective,
        random_state=cfg.training.random_state,
        verbose=-1,
    )

    lgb_model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(cfg.training.early_stopping_rounds, verbose=False)],
    )

    print(f"Best iteration: {lgb_model.best_iteration_}")

    # Evaluate on validation set
    y_pred_val = lgb_model.predict(X_val)
    val_mae = mae(y_val.to_numpy(), y_pred_val)
    val_rmse = rmse(y_val.to_numpy(), y_pred_val)
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")

    # Save artifact using path helpers
    lgb_path = cfg.lgbm_baseline_path()
    lgb_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {lgb_path}...")
    lgb_model.booster_.save_model(str(lgb_path))

    print("Training complete.")


if __name__ == "__main__":
    train_lgbm_baseline()
