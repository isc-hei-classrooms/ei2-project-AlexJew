"""Train a Ridge regression model for energy load forecasting."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path to import utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config import load_config  # noqa: E402
from utils.metrics import mae, rmse  # noqa: E402

def train_ridge():
    """Load data, train RidgeCV, and save the model and scaler."""
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
    # Filling missing values as in the baseline
    for col in cfg.dataset.fill_columns:
        if col in X_train.columns:
            X_train[col] = X_train[col].bfill().ffill()

    y_train = df_train[cfg.dataset.target].to_pandas()

    print(f"Training on {len(X_train)} rows with {len(model_features)} features...")

    # Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Ridge with built-in CV over a log-spaced alpha grid from config
    rc = cfg.training.ridge
    alphas = np.logspace(rc.alpha_min_log, rc.alpha_max_log, rc.alpha_num)
    ridge_model = RidgeCV(alphas=alphas, cv=rc.cv_folds)
    ridge_model.fit(X_train_scaled, y_train)

    print(f"Best alpha: {ridge_model.alpha_:.4f}")

    # Evaluate on training set (for sanity check)
    y_pred_train = ridge_model.predict(X_train_scaled)
    train_mae = mae(y_train.to_numpy(), y_pred_train)
    train_rmse = rmse(y_train.to_numpy(), y_pred_train)
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")

    # Save artifacts using path helpers
    scaler_path = cfg.scaler_path()
    ridge_path = cfg.ridge_path()

    # Ensure models directory exists
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)

    print(f"Saving model to {ridge_path}...")
    joblib.dump(ridge_model, ridge_path)

    print("Training complete.")


if __name__ == "__main__":
    train_ridge()
