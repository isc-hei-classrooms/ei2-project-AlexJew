"""Train a Ridge regression model for energy load forecasting."""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path to import utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.metrics import mae, rmse  # noqa: E402
from utils.model_preparation import load_data_and_features, prepare_X_y  # noqa: E402


def train_ridge(
    data_path: str = "data/df_train_latest.parquet",
    features_path: str = "models/model_features_latest.json",
    models_dir: str = "models",
):
    """Load data, train RidgeCV, and save the model and scaler."""
    print("Loading data and features...")
    df_train, model_features = load_data_and_features(data_path, features_path)

    # Prepare data
    X_train, y_train = prepare_X_y(df_train, model_features)

    print(f"Training on {len(X_train)} rows with {len(model_features)} features...")

    # Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Ridge with built-in CV over a log-spaced alpha grid
    alphas = np.logspace(-2, 3, 20)
    ridge_model = RidgeCV(alphas=alphas, cv=5)
    ridge_model.fit(X_train_scaled, y_train)

    print(f"Best alpha: {ridge_model.alpha_:.4f}")

    # Evaluate on training set (for sanity check)
    y_pred_train = ridge_model.predict(X_train_scaled)
    train_mae = mae(y_train.to_numpy(), y_pred_train)
    train_rmse = rmse(y_train.to_numpy(), y_pred_train)
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler_latest.joblib")
    ridge_path = os.path.join(models_dir, "ridge_latest.joblib")

    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)

    print(f"Saving model to {ridge_path}...")
    joblib.dump(ridge_model, ridge_path)

    print("Training complete.")


if __name__ == "__main__":
    train_ridge()
