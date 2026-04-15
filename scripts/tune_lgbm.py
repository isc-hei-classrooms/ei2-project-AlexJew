"""Hyperparameter tuning for LightGBM using expanding-window cross-validation."""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import cast

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to sys.path to import utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.metrics import mae  # noqa: E402
from utils.model_preparation import load_data_and_features, prepare_X_y  # noqa: E402


def tune_lgbm(
    data_path: str = "data/df_train_latest.parquet",
    features_path: str = "models/model_features_latest.json",
    n_trials: int = 100,
    output_dir: str = "tuning_results",
):
    """Run Optuna tuning for LightGBM locally."""
    print("Loading data and features...")
    df_train_full, model_features = load_data_and_features(data_path, features_path)

    # Prepare data (handle yield ratio gaps)
    df_train_full = df_train_full.with_columns(
        pl.col("solar_remote_yield_ratio").backward_fill().forward_fill()
    )

    # Define 4 folds covering a full year (Oct 2023 to Sep 2024)
    folds = [
        {"split": datetime.datetime(2023, 10, 1), "name": "Q4 2023"},
        {"split": datetime.datetime(2024, 1, 1), "name": "Q1 2024"},
        {"split": datetime.datetime(2024, 4, 1), "name": "Q2 2024"},
        {"split": datetime.datetime(2024, 7, 1), "name": "Q3 2024"},
    ]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression_l1",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": 2000,
            "random_state": 42,
            "verbose": -1,
        }

        fold_maes = []

        for fold in folds:
            split_date = fold["split"]
            val_end = split_date + datetime.timedelta(days=90)

            df_fit = df_train_full.filter(pl.col("utc_timestamp") < split_date)
            df_val = df_train_full.filter(
                (pl.col("utc_timestamp") >= split_date) & (pl.col("utc_timestamp") < val_end)
            )

            X_fit, y_fit = prepare_X_y(df_fit, model_features)
            X_val, y_val = prepare_X_y(df_val, model_features)

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            y_pred = cast(np.ndarray, model.predict(X_val))
            fold_maes.append(mae(np.asarray(y_val), y_pred))

        return float(np.mean(fold_maes))

    os.makedirs(output_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.abspath(output_dir)}/lgbm_study.db"

    study = optuna.create_study(
        study_name="lgbm_tuning",
        direction="minimize",
        storage=storage_path,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=10),
    )

    print(f"Starting local tuning with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save outputs
    best_params_path = os.path.join(output_dir, "best_params.json")
    trials_path = os.path.join(output_dir, "trials.csv")

    print(f"Saving best parameters to {best_params_path}...")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"Saving trials history to {trials_path}...")
    study.trials_dataframe().to_csv(trials_path, index=False)

    print(f"Tuning complete. Best MAE: {study.best_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning LightGBM locally.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/df_train_latest.parquet",
        help="Path to training parquet",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="models/model_features_latest.json",
        help="Path to features JSON",
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument(
        "--output", type=str, default="tuning_results", help="Output directory"
    )

    args = parser.parse_args()

    tune_lgbm(
        data_path=args.data,
        features_path=args.features,
        n_trials=args.trials,
        output_dir=args.output,
    )
