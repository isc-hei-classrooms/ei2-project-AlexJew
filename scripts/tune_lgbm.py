"""Hyperparameter tuning for LightGBM using expanding-window cross-validation."""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import cast, Any

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

from utils.config import load_config  # noqa: E402
from utils.metrics import mae  # noqa: E402


def tune_lgbm(n_trials_override: int = None):
    """Run Optuna tuning for LightGBM locally."""
    cfg = load_config()
    
    data_path = cfg.train_parquet_path()
    features_path = cfg.features_json_path()
    output_dir = Path(cfg.paths.tuning_dir)
    
    n_trials = n_trials_override if n_trials_override is not None else cfg.tuning.n_trials

    print(f"Loading data from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Run notebook first.")
    df_train_full = pl.read_parquet(data_path)

    print(f"Loading features from {features_path}...")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found at {features_path}. Run notebook first.")
    with open(features_path) as f:
        model_features = json.load(f)

    # Prepare data (handle yield ratio gaps as in notebook)
    for col in cfg.dataset.fill_columns:
        if col in df_train_full.columns:
            df_train_full = df_train_full.with_columns(
                pl.col(col).backward_fill().forward_fill()
            )

    # Use folds from config
    folds = cfg.tuning.folds

    def objective(trial: optuna.Trial) -> float:
        # Build params using search space from config
        ss = cfg.tuning.search_space
        params = {
            "objective": cfg.tuning.objective,
            "bagging_freq": cfg.tuning.bagging_freq,
            "n_estimators": cfg.tuning.n_estimators,
            "random_state": cfg.training.random_state,
            "verbose": -1,
        }
        
        # Add dynamic parameters from search space
        for param_name, space in ss.items():
            if "log" in space and space["log"]:
                params[param_name] = trial.suggest_float(param_name, space["low"], space["high"], log=True)
            elif isinstance(space["low"], float):
                params[param_name] = trial.suggest_float(param_name, space["low"], space["high"])
            else:
                params[param_name] = trial.suggest_int(param_name, space["low"], space["high"])

        fold_maes = []

        for fold in folds:
            split_date = fold["split"]
            if isinstance(split_date, str):
                split_date = datetime.datetime.fromisoformat(split_date.replace("Z", "+00:00"))
            
            val_end = split_date + datetime.timedelta(days=cfg.training.validation_days)

            # Ensure utc_timestamp is comparable
            df_fit = df_train_full.filter(pl.col("utc_timestamp") < split_date)
            df_val = df_train_full.filter(
                (pl.col("utc_timestamp") >= split_date) & (pl.col("utc_timestamp") < val_end)
            )

            X_fit = df_fit.select(model_features).to_pandas()
            y_fit = df_fit[cfg.dataset.target].to_pandas()
            X_val = df_val.select(model_features).to_pandas()
            y_val = df_val[cfg.dataset.target].to_pandas()

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(cfg.training.early_stopping_rounds, verbose=False)],
            )

            y_pred = cast(np.ndarray, model.predict(X_val))
            fold_maes.append(mae(np.asarray(y_val), y_pred))

        return float(np.mean(fold_maes))

    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = f"sqlite:///{output_dir.absolute()}/lgbm_study.db"

    study = optuna.create_study(
        study_name="lgbm_tuning",
        direction="minimize",
        storage=storage_path,
        load_if_exists=True,
        sampler=TPESampler(seed=cfg.training.random_state),
        pruner=MedianPruner(n_warmup_steps=10),
    )

    print(f"Starting local tuning with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save outputs
    best_params_path = output_dir / "best_params.json"
    trials_path = output_dir / "trials.csv"

    print(f"Saving best parameters to {best_params_path}...")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"Saving trials history to {trials_path}...")
    study.trials_dataframe().to_csv(trials_path, index=False)

    print(f"Tuning complete. Best MAE: {study.best_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning LightGBM locally.")
    parser.add_argument("--trials", type=int, help="Number of Optuna trials (overrides config)")

    args = parser.parse_args()
    tune_lgbm(n_trials_override=args.trials)
