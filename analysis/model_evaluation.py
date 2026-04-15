# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "joblib>=1.5.3",
#     "lightgbm>=4.6.0",
#     "marimo>=0.22.4",
#     "metrics>=0.0.2",
#     "numpy>=2.4.4",
#     "polars>=1.39.3",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import json
    import os
    import sys

    import joblib
    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import polars as pl
    from pathlib import Path

    utils_path = Path("./utils")
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))

    import metrics

    return joblib, json, lgb, metrics, mo, np, os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Model evaluation

    This notebook loads pre-trained models and evaluates their forecasting performance on the test set.

    Models are trained by running the scripts in `scripts/` from the terminal:
    ```
    uv run python scripts/prepare_data.py       # generate train/test parquet files
    uv run python scripts/train_ridge.py        # Ridge regression
    uv run python scripts/train_lgbm_baseline.py  # LightGBM (default)
    uv run python scripts/train_lgbm_tuned.py   # LightGBM (tuned)
    ```

    ### Evaluation metrics
    - **MAE** (Mean Absolute Error): primary metric — average absolute prediction error
    - **RMSE** (Root Mean Square Error): secondary diagnostic — penalises large errors, useful for detecting struggles at peak periods

    ### Train / test split
    A **temporal split** at 1 October 2024 divides the data into:
    - **Training set**: Oct 2022 – Sep 2024 (~2 years)
    - **Test set**: Oct 2024 – Sep 2025 (~1 year)
    """)
    return


@app.cell(hide_code=True)
def _(json, metrics, mo, pl):
    _train_path = "data/df_train_latest.parquet"
    _test_path = "data/df_test_latest.parquet"
    _feat_path = "models/model_features_latest.json"

    df_train = pl.read_parquet(_train_path)
    df_test = pl.read_parquet(_test_path)
    with open(_feat_path) as _f:
        model_features = json.load(_f)

    _tr_min = df_train["utc_timestamp"].min().date()
    _tr_max = df_train["utc_timestamp"].max().date()
    _te_min = df_test["utc_timestamp"].min().date()
    _te_max = df_test["utc_timestamp"].max().date()

    if model_features and df_test.height > 0:
        X_test = df_test.select(model_features).to_pandas()
        y_test = df_test["load"].to_pandas()
    else:
        import pandas as _pd

        X_test = _pd.DataFrame()
        y_test = _pd.Series(dtype=float)

    mae = metrics.mae
    rmse = metrics.rmse


    def _cat(f):
        if any(f.startswith(p) for p in ("utc_sin_", "utc_cos_")):
            return "Cyclical"
        if "_forecast_" in f:
            return "Weather forecast"
        if "_measured_" in f:
            return "Weather measurement"
        if f.startswith(("load_lag_", "forecast_load_lag_")):
            return "Load lag"
        if (
            f in ("poa_irradiance", "solar_yield_30d", "solar_remote_yield_ratio")
            or "solar_capacity" in f
        ):
            return "Solar capacity"
        if f.startswith("solar_"):
            return "Solar production"
        if f.startswith("local_"):
            return "Temporal"
        return "Other"


    _feat_df = pl.DataFrame(
        {
            "feature": model_features,
            "category": [_cat(f) for f in model_features],
        }
    ).sort("category")

    mo.vstack(
        [
            mo.accordion({"Feature overview": _feat_df}),
            mo.accordion(
                {
                    "Training data": mo.ui.table(
                        df_train, max_columns=df_train.width
                    )
                }
            ),
        ]
    )
    return X_test, df_test, mae, model_features, rmse, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Baselines

    Two simple baselines establish reference performance to judge whether more complex models add value:

    1. **Persistence (t − 7 days)**: predict load at time _t_ using the load value from the same quarter-hour, one week earlier. Exploits the weekly periodicity (r ≈ 0.90 from section 5.5).
    2. **OIKEN forecast**: the operator's own day-ahead forecast (`forecast_load` column). This is the production benchmark that any new model should aim to beat.

    Both are evaluated on the same test window (2024-10-10 → 2025-09-29).
    """)
    return


@app.cell(hide_code=True)
def _(df_test, mae, mo, pl, rmse, y_test):
    # --- Persistence baseline: load at t - 7 days ---------------------------
    y_pred_persistence = df_test["load_persistence_7d"].to_numpy()

    # --- OIKEN baseline: forecast_load column --------------------------------
    y_pred_oiken = df_test["forecast_load"].to_numpy()
    y_test_np = y_test.to_numpy()

    # --- Evaluate ------------------------------------------------------------
    baseline_results = pl.DataFrame(
        {
            "model": ["Persistence (t-7d)", "OIKEN forecast"],
            "MAE": [
                mae(y_test_np, y_pred_persistence),
                mae(y_test_np, y_pred_oiken),
            ],
            "RMSE": [
                rmse(y_test_np, y_pred_persistence),
                rmse(y_test_np, y_pred_oiken),
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    # Store baseline predictions in a dict so they can be reused/compared later
    baseline_predictions = {
        "Persistence (t-7d)": y_pred_persistence,
        "OIKEN forecast": y_pred_oiken,
    }

    mo.vstack(
        [
            mo.md(
                "**Baseline performance on test set (load is standardised, mean 0 / std 1)**"
            ),
            mo.accordion({"Results table": baseline_results}),
        ]
    )
    return (baseline_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ridge regression

    A linear baseline with L2 regularisation. Ridge is a natural first-choice ML model:
    - **Fast**: closed-form solution, trains in seconds even on 69k × 131
    - **Interpretable**: each feature gets a signed weight
    - **Handles correlated features**: the L2 penalty shrinks collinear coefficients together (we have several correlated weather features across stations)

    **Preprocessing**: features are standardised (zero mean, unit variance) using statistics computed on the **training set only**, then applied to the test set. This prevents information from the test period from leaking into the preprocessing step.

    **Regularisation strength** (α): tuned via a small grid on a time-series cross-validation split inside the training data. For a first pass we use `RidgeCV` which automates this selection.
    """)
    return


@app.cell(hide_code=True)
def _(
    X_test,
    baseline_predictions,
    joblib,
    mae,
    mo,
    model_features,
    np,
    os,
    pl,
    rmse,
    y_test,
):
    _scaler_path = "models/scaler_latest.joblib"
    _ridge_path = "models/ridge_latest.joblib"

    if not os.path.exists(_scaler_path) or not os.path.exists(_ridge_path):
        mo.md(
            "⚠️ **Ridge model or scaler not found!** Run `python scripts/train_ridge.py` first."
        )
        ridge_model = None
        y_pred_ridge = np.zeros_like(y_test.to_numpy())
    else:
        # Load pre-trained scaler and model
        _scaler = joblib.load(_scaler_path)
        ridge_model = joblib.load(_ridge_path)

        # Standardise test features
        _X_test_scaled = _scaler.transform(X_test)
        y_pred_ridge = ridge_model.predict(_X_test_scaled)

    # Evaluate
    ridge_mae = mae(y_test.to_numpy(), y_pred_ridge)
    ridge_rmse = rmse(y_test.to_numpy(), y_pred_ridge)

    # Top features by absolute coefficient (if model exists)
    if ridge_model is not None:
        _coefs = (
            pl.DataFrame(
                {
                    "feature": model_features,
                    "coefficient": ridge_model.coef_,
                    "abs_coef": np.abs(ridge_model.coef_),
                }
            )
            .sort("abs_coef", descending=True)
            .drop("abs_coef")
        )
    else:
        _coefs = pl.DataFrame({"feature": [], "coefficient": []})

    # Combined results table
    all_results = pl.DataFrame(
        {
            "model": ["Persistence (t-7d)", "OIKEN forecast", "Ridge regression"],
            "MAE": [
                mae(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                mae(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                ridge_mae,
            ],
            "RMSE": [
                rmse(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                rmse(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                ridge_rmse,
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    baseline_predictions["Ridge regression"] = y_pred_ridge

    mo.vstack(
        [
            mo.md("**Model performance: Ridge regression (linear baseline)**"),
            mo.accordion({"Results table": all_results}),
            mo.accordion({"Ridge top 20 features": _coefs.head(20)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LightGBM

    Gradient-boosted decision trees — the workhorse for tabular time-series forecasting. Unlike Ridge, LightGBM:
    - Captures **non-linear relationships** (e.g. temperature thresholds above which cooling kicks in)
    - Captures **feature interactions** automatically (e.g. `hour × is_working_day × temperature`)
    - **No scaling required** — trees are invariant to monotonic feature transformations
    - Handles **hundreds of features** efficiently via histogram-based splits

    **Validation strategy**: reserve the last 3 months of training data (Jul-Sep 2024) as a validation set. Use **early stopping** on this set to pick the optimal number of boosting rounds — this mimics a realistic deployment scenario where the latest data validates the model.

    **Hyperparameters** (first pass, reasonable defaults):
    - `n_estimators=2000` with early stopping patience 50
    - `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=20`
    - `reg_lambda=0.1` (mild L2 regularisation)
    - `objective="regression_l1"` (MAE objective, matches our primary metric)
    """)
    return


@app.cell(hide_code=True)
def _(X_test, baseline_predictions, lgb, mae, mo, np, os, pl, rmse, y_test):
    _lgb_path = "models/lgb_default_latest.txt"

    if not os.path.exists(_lgb_path):
        mo.md(
            "⚠️ **LightGBM baseline model not found!** Run `python scripts/train_lgbm_baseline.py` first."
        )
        lgb_model = None
        y_pred_lgb = np.zeros_like(y_test.to_numpy())
    else:
        # Load pre-trained booster
        lgb_model = lgb.Booster(model_file=_lgb_path)
        y_pred_lgb = lgb_model.predict(X_test)

    # --- Evaluate ------------------------------------------------------------
    lgb_mae = mae(y_test.to_numpy(), y_pred_lgb)
    lgb_rmse = rmse(y_test.to_numpy(), y_pred_lgb)

    baseline_predictions["LightGBM"] = y_pred_lgb

    # Combined results
    lgb_results = pl.DataFrame(
        {
            "model": [
                "Persistence (t-7d)",
                "OIKEN forecast",
                "Ridge regression",
                "LightGBM",
            ],
            "MAE": [
                mae(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                mae(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                mae(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                lgb_mae,
            ],
            "RMSE": [
                rmse(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                rmse(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                rmse(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                lgb_rmse,
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    if lgb_model is not None:
        lgb_importance = pl.DataFrame(
            {
                "feature": lgb_model.feature_name(),
                "gain": lgb_model.feature_importance(importance_type="gain"),
            }
        ).sort("gain", descending=True)
    else:
        lgb_importance = pl.DataFrame({"feature": [], "gain": []})

    mo.vstack(
        [
            mo.md("**Model performance: LightGBM (baseline)**"),
            mo.accordion({"Results table": lgb_results}),
            mo.accordion({"Top 20 features by gain": lgb_importance.head(20)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LightGBM hyperparameter tuning

    To ensure robust performance across different seasonal patterns, hyperparameter tuning is performed outside this notebook using the `scripts/tune_lgbm.py` script.

    **Tuning Strategy:**
    - **Objective**: Minimize the Mean Absolute Error (MAE).
    - **Cross-Validation**: 4-fold expanding-window CV covering a full year (Oct 2023 – Sep 2024).
    - **Search Space**: Tunes learning rate, tree complexity, regularization, and sampling ratios using Optuna's TPE sampler.
    - **Refit**: After tuning, the model is refit on the full training set using the best parameters via `scripts/train_lgbm_tuned.py`.

    This section loads the resulting hyperparameters and the final tuned model for evaluation on the untouched Q4 2024 test set.
    """)
    return


@app.cell(hide_code=True)
def _(
    X_test,
    baseline_predictions,
    json,
    lgb,
    mae,
    mo,
    np,
    os,
    pl,
    rmse,
    y_test,
):
    _params_path = "tuning_results/best_params.json"
    _lgb_tuned_path = "models/lgb_tuned_latest.txt"

    if not os.path.exists(_params_path) or not os.path.exists(_lgb_tuned_path):
        mo.md(
            """
            ⚠️ **Tuned model or parameters not found!**
            Run the following in your terminal:
            1. `uv run python scripts/tune_lgbm.py --trials 100`
            2. `uv run python scripts/train_lgbm_tuned.py`
            """
        )
        lgb_tuned_model = None
        y_pred_lgb_tuned = np.zeros_like(y_test.to_numpy())
        best_params = {}
    else:
        # Load best parameters for display
        with open(_params_path) as f:
            best_params = json.load(f)

        # Load pre-trained tuned booster
        lgb_tuned_model = lgb.Booster(model_file=_lgb_tuned_path)
        y_pred_lgb_tuned = lgb_tuned_model.predict(X_test)

    # --- Evaluate ------------------------------------------------------------
    lgb_tuned_mae = mae(y_test.to_numpy(), y_pred_lgb_tuned)
    lgb_tuned_rmse = rmse(y_test.to_numpy(), y_pred_lgb_tuned)

    baseline_predictions["LightGBM (tuned)"] = y_pred_lgb_tuned

    # --- Results summary -----------------------------------------------------
    tuned_results = pl.DataFrame(
        {
            "model": [
                "Persistence (t-7d)",
                "OIKEN forecast",
                "Ridge regression",
                "LightGBM (default)",
                "LightGBM (tuned)",
            ],
            "MAE": [
                mae(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                mae(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                mae(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                mae(y_test.to_numpy(), baseline_predictions["LightGBM"]),
                lgb_tuned_mae,
            ],
            "RMSE": [
                rmse(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                rmse(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                rmse(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                rmse(y_test.to_numpy(), baseline_predictions["LightGBM"]),
                lgb_tuned_rmse,
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    _best_params_df = pl.DataFrame(
        {
            "parameter": list(best_params.keys()),
            "value": [
                f"{v:.4g}" if isinstance(v, float) else str(v)
                for v in best_params.values()
            ],
        }
    )

    if lgb_tuned_model is not None:
        lgb_tuned_importance = pl.DataFrame(
            {
                "feature": lgb_tuned_model.feature_name(),
                "gain": lgb_tuned_model.feature_importance(importance_type="gain"),
            }
        ).sort("gain", descending=True)
    else:
        lgb_tuned_importance = pl.DataFrame({"feature": [], "gain": []})

    mo.vstack(
        [
            mo.md("**Model performance: LightGBM (tuned)**"),
            mo.accordion({"Results table": tuned_results}),
            mo.accordion(
                {
                    "Best hyperparameters (from local tuning)": _best_params_df,
                    "Top 20 features by gain": lgb_tuned_importance.head(20),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model analysis

    In this section, we explore the results of our forecasting models in more detail, analyzing error distributions, seasonal performance, and specific failure modes.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
