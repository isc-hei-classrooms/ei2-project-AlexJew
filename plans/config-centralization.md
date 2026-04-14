# Centralize configuration into `config.toml` + secrets-only `.env`

## Context

Hardcoded values are scattered across `scripts/*.py`, `utils/*.py`, and the
marimo notebook: paths, hyperparameters, CV folds, Optuna search bounds,
city lists, the target column, validation windows, and InfluxDB endpoints.
Dataset and model versioning is currently handled with `_latest`-suffixed
file copies written alongside each timestamped artifact — losing the link
between "what was the baseline on date X" and any reviewable history.

Goal: a single committed `config.toml` holds all non-secret configuration,
including explicit `version` pointers for the active dataset and each
model. `.env` keeps only `INFLUXDB_TOKEN`. Scripts auto-update the version
pointers in `config.toml` after successful runs; the user reviews the git
diff and commits. Reproducibility moves from filesystem state (`_latest`)
to git history.

## Design decisions (agreed with user)

- **Format:** TOML (`config.toml`). Python stdlib `tomllib` reads; `tomlkit`
  writes while preserving comments and layout.
- **Split:** `.env` holds only secrets (`INFLUXDB_TOKEN`). Everything else
  lives in `config.toml`, committed to git.
- **Version pointers:** written into `config.toml` by scripts after success
  (A1). User reviews and commits the diff.
- **Raw CSV acquisition:** per-city timestamps in `[raw_data.acquisition]`.
- **Notebook refactor:** extract parquet + features-JSON write logic into
  `utils/data_processing.py` so the notebook and a future CLI share it.
- **`_latest` files:** deleted at the end of the migration.

## `config.toml` schema

```toml
[paths]
data_dir    = "data"
models_dir  = "models"
tuning_dir  = "tuning_results"

[influx]
url    = "https://timeseries.hevs.ch"
org    = "HESSOVS"
bucket = "MeteoSuisse"
# INFLUXDB_TOKEN stays in .env

[dataset]
version      = "2026-04-14_19-13"   # auto-updated by data pipeline
target       = "load"
fill_columns = ["solar_remote_yield_ratio"]

[locations]
cities = ["basel", "bern", "evionnaz", "evolene_villa",
          "geneve", "montana", "pully", "sion", "visp", "zurich"]

[raw_data.acquisition]
# per-city timestamps, written by utils/data_acquisition.py
basel = "2026-04-10_10-39"
# ... etc

[models]
features_version      = "2026-04-14_19-13"
lgbm_baseline_version = "2026-04-14_19-13"
lgbm_tuned_version    = "2026-04-14_19-13"
ridge_version         = "2026-04-14_19-13"

[training]
validation_days       = 90
random_state          = 42
early_stopping_rounds = 50

[training.lgbm_baseline]
n_estimators      = 2000
learning_rate     = 0.05
num_leaves        = 63
min_child_samples = 20
reg_lambda        = 0.1
objective         = "regression_l1"

[training.lgbm_tuned]
objective    = "regression_l1"
bagging_freq = 5
n_estimators = 2000

[training.ridge]
alpha_min_log = -2
alpha_max_log = 3
alpha_num     = 20
cv_folds      = 5

[tuning]
n_trials     = 100
objective    = "regression_l1"
bagging_freq = 5
n_estimators = 2000

[[tuning.folds]]
split = 2023-10-01T00:00:00
name  = "Q4 2023"
# ... Q1/Q2/Q3 2024

[tuning.search_space]
learning_rate     = { low = 0.01,  high = 0.2,  log = true  }
num_leaves        = { low = 16,    high = 255               }
min_child_samples = { low = 5,     high = 100               }
feature_fraction  = { low = 0.5,   high = 1.0               }
bagging_fraction  = { low = 0.5,   high = 1.0               }
reg_alpha         = { low = 0.001, high = 10.0, log = true  }
reg_lambda        = { low = 0.001, high = 10.0, log = true  }
```

## `utils/config.py` API

```python
# New file. Reads config.toml + .env, returns a frozen dataclass tree.
def load_config() -> Config: ...
def update_version(section: str, key: str, value: str) -> None: ...

# Config exposes path helpers that resolve relative to project root:
#   cfg.train_parquet_path()       -> Path(".../data/df_train_<version>.parquet")
#   cfg.test_parquet_path()        -> Path(".../data/df_test_<version>.parquet")
#   cfg.features_json_path()       -> Path(".../models/model_features_<version>.json")
#   cfg.lgbm_baseline_path()       -> Path(".../models/lgb_default_<version>.txt")
#   cfg.lgbm_tuned_path()          -> Path(".../models/lgb_tuned_<version>.txt")
#   cfg.ridge_path()               -> Path(".../models/ridge_<version>.joblib")
#   cfg.scaler_path()              -> Path(".../models/scaler_<version>.joblib")
#   cfg.influx.token               -> merged from .env
```

## Writer ownership

| File                                    | Keys auto-updated on success                          |
| --------------------------------------- | ----------------------------------------------------- |
| `utils/data_acquisition.py`             | `[raw_data.acquisition] <city>`                       |
| `utils/data_processing.py` (new writers, called by notebook) | `[dataset] version`, `[models] features_version` |
| `scripts/train_ridge.py`                | `[models] ridge_version`                              |
| `scripts/train_lgbm_baseline.py`        | `[models] lgbm_baseline_version`                      |
| `scripts/train_lgbm_tuned.py`           | `[models] lgbm_tuned_version`                         |
| `scripts/tune_lgbm.py`                  | (no config mutation; writes `best_params.json` as today) |

## Execution checklist (one commit per step)

- [x] **1. Dependency + initial config file.** Add `tomlkit` to `pyproject.toml` (run `uv add tomlkit`). Create `config.toml` with only `[influx]` section (URL/org/bucket moved from `.env`). Remove those three keys from `.env`, leaving only `INFLUXDB_TOKEN`.
- [x] **2. Config loader.** Create `utils/config.py` with `load_config()` returning a dataclass tree and `update_version(section, key, value)` using `tomlkit`. Smoke-test via `python -c "from utils.config import load_config; print(load_config().influx.url)"`.
- [x] **3. Migrate InfluxDB consumers.** Replace `os.environ["INFLUXDB_URL/ORG/BUCKET"]` reads in `utils/data_acquisition.py` (and anywhere else) with `cfg.influx.*`. Token stays from env.
- [x] **4. Extend config schema.** Add `[paths]`, `[dataset]`, `[models]`, `[locations]`, `[training.*]`, `[tuning.*]` sections with current values. Extend `Config` dataclass and add path helpers. No consumer changes yet.
- [x] **5. Refactor `train_ridge.py`.** Read data/features/output paths and (no-op here) hyperparams from `Config`. Verify script runs end-to-end against existing artifacts.
- [x] **6. Refactor `train_lgbm_baseline.py`.** Same pattern; read LightGBM hyperparameters from `[training.lgbm_baseline]` and `validation_days` / `early_stopping_rounds` from `[training]`. Verify run.
- [x] **7. Refactor `train_lgbm_tuned.py`.** Read paths and merged params from config. Verify run.
- [x] **8. Refactor `tune_lgbm.py`.** Read `folds`, `search_space`, `n_trials`, `objective`, `n_estimators`, `random_state` from `[tuning.*]`. Verify a short trial run (e.g. `--trials 2`).
- [x] **9. Auto-write model versions.** Add `update_version("models", "<name>_version", ts)` calls to each training script on successful save. Verify by running a script and inspecting `git diff config.toml`.
- [x] **10. Extract notebook write helpers into `utils/data_processing.py`.** Add `write_train_test_parquets(...)` and `write_model_features(...)` functions that write timestamped files AND call `update_version()` for `dataset.version` and `models.features_version`. Wire the notebook to call them.
- [x] **11. Per-city acquisition timestamps.** Update `utils/data_acquisition.py` to call `update_version("raw_data.acquisition", city, ts)` after each successful fetch.
- [ ] **12. Cleanup.** Delete all `*_latest.*` files under `data/` and `models/`. Update `README.md` to document the new config flow. Grep for any remaining hardcoded paths/cities/hyperparameters and migrate.

## Verification

After each step, before committing:
- Run `ruff check .` and pylance (per `AGENTS.md`).
- Run the affected script end-to-end and confirm expected output files exist.

After step 12 (end-to-end):
- Run the full notebook top-to-bottom, producing new timestamped parquets and features JSON, and confirm `config.toml` is updated with the new `dataset.version` and `models.features_version`.
- Run `scripts/train_ridge.py`, `scripts/train_lgbm_baseline.py`, `scripts/tune_lgbm.py --trials 5`, then `scripts/train_lgbm_tuned.py`, confirming each updates the right `[models]` key.
- `git diff config.toml` should show only the expected version bumps.
- Confirm no `_latest` files remain and no script references them.

## Critical files

- `config.toml` (new, committed)
- `.env` (trimmed to secrets)
- `utils/config.py` (new)
- `utils/data_acquisition.py`
- `utils/data_processing.py` (extended with write helpers)
- `utils/feature_engineering.py` (audit for hardcoded values when touched)
- `notebooks/energy_prediction.py` (audit for hardcoded values; wire write helpers)
- `scripts/train_ridge.py`
- `scripts/train_lgbm_baseline.py`
- `scripts/train_lgbm_tuned.py`
- `scripts/tune_lgbm.py`
- `pyproject.toml` / `uv.lock` (add `tomlkit`)
- `README.md` (document new config flow)
