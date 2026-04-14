# Energy Informatics II

This project provides a complete pipeline for net electricity load forecasting using various models including Ridge Regression and LightGBM.

## Features

- **Dependency management** with [`uv`](https://docs.astral.sh/uv/)
- **Code quality** via [`ruff`](https://docs.astral.sh/ruff/) and [`pyright`](https://github.com/microsoft/pyright)
- **Standalone Training Scripts** for reproducible model generation
- **Hyperparameter Tuning** with [`optuna`](https://optuna.org/)
- **Interactive Evaluation** using [`marimo`](https://marimo.io/) notebooks

---

## Getting Started

### 1. Environment Setup

Install the dependencies using `uv`:

```sh
uv sync
```

### 2. End-to-End Pipeline

The project is structured to separate data preparation, training, and evaluation.

#### Phase A: Data Preparation
Open the `notebooks/energy_prediction.py` notebook using marimo:

```sh
uv run marimo edit notebooks/energy_prediction.py
```

Run the notebook up to **Section 6.2 (Warmup period and clipping)**. This will generate timestamped training and testing snapshots in the `data/` directory and update the `dataset.version` in `config.toml`.

#### Phase B: Model Training
Once the data is prepared, you can train the models using the standalone scripts in the `scripts/` directory. These scripts read paths and hyperparameters from `config.toml` and automatically update the model version pointers after a successful run.

**Train Ridge Regression (Baseline):**
```sh
uv run python scripts/train_ridge.py
```

**Train LightGBM (Baseline):**
```sh
uv run python scripts/train_lgbm_baseline.py
```

#### Phase C: Hyperparameter Tuning (Optional)
To find the optimal parameters for the LightGBM model, run the tuning script. This uses expanding-window cross-validation with settings from `config.toml`.

```sh
uv run python scripts/tune_lgbm.py --trials 100
```
The best parameters will be saved to `tuning_results/best_params.json`.

#### Phase D: Retrain Tuned Model
After tuning, refit the LightGBM model on the full training set using the discovered best parameters:

```sh
uv run python scripts/train_lgbm_tuned.py
```

#### Phase E: Final Evaluation
Return to the `energy_prediction.py` notebook. The evaluation sections will use `config.toml` to identify and load the active model versions from the `models/` directory for comparison and visualization.

---

## Configuration

All non-secret configuration is centralized in `config.toml`. This includes:
- Paths for data, models, and tuning results.
- Active dataset and model versions.
- Model hyperparameters and tuning search spaces.
- Target column and feature fill strategies.

Secrets (like `INFLUXDB_TOKEN`) are stored in a local `.env` file (not committed to git).

---

## Project Structure

- `notebooks/`: Marimo notebooks for analysis and evaluation.
- `scripts/`: Standalone Python scripts for training and tuning.
- `utils/`: Core logic for feature engineering, data processing, and metrics.
- `models/`: Storage for trained model boosters (`.txt`) and scalers (`.joblib`).
- `data/`: Parquet snapshots of prepared datasets.

## Development

### Pre-commit hooks

Install the pre-commit hooks to ensure code quality:

```sh
uv run pre-commit install
```

### Code formatting

We use `ruff` for formatting. You can run it manually:

```sh
uv run ruff format .
uv run ruff check . --fix
```
