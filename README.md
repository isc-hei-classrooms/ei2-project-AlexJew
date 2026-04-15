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

Run the notebook up to **Section 6.2 (Warmup period and clipping)**. This will generate the necessary training and testing snapshots in the `data/` directory:
- `data/df_train_latest.parquet`
- `data/df_test_latest.parquet`
- `models/model_features_latest.json`

#### Phase B: Model Training
Once the data is prepared, you can train the models using the standalone scripts in the `scripts/` directory.

**Train Ridge Regression (Baseline):**
```sh
uv run python scripts/train_ridge.py
```

**Train LightGBM (Baseline):**
```sh
uv run python scripts/train_lgbm_baseline.py
```

#### Phase C: Hyperparameter Tuning (Optional)
To find the optimal parameters for the LightGBM model, run the tuning script. This uses 3-fold expanding-window cross-validation over the 2024 training data.

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
Return to the `energy_prediction.py` notebook. The evaluation sections (6.4, 6.5, 6.6) will automatically detect and load the `latest` model files from the `models/` directory for comparison and visualization.

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
