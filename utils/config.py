import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomlkit
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class InfluxConfig:
    url: str
    org: str
    bucket: str
    token: Optional[str]


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    models_dir: str
    tuning_dir: str


@dataclass(frozen=True)
class DatasetConfig:
    version: str
    target: str
    fill_columns: List[str]


@dataclass(frozen=True)
class LocationsConfig:
    cities: List[str]


@dataclass(frozen=True)
class ModelsConfig:
    features_version: str
    lgbm_baseline_version: str
    lgbm_tuned_version: str
    ridge_version: str


@dataclass(frozen=True)
class TrainingLgbmConfig:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    reg_lambda: float
    objective: str


@dataclass(frozen=True)
class TrainingLgbmTunedConfig:
    objective: str
    bagging_freq: int
    n_estimators: int


@dataclass(frozen=True)
class TrainingRidgeConfig:
    alpha_min_log: float
    alpha_max_log: float
    alpha_num: int
    cv_folds: int


@dataclass(frozen=True)
class TrainingConfig:
    validation_days: int
    random_state: int
    early_stopping_rounds: int
    lgbm_baseline: TrainingLgbmConfig
    lgbm_tuned: TrainingLgbmTunedConfig
    ridge: TrainingRidgeConfig


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int
    objective: str
    bagging_freq: int
    n_estimators: int
    folds: List[Dict[str, Any]]
    search_space: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class Config:
    influx: InfluxConfig
    paths: PathsConfig
    dataset: DatasetConfig
    locations: LocationsConfig
    models: ModelsConfig
    training: TrainingConfig
    tuning: TuningConfig
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def _data_path(self) -> Path:
        return PROJECT_ROOT / self.paths.data_dir

    def _models_path(self) -> Path:
        return PROJECT_ROOT / self.paths.models_dir

    def train_parquet_path(self) -> Path:
        return self._data_path() / f"df_train_{self.dataset.version}.parquet"

    def test_parquet_path(self) -> Path:
        return self._data_path() / f"df_test_{self.dataset.version}.parquet"

    def features_json_path(self) -> Path:
        return self._models_path() / f"model_features_{self.models.features_version}.json"

    def lgbm_baseline_path(self) -> Path:
        return self._models_path() / f"lgb_default_{self.models.lgbm_baseline_version}.txt"

    def lgbm_tuned_path(self) -> Path:
        return self._models_path() / f"lgb_tuned_{self.models.lgbm_tuned_version}.txt"

    def ridge_path(self) -> Path:
        return self._models_path() / f"ridge_{self.models.ridge_version}.joblib"

    def scaler_path(self) -> Path:
        # Scaler usually follows features or general dataset version
        return self._models_path() / f"scaler_{self.models.features_version}.joblib"


def load_config() -> Config:
    """Loads config.toml and .env, returning a frozen dataclass tree."""
    load_dotenv(PROJECT_ROOT / ".env")

    with open(CONFIG_PATH, "rb") as f:
        toml_data = tomlkit.load(f)

    # Influx
    influx_data = toml_data.get("influx", {})
    influx_config = InfluxConfig(
        url=influx_data.get("url", ""),
        org=influx_data.get("org", ""),
        bucket=influx_data.get("bucket", ""),
        token=os.getenv("INFLUXDB_TOKEN"),
    )

    # Paths
    paths_data = toml_data.get("paths", {})
    paths_config = PathsConfig(**paths_data)

    # Dataset
    dataset_data = toml_data.get("dataset", {})
    dataset_config = DatasetConfig(**dataset_data)

    # Locations
    locations_data = toml_data.get("locations", {})
    locations_config = LocationsConfig(**locations_data)

    # Models
    models_data = toml_data.get("models", {})
    models_config = ModelsConfig(**models_data)

    # Training
    training_data = toml_data.get("training", {})
    lgbm_baseline = TrainingLgbmConfig(**training_data.get("lgbm_baseline", {}))
    lgbm_tuned = TrainingLgbmTunedConfig(**training_data.get("lgbm_tuned", {}))
    ridge = TrainingRidgeConfig(**training_data.get("ridge", {}))

    training_config = TrainingConfig(
        validation_days=training_data.get("validation_days", 90),
        random_state=training_data.get("random_state", 42),
        early_stopping_rounds=training_data.get("early_stopping_rounds", 50),
        lgbm_baseline=lgbm_baseline,
        lgbm_tuned=lgbm_tuned,
        ridge=ridge,
    )

    # Tuning
    tuning_data = toml_data.get("tuning", {})
    tuning_config = TuningConfig(**tuning_data)

    # Raw Data (nested)
    raw_data = toml_data.get("raw_data", {})

    return Config(
        influx=influx_config,
        paths=paths_config,
        dataset=dataset_config,
        locations=locations_config,
        models=models_config,
        training=training_config,
        tuning=tuning_config,
        raw_data=raw_data,
    )


def update_version(section: str, key: str, value: str) -> None:
    """Updates a value in config.toml while preserving formatting and comments."""
    with open(CONFIG_PATH, "r") as f:
        toml_data = tomlkit.parse(f.read())

    # Handle nested sections (e.g. "raw_data.acquisition")
    parts = section.split(".")
    target = toml_data
    for part in parts:
        if part not in target:
            target[part] = tomlkit.table()
        target = target[part]

    target[key] = value

    with open(CONFIG_PATH, "w") as f:
        f.write(tomlkit.dumps(toml_data))
