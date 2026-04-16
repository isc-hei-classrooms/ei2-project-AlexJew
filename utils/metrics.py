"""Evaluation metrics for energy load forecasting."""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def aggregate_to_hourly(
    y: np.ndarray,
    timestamps: pd.Series | np.ndarray,
) -> np.ndarray:
    """Average values within each UTC hour and broadcast back to 15-min timestamps.

    Each quarter-hour slot receives its hour's mean, producing an array of the
    same length as y. This matches OIKEN's forecast format: a single hourly
    value repeated for all four 15-min slots within that hour.
    """
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "y": np.asarray(y, dtype=float),
        }
    ).dropna(subset=["timestamp"])
    # OIKEN forecast intervals start at :15 (e.g. [00:15, 01:00], [01:15, 02:00]).
    # Subtract 15 min before flooring so these slots are grouped correctly.
    frame["hour"] = (frame["timestamp"] - pd.Timedelta("15min")).dt.floor("h")
    frame["y_hourly"] = frame.groupby("hour")["y"].transform("mean")
    return frame["y_hourly"].to_numpy()


def _hourly_means(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average both arrays to hourly resolution, returning (y_true_hourly, y_pred_hourly)."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    ).dropna(subset=["timestamp"])

    hourly = (
        frame.assign(hour=frame["timestamp"].dt.floor("h"))
        .groupby("hour", sort=True, as_index=False)[["y_true", "y_pred"]]
        .mean()
    )
    return hourly["y_true"].to_numpy(), hourly["y_pred"].to_numpy()


def mae_hourly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series | np.ndarray,
) -> float:
    """Calculate hourly Mean Absolute Error after averaging within each hour."""
    y_true_hourly, y_pred_hourly = _hourly_means(y_true, y_pred, timestamps)
    return mae(y_true_hourly, y_pred_hourly)


def rmse_hourly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series | np.ndarray,
) -> float:
    """Calculate hourly Root Mean Square Error after averaging within each hour."""
    y_true_hourly, y_pred_hourly = _hourly_means(y_true, y_pred, timestamps)
    return rmse(y_true_hourly, y_pred_hourly)
