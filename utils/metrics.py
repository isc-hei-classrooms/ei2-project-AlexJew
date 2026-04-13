"""Evaluation metrics for energy load forecasting."""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
