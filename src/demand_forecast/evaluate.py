"""Offline and streaming metrics. One implementation, used by training, tests and monitoring."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def wape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error over total demand."""
    denominator = float(np.sum(np.abs(y)))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.abs(y - y_hat)) / denominator)


def bias(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Signed relative bias. Positive means the model over-forecasts."""
    denominator = float(np.mean(y))
    if denominator == 0:
        return 0.0
    return float((np.mean(y_hat) - denominator) / denominator)


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals inside the predicted interval."""
    return float(np.mean((y >= lower) & (y <= upper)))


def pinball(y: np.ndarray, y_hat: np.ndarray, alpha: float) -> float:
    diff = y - y_hat
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def summarize(
    y: np.ndarray,
    y_hat: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    alpha: float = 0.8,
) -> dict[str, float]:
    """Standard report for one model or one monitoring window."""
    metrics = {
        "mae": mae(y, y_hat),
        "rmse": rmse(y, y_hat),
        "wape": wape(y, y_hat),
        "bias": bias(y, y_hat),
        "rows": float(len(y)),
    }
    if lower is not None and upper is not None:
        metrics["coverage_p10_p90"] = coverage(y, lower, upper)
        metrics["pinball_lower"] = pinball(y, lower, (1 - alpha) / 2)
        metrics["pinball_upper"] = pinball(y, upper, 1 - (1 - alpha) / 2)
    return metrics
