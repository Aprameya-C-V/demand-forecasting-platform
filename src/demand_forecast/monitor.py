"""Monitoring: accuracy on realised demand, feature drift, and the alert decision."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from demand_forecast.config import MonitoringConfig
from demand_forecast.evaluate import summarize, wape
from demand_forecast.schemas import DriftFeature, DriftReport, MetricsReport, MonitorReport

EPSILON = 1e-6


def psi(edges: list[float], expected: list[float], values: np.ndarray) -> float:
    """Population stability index between a reference distribution and a new sample."""
    counts, _ = np.histogram(values, bins=np.array(edges))
    actual = counts / max(counts.sum(), 1)
    expected_arr = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    actual_arr = np.clip(actual, EPSILON, None)
    return float(np.sum((actual_arr - expected_arr) * np.log(actual_arr / expected_arr)))


def classify(value: float, threshold: float) -> Literal["stable", "warning", "drift"]:
    if value >= threshold:
        return "drift"
    if value >= threshold / 2:
        return "warning"
    return "stable"


def load_reference(path: Path) -> dict[str, dict[str, list[float]]]:
    payload: dict[str, dict[str, list[float]]] = json.loads(path.read_text())
    return payload


def drift_report(
    reference: dict[str, dict[str, list[float]]],
    features: dict[str, np.ndarray],
    *,
    threshold: float,
    reference_window: str,
    current_window: str,
) -> DriftReport:
    """PSI per drifting feature, with an overall status driven by the worst feature."""
    items = [
        DriftFeature(
            feature=name,
            psi=round(value, 4),
            status=classify(value, threshold),
        )
        for name, stats in reference.items()
        if name in features
        for value in [psi(stats["edges"], stats["proportions"], features[name])]
    ]
    items.sort(key=lambda item: item.psi, reverse=True)
    worst = max((item.psi for item in items), default=0.0)
    return DriftReport(
        reference_window=reference_window,
        current_window=current_window,
        features=items,
        max_psi=round(worst, 4),
        status=classify(worst, threshold),
    )


def metrics_report(
    ts: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> MetricsReport:
    """Accuracy over a window, overall and by hour of day."""
    summary = summarize(y, y_hat, lower=lower, upper=upper)
    hours = ts.astype("datetime64[h]").astype(int) % 24
    by_hour = {
        str(hour): round(wape(y[hours == hour], y_hat[hours == hour]), 4)
        for hour in range(24)
        if (hours == hour).any()
    }
    return MetricsReport(
        window_start=datetime.fromtimestamp(int(ts.min().astype("datetime64[s]").astype(int)), UTC),
        window_end=datetime.fromtimestamp(int(ts.max().astype("datetime64[s]").astype(int)), UTC),
        rows=int(summary["rows"]),
        wape=round(summary["wape"], 4),
        mae=round(summary["mae"], 4),
        rmse=round(summary["rmse"], 4),
        bias=round(summary["bias"], 4),
        coverage_p10_p90=round(summary["coverage_p10_p90"], 4)
        if "coverage_p10_p90" in summary
        else None,
        by_hour=by_hour,
    )


def hourly_series(window: object, hours: int = 72) -> list[dict[str, object]]:
    """Actual versus predicted totals per hour, for the dashboard chart."""
    ts = window.ts  # type: ignore[attr-defined]
    y = window.y  # type: ignore[attr-defined]
    y_hat = window.y_hat  # type: ignore[attr-defined]
    buckets = ts.astype("datetime64[h]")
    unique = np.unique(buckets)[-hours:]
    series: list[dict[str, object]] = []
    for hour in unique:
        mask = buckets == hour
        series.append(
            {
                "hour": str(hour).replace("T", " "),
                "actual": round(float(y[mask].sum()), 1),
                "predicted": round(float(y_hat[mask].sum()), 1),
            }
        )
    return series


def zone_breakdown(window: object, limit: int = 8) -> list[dict[str, object]]:
    """Per zone accuracy over the window, busiest zones first."""
    zone = window.zone  # type: ignore[attr-defined]
    y = window.y  # type: ignore[attr-defined]
    y_hat = window.y_hat  # type: ignore[attr-defined]
    rows: list[dict[str, object]] = []
    for zone_id in np.unique(zone):
        mask = zone == zone_id
        rows.append(
            {
                "zone_id": int(zone_id),
                "trips": round(float(y[mask].sum()), 1),
                "wape": round(float(wape(y[mask], y_hat[mask])), 4),
            }
        )
    rows.sort(key=lambda row: float(row["trips"]), reverse=True)  # type: ignore[arg-type]
    return rows[:limit]


def monitor(
    *,
    model_version: str,
    metrics: MetricsReport,
    drift: DriftReport,
    cfg: MonitoringConfig,
) -> MonitorReport:
    """Apply thresholds. Any breach fails the run and is reported as an alert."""
    alerts: list[str] = []
    if metrics.rows < cfg.min_rows:
        alerts.append(f"only {metrics.rows} rows in window, below the {cfg.min_rows} minimum")
    if metrics.wape > cfg.wape_threshold:
        alerts.append(f"WAPE {metrics.wape:.3f} above threshold {cfg.wape_threshold:.3f}")
    if abs(metrics.bias) > cfg.bias_threshold:
        alerts.append(f"bias {metrics.bias:+.3f} outside +-{cfg.bias_threshold:.3f}")
    if drift.status == "drift":
        alerts.append(f"feature drift, max PSI {drift.max_psi:.3f} above {cfg.psi_threshold:.3f}")
    return MonitorReport(
        generated_at=datetime.now(UTC),
        model_version=model_version,
        metrics=metrics,
        drift=drift,
        alerts=alerts,
        passed=not alerts,
    )
