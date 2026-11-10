"""Inference: load a registered model and the feature store, answer forecasts, replay traffic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from demand_forecast.features import FeatureMatrix, build_matrix
from demand_forecast.registry import ModelRegistry
from demand_forecast.schemas import ForecastPoint, ModelInfo


class NotReadyError(RuntimeError):
    """Raised when no model or no feature store is available."""


@dataclass
class ReplayWindow:
    ts: np.ndarray
    y: np.ndarray
    y_hat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    features: dict[str, np.ndarray]


class Predictor:
    """Loads the production model plus the panel once, then serves from memory.

    The panel is the feature store: it holds realised demand up to the last published slot and one
    extra day of empty slots, which is what the service forecasts.
    """

    def __init__(self, model_dir: Path, panel_path: Path, *, alias: str = "production") -> None:
        self.model_dir = model_dir
        self.alias = alias
        self.registry = ModelRegistry(model_dir)
        resolved = self.registry.resolve(alias)
        if resolved is None:
            raise NotReadyError(f"no model registered under alias {alias!r} in {model_dir}")
        self.version = resolved
        if not panel_path.exists():
            raise NotReadyError(f"feature store not found at {panel_path}")

        self.metadata = json.loads((self.version.path / "metadata.json").read_text())
        self.features: list[str] = list(self.metadata["features"])
        self.boosters = {
            name: lgb.Booster(model_file=str(self.version.path / f"{name}.txt"))
            for name in ("model_mean", "model_q10", "model_q90")
            if (self.version.path / f"{name}.txt").exists()
        }
        reference_path = self.version.path / "reference.json"
        self.reference: dict[str, dict[str, list[float]]] = (
            json.loads(reference_path.read_text()) if reference_path.exists() else {}
        )

        panel = pl.read_parquet(panel_path)
        self.matrix: FeatureMatrix = build_matrix(panel, horizon_days=1)
        self.data_through = self.matrix.last_ts
        self.forecast_start = self.data_through + timedelta(minutes=30)
        self.forecast_end = self.forecast_start + timedelta(days=1) - timedelta(minutes=30)

    # ------------------------------------------------------------------ serving
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            version=self.version.version,
            alias=self.alias,
            trained_at=datetime.fromisoformat(self.metadata["trained_at"]),
            metrics={key: float(value) for key, value in self.metadata["metrics"].items()},
            features=self.features,
            data_window=dict(self.metadata["data_window"]),
        )

    def _predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        mean = np.asarray(self.boosters["model_mean"].predict(X), dtype=float)
        lower = (
            np.asarray(self.boosters["model_q10"].predict(X), dtype=float)
            if "model_q10" in self.boosters
            else None
        )
        upper = (
            np.asarray(self.boosters["model_q90"].predict(X), dtype=float)
            if "model_q90" in self.boosters
            else None
        )
        return mean, lower, upper

    def forecast(
        self,
        *,
        zone_id: int,
        slots: int = 48,
        start: datetime | None = None,
        quantiles: bool = True,
    ) -> list[ForecastPoint]:
        """Forecast consecutive slots for one zone from the newest data onwards."""
        start_ts = start or self.forecast_start
        if start_ts.tzinfo is not None:
            start_ts = start_ts.astimezone(UTC).replace(tzinfo=None)
        if zone_id not in {int(zone) for zone in self.matrix.zone_ids}:
            raise LookupError(f"unknown zone {zone_id}")
        latest = np.datetime64(start_ts, "us")
        if latest < np.datetime64(self.forecast_start) or latest > np.datetime64(self.forecast_end):
            raise ValueError(
                "start must fall inside the forecastable window "
                f"[{self.forecast_start}, {self.forecast_end}]"
            )

        mask = (self.matrix.zone == zone_id) & (self.matrix.ts >= latest)
        indices = np.flatnonzero(mask)[:slots]
        if indices.size == 0:
            raise ValueError(f"no slots available for zone {zone_id} from {start_ts}")
        X = self.matrix.X[indices]
        timestamps = self.matrix.ts[indices]
        mean, lower, upper = self._predict(X)
        return [
            ForecastPoint(
                ts=datetime.fromtimestamp(int(timestamp.astype("datetime64[s]").astype(int)), UTC).replace(
                    tzinfo=None
                ),
                zone_id=zone_id,
                trips_p50=max(0.0, float(value)),
                trips_p10=max(0.0, float(lower[index])) if quantiles and lower is not None else None,
                trips_p90=max(0.0, float(upper[index])) if quantiles and upper is not None else None,
            )
            for index, (timestamp, value) in enumerate(zip(timestamps, mean, strict=True))
        ]

    # --------------------------------------------------------------- monitoring
    def replay(self, days: int) -> ReplayWindow:
        """Score the most recent ``days`` of realised demand, as if it were live traffic."""
        history = ~np.isnan(self.matrix.y)
        cutoff = self.matrix.ts[history].max() - np.timedelta64(days, "D")
        mask = history & (self.matrix.ts > cutoff)
        X = self.matrix.X[mask]
        mean, lower, upper = self._predict(X)
        features = {
            name: X[:, index]
            for index, name in enumerate(self.features)
            if name in self.reference
        }
        return ReplayWindow(
            ts=self.matrix.ts[mask],
            y=self.matrix.y[mask],
            y_hat=mean,
            lower=lower if lower is not None else mean,
            upper=upper if upper is not None else mean,
            features=features,
        )
