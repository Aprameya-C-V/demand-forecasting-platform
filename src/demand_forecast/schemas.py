"""Pydantic contracts shared by the pipelines and the API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_id: int = Field(ge=1, le=263, description="TLC pickup zone id")
    start: datetime | None = Field(
        default=None, description="First slot to forecast. Defaults to the next slot after the data."
    )
    slots: int = Field(default=48, ge=1, le=48 * 7, description="Number of 30 minute slots")
    quantiles: bool = Field(default=True, description="Include the p10 and p90 bands")


class ForecastPoint(BaseModel):
    ts: datetime
    zone_id: int
    trips_p50: float = Field(description="Median forecast, trips per slot")
    trips_p10: float | None = None
    trips_p90: float | None = None


class ForecastResponse(BaseModel):
    model_version: str
    generated_at: datetime
    latency_ms: float
    points: list[ForecastPoint]


class ModelInfo(BaseModel):
    version: str
    alias: str
    trained_at: datetime
    metrics: dict[str, float]
    features: list[str]
    data_window: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    model_version: str | None
    data_through: datetime | None


class PredictionLog(BaseModel):
    request_id: str
    zone_id: int
    ts: datetime
    y_pred: float
    model_version: str
    latency_ms: float
    served_at: datetime


class MetricsReport(BaseModel):
    window_start: datetime
    window_end: datetime
    rows: int
    wape: float
    mae: float
    rmse: float
    bias: float
    coverage_p10_p90: float | None = None
    by_hour: dict[str, float] = Field(default_factory=dict)


class DriftFeature(BaseModel):
    feature: str
    psi: float
    status: Literal["stable", "warning", "drift"]


class DriftReport(BaseModel):
    reference_window: str
    current_window: str
    features: list[DriftFeature]
    max_psi: float
    status: Literal["stable", "warning", "drift"]


class MonitorReport(BaseModel):
    generated_at: datetime
    model_version: str
    metrics: MetricsReport
    drift: DriftReport
    alerts: list[str]
    passed: bool
