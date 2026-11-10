"""FastAPI service: health, readiness, model metadata, forecasts, Prometheus metrics."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from demand_forecast.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfo,
    ReadyResponse,
)
from demand_forecast.serve.predictor import NotReadyError, Predictor

REQUESTS = Counter("forecast_requests_total", "Forecast requests", ["endpoint", "status"])
LATENCY = Histogram(
    "forecast_request_seconds",
    "Wall clock seconds per request",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
MODEL_INFO = Gauge("forecast_model_info", "1 when a model is loaded", ["version"])
POINTS = Counter("forecast_points_total", "Forecast points served")


class PredictionLog:
    """Append only JSONL log. One line per request, shipped to a warehouse by the log pipeline."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class Service:
    def __init__(self) -> None:
        self.predictor: Predictor | None = None
        self.error: str | None = None
        self.log = PredictionLog(Path(os.environ.get("DF_PREDICTION_LOG", "reports/predictions.jsonl")))

    def load(self) -> None:
        model_dir = Path(os.environ.get("DF_MODEL_DIR", "models"))
        panel_path = Path(os.environ.get("DF_PANEL_PATH", "data/processed/panel.parquet"))
        try:
            self.predictor = Predictor(model_dir, panel_path)
        except NotReadyError as exc:
            self.predictor = None
            self.error = str(exc)
        else:
            self.error = None
            MODEL_INFO.labels(version=self.predictor.version.version if self.predictor.version else "none").set(1)


service = Service()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    service.load()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Demand Forecasting Service",
        version="1.0.0",
        description="NYC taxi demand forecasts, zone by zone, every 30 minutes.",
        lifespan=lifespan,
    )

    def predictor_or_fail() -> Predictor:
        if service.predictor is None:
            raise HTTPException(status_code=503, detail=service.error or "model not loaded")
        return service.predictor

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    def healthz() -> HealthResponse:
        """Liveness: the process is up. No dependency checks."""
        return HealthResponse()

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
    def readyz() -> ReadyResponse:
        """Readiness: a model and a feature store are loaded."""
        predictor = service.predictor
        return ReadyResponse(
            status="ready" if predictor else "degraded",
            model_version=predictor.version.version if predictor and predictor.version else None,
            data_through=predictor.data_through if predictor else None,
        )

    @app.get("/v1/model", response_model=ModelInfo, tags=["model"])
    def model_info() -> ModelInfo:
        return predictor_or_fail().model_info()

    @app.get("/v1/zones/{zone_id}/forecast", response_model=ForecastResponse, tags=["forecast"])
    def zone_forecast(
        zone_id: int,
        slots: Annotated[int, Query(ge=1, le=48)] = 48,
        start: Annotated[datetime | None, Query(description="First slot, defaults to the next slot")] = None,
        quantiles: Annotated[bool, Query()] = True,
    ) -> ForecastResponse:
        return _run(zone_id=zone_id, slots=slots, start=start, quantiles=quantiles)

    @app.post("/v1/forecast", response_model=ForecastResponse, tags=["forecast"])
    def forecast(request: ForecastRequest) -> ForecastResponse:
        return _run(
            zone_id=request.zone_id,
            slots=request.slots,
            start=request.start,
            quantiles=request.quantiles,
        )

    def _run(
        *, zone_id: int, slots: int, start: datetime | None, quantiles: bool
    ) -> ForecastResponse:
        predictor = predictor_or_fail()
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            points = predictor.forecast(zone_id=zone_id, slots=slots, start=start, quantiles=quantiles)
        except LookupError as exc:
            REQUESTS.labels(endpoint="forecast", status="not_found").inc()
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            REQUESTS.labels(endpoint="forecast", status="invalid").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        REQUESTS.labels(endpoint="forecast", status="ok").inc()
        POINTS.inc(len(points))
        LATENCY.labels(endpoint="forecast").observe(latency_ms / 1000)
        served_at = datetime.now(UTC)
        service.log.write(
            {
                "request_id": request_id,
                "zone_id": zone_id,
                "slots": len(points),
                "model_version": predictor.version.version if predictor.version else "unknown",
                "latency_ms": round(latency_ms, 3),
                "served_at": served_at.isoformat(),
                "y_pred_mean": round(sum(point.trips_p50 for point in points) / max(len(points), 1), 4),
            }
        )
        return ForecastResponse(
            model_version=predictor.version.version if predictor.version else "unknown",
            generated_at=served_at,
            latency_ms=round(latency_ms, 3),
            points=points,
        )

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
