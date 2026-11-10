"""FastAPI service: health, readiness, model metadata, forecasts, Prometheus metrics.

Production behaviour, not a demo:

- every request gets an ID, echoed in ``X-Request-ID`` and in the structured access log
- the access log is one JSON object per line, ready for Loki, CloudWatch or a file shipper
- latency is measured by a middleware, exported to Prometheus and returned in ``Server-Timing``
- unhandled errors return a JSON envelope with the request ID, never a stack trace
- the model version is echoed in ``X-Model-Version`` so a dashboard or an A/B router can split on it
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from demand_forecast import __version__
from demand_forecast.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfo,
    ReadyResponse,
    ZoneForecast,
)
from demand_forecast.serve.predictor import NotReadyError, Predictor

REQUESTS = Counter("forecast_requests_total", "Forecast requests", ["endpoint", "status"])
LATENCY = Histogram(
    "forecast_request_seconds",
    "Wall clock seconds per request",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
MODEL_INFO = Gauge("forecast_model_info", "1 when a model is loaded", ["version"])
POINTS = Counter("forecast_points_total", "Forecast points served")
BATCH_ZONES = Histogram(
    "forecast_batch_zones",
    "Zones per batch request",
    buckets=(1, 5, 10, 25, 50),
)

LOGGER = logging.getLogger("demand_forecast.access")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, no free text, no tracebacks on the wire."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    level = os.environ.get("DF_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = True


class PredictionLog:
    """Append only JSONL log with one buffered handle.

    Opening and closing the file on every request turns a 2 ms forecast into a 40 ms one under
    concurrency, so the handle is kept for the process lifetime and guarded by a lock.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=8192)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, object]) -> None:
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            self._handle.close()


class Service:
    def __init__(self) -> None:
        self.predictor: Predictor | None = None
        self.error: str | None = None
        self.log = PredictionLog(
            Path(os.environ.get("DF_PREDICTION_LOG", "reports/predictions.jsonl"))
        )

    def load(self) -> None:
        model_dir = Path(os.environ.get("DF_MODEL_DIR", "models"))
        panel_path = Path(os.environ.get("DF_PANEL_PATH", "data/processed/panel.parquet"))
        try:
            self.predictor = Predictor(model_dir, panel_path)
        except NotReadyError as exc:
            self.predictor = None
            self.error = str(exc)
            LOGGER.warning("model not loaded", extra={"fields": {"error": str(exc)}})
        else:
            self.error = None
            version = self.predictor.version.version if self.predictor.version else "none"
            MODEL_INFO.labels(version=version).set(1)
            LOGGER.info(
                "model loaded",
                extra={
                    "fields": {"version": version, "data_through": str(self.predictor.data_through)}
                },
            )


service = Service()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    service.load()
    yield


DESCRIPTION = """
Forecast taxi trips per TLC zone per 30 minute slot, one day ahead.

The service reads a registered model version and a dense feature store, both loaded at startup.
It exposes the same metrics the promotion gate used, so a caller can see what the model was
promoted on. Every request is logged with a request ID, a version and a latency.
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Demand Forecasting API",
        version=__version__,
        description=DESCRIPTION,
        license_info={"name": "MIT"},
        openapi_tags=[
            {"name": "ops", "description": "Liveness, readiness and metrics."},
            {"name": "model", "description": "What is deployed and what it scored."},
            {"name": "forecast", "description": "Zone demand forecasts."},
        ],
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - the handler below turns this into a 500
            LOGGER.exception(
                "request failed",
                extra={"fields": {"request_id": request_id, "path": request.url.path}},
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        version = (
            service.predictor.version.version
            if service.predictor and service.predictor.version
            else "none"
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Model-Version"] = version
        response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.2f}"
        LOGGER.info(
            "request",
            extra={
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed_ms, 3),
                    "model_version": version,
                }
            },
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, _exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        LOGGER.exception("unhandled error", extra={"fields": {"request_id": request_id}})
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": request_id},
        )

    def predictor_or_fail() -> Predictor:
        if service.predictor is None:
            raise HTTPException(status_code=503, detail=service.error or "model not loaded")
        return service.predictor

    def _served_at() -> datetime:
        return datetime.now(UTC)

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"], summary="Liveness")
    def healthz() -> HealthResponse:
        """The process is up. No dependency checks, so a slow model never restarts the pod."""
        return HealthResponse()

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"], summary="Readiness")
    def readyz() -> ReadyResponse:
        """A model and a feature store are loaded and the panel has data."""
        predictor = service.predictor
        return ReadyResponse(
            status="ready" if predictor else "degraded",
            model_version=predictor.version.version if predictor and predictor.version else None,
            data_through=predictor.data_through if predictor else None,
        )

    @app.get("/v1/model", response_model=ModelInfo, tags=["model"], summary="Deployed model")
    def model_info() -> ModelInfo:
        return predictor_or_fail().model_info()

    @app.get(
        "/v1/zones",
        response_model=list[int],
        tags=["forecast"],
        summary="Zones covered by the feature store",
    )
    def zones() -> list[int]:
        return [int(zone) for zone in predictor_or_fail().matrix.zone_ids]

    @app.get(
        "/v1/zones/{zone_id}/forecast",
        response_model=ForecastResponse,
        tags=["forecast"],
        summary="Forecast one zone",
    )
    def zone_forecast(
        zone_id: int,
        slots: Annotated[int, Query(ge=1, le=48)] = 48,
        start: Annotated[
            datetime | None, Query(description="First slot, defaults to the next slot")
        ] = None,
        quantiles: Annotated[bool, Query(description="Include the p10 and p90 band")] = True,
    ) -> ForecastResponse:
        return _run(zone_id=zone_id, slots=slots, start=start, quantiles=quantiles)

    @app.post(
        "/v1/forecast",
        response_model=ForecastResponse,
        tags=["forecast"],
        summary="Forecast one zone, validated body",
    )
    def forecast(request: ForecastRequest) -> ForecastResponse:
        return _run(
            zone_id=request.zone_id,
            slots=request.slots,
            start=request.start,
            quantiles=request.quantiles,
        )

    @app.post(
        "/v1/forecast/batch",
        response_model=BatchForecastResponse,
        tags=["forecast"],
        summary="Forecast up to 50 zones in one call",
    )
    def forecast_batch(request: BatchForecastRequest) -> BatchForecastResponse:
        predictor = predictor_or_fail()
        started = time.perf_counter()
        try:
            forecasts = predictor.forecast_many(
                zone_ids=request.zone_ids,
                slots=request.slots,
                start=request.start,
                quantiles=request.quantiles,
            )
        except LookupError as exc:
            REQUESTS.labels(endpoint="batch", status="not_found").inc()
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            REQUESTS.labels(endpoint="batch", status="invalid").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        points = sum(len(items) for items in forecasts.values())
        REQUESTS.labels(endpoint="batch", status="ok").inc()
        POINTS.inc(points)
        BATCH_ZONES.observe(len(forecasts))
        LATENCY.labels(endpoint="batch").observe(latency_ms / 1000)
        return BatchForecastResponse(
            model_version=predictor.version.version if predictor.version else "unknown",
            generated_at=_served_at(),
            latency_ms=round(latency_ms, 3),
            zones=len(forecasts),
            points=points,
            forecasts=[
                ZoneForecast(zone_id=zone_id, points=items)
                for zone_id, items in sorted(forecasts.items())
            ],
        )

    def _run(
        *, zone_id: int, slots: int, start: datetime | None, quantiles: bool
    ) -> ForecastResponse:
        predictor = predictor_or_fail()
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            points = predictor.forecast(
                zone_id=zone_id, slots=slots, start=start, quantiles=quantiles
            )
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
        served_at = _served_at()
        service.log.write(
            {
                "request_id": request_id,
                "zone_id": zone_id,
                "slots": len(points),
                "model_version": predictor.version.version if predictor.version else "unknown",
                "latency_ms": round(latency_ms, 3),
                "served_at": served_at.isoformat(),
                "y_pred_mean": round(
                    sum(point.trips_p50 for point in points) / max(len(points), 1), 4
                ),
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

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "demand-forecasting-api",
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    return app


app = create_app()
