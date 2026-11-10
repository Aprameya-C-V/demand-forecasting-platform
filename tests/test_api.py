from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from demand_forecast.schemas import ForecastPoint, ModelInfo
from demand_forecast.serve import app as app_module


class FakeVersion:
    version = "v-test"


class FakeMatrix:
    zone_ids = __import__("numpy").array([1, 2, 161])


class FakePredictor:
    version = FakeVersion()
    data_through = datetime(2026, 1, 31, 23, 30)
    forecast_start = datetime(2026, 2, 1)
    forecast_end = datetime(2026, 2, 1, 23, 30)
    matrix = FakeMatrix()

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            version="v-test",
            alias="production",
            trained_at=datetime(2026, 2, 1, tzinfo=UTC),
            metrics={"wape": 0.21},
            features=["zone_id", "hour"],
            data_window={"train_start": "2026-01-01", "train_end": "2026-01-31"},
        )

    def forecast_many(
        self,
        *,
        zone_ids: list[int] | None = None,
        slots: int = 48,
        start: datetime | None = None,
        quantiles: bool = True,
    ) -> dict[int, list[ForecastPoint]]:
        selected = zone_ids or [int(zone) for zone in self.matrix.zone_ids]
        if len(selected) > 263:
            raise ValueError("at most 50 zones per batch request")
        return {
            zone: self.forecast(zone_id=zone, slots=slots, start=start, quantiles=quantiles)
            for zone in selected
        }

    def forecast(
        self,
        *,
        zone_id: int,
        slots: int = 48,
        start: datetime | None = None,
        quantiles: bool = True,
    ) -> list[ForecastPoint]:
        if zone_id == 404:
            raise LookupError(f"unknown zone {zone_id}")
        if zone_id == 400:
            raise ValueError("start must fall inside the forecastable window")
        base = start or self.forecast_start
        return [
            ForecastPoint(
                ts=base + timedelta(minutes=30 * index),
                zone_id=zone_id,
                trips_p50=10.0 + index,
                trips_p10=5.0 if quantiles else None,
                trips_p90=15.0 if quantiles else None,
            )
            for index in range(slots)
        ]


@pytest.fixture
def client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setattr(app_module.service, "load", lambda: None)
    monkeypatch.setattr(
        app_module.service,
        "log",
        app_module.PredictionLog(tmp_path / "predictions.jsonl"),
    )
    app_module.service.predictor = FakePredictor()  # type: ignore[assignment]
    app_module.service.error = None
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_the_model(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["model_version"] == "v-test"
    assert body["data_through"].startswith("2026-01-31")


def test_model_metadata(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["version"] == "v-test"
    assert body["metrics"]["wape"] == 0.21


def test_get_forecast_default_length(client: TestClient) -> None:
    body = client.get("/v1/zones/161/forecast?slots=6").json()
    assert len(body["points"]) == 6
    assert body["points"][0]["trips_p50"] == 10.0
    assert body["points"][-1]["ts"].startswith("2026-02-01T02:30")


def test_post_forecast_validates_the_body(client: TestClient) -> None:
    ok = client.post("/v1/forecast", json={"zone_id": 132, "slots": 2, "quantiles": False})
    assert ok.status_code == 200
    assert ok.json()["points"][0]["trips_p10"] is None

    invalid = client.post("/v1/forecast", json={"zone_id": 999, "slots": 2})
    assert invalid.status_code == 422

    extra = client.post("/v1/forecast", json={"zone_id": 1, "slots": 2, "bogus": 1})
    assert extra.status_code == 422


def test_errors_map_to_http_status(client: TestClient) -> None:
    assert client.get("/v1/zones/404/forecast").status_code == 404
    assert client.get("/v1/zones/400/forecast").status_code == 422


def test_predictions_are_logged(client: TestClient, tmp_path) -> None:
    client.get("/v1/zones/161/forecast?slots=2")
    log = tmp_path / "predictions.jsonl"
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    assert '"zone_id": 161' in lines[0]
    assert '"latency_ms"' in lines[0]


def test_zones_endpoint_lists_the_feature_store(client: TestClient) -> None:
    assert client.get("/v1/zones").json() == [1, 2, 161]


def test_batch_forecast_returns_every_zone(client: TestClient) -> None:
    body = client.post("/v1/forecast/batch", json={"slots": 4, "quantiles": False}).json()
    assert body["zones"] == 3
    assert body["points"] == 12
    assert [zone["zone_id"] for zone in body["forecasts"]] == [1, 2, 161]


def test_batch_rejects_more_than_the_zone_cap(client: TestClient) -> None:
    response = client.post("/v1/forecast/batch", json={"zone_ids": list(range(1, 265))})
    assert response.status_code == 422


def test_batch_unknown_zone_is_not_found(client: TestClient) -> None:
    response = client.post("/v1/forecast/batch", json={"zone_ids": [404]})
    assert response.status_code == 404


def test_responses_carry_request_and_model_headers(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Model-Version"] == "v-test"
    assert response.headers["Server-Timing"].startswith("app;dur=")


def test_access_log_is_json(client: TestClient, caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="demand_forecast.access"):
        client.get("/v1/zones/161/forecast?slots=1")
    assert any("request" in record.getMessage() for record in caplog.records)


def test_root_points_at_the_docs(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert body["version"]


def test_metrics_endpoint_exposes_prometheus(client: TestClient) -> None:
    client.get("/v1/zones/161/forecast?slots=1")
    body = client.get("/metrics").text
    assert "forecast_requests_total" in body
    assert "forecast_points_total" in body


def test_service_is_unavailable_without_a_model(monkeypatch) -> None:
    monkeypatch.setattr(app_module.service, "load", lambda: None)
    app_module.service.predictor = None
    app_module.service.error = "no model registered"
    with TestClient(app_module.app) as test_client:
        assert test_client.get("/readyz").json()["status"] == "degraded"
        assert test_client.get("/v1/model").status_code == 503
        assert test_client.get("/v1/zones/1/forecast").status_code == 503
