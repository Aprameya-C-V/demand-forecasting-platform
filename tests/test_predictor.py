from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from demand_forecast.config import ModelConfig, PipelineConfig, SplitConfig
from demand_forecast.features import build_matrix
from demand_forecast.registry import ModelRegistry
from demand_forecast.serve.predictor import NotReadyError, Predictor
from demand_forecast.train import train

SLOTS = 48


def _panel(days: int = 24, zones: int = 3) -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    rows = [
        {
            "ts": start + timedelta(days=day, minutes=30 * slot),
            "zone_id": zone,
            "trips": (slot % 10) + zone * 2 + (day % 7),
        }
        for zone in range(1, zones + 1)
        for day in range(days)
        for slot in range(SLOTS)
    ]
    return pl.DataFrame(
        rows, schema={"ts": pl.Datetime("us"), "zone_id": pl.Int16, "trips": pl.Int32}
    )


def _config() -> PipelineConfig:
    return PipelineConfig(
        data={"months": ["2026-01"], "base_url": "", "zone_lookup_url": "", "zone_max": 3},
        split=SplitConfig(validation_days=4, test_days=3),
        model=ModelConfig(params={"n_estimators": 20, "num_leaves": 8, "verbosity": -1}),
    )


@pytest.fixture
def predictor(tmp_path) -> Predictor:
    panel = _panel()
    panel_path = tmp_path / "panel.parquet"
    panel.write_parquet(panel_path)

    matrix = build_matrix(panel)
    result = train(matrix, _config(), artefact_root=tmp_path / "staging", version="v-test")

    registry = ModelRegistry(tmp_path / "models")
    registry.register(
        version="v-test",
        artefact_dir=result.artefact_dir,
        metrics=result.metrics,
        metadata=__import__("json").loads((result.artefact_dir / "metadata.json").read_text()),
    )
    registry.set_alias("production", "v-test")
    return Predictor(tmp_path / "models", panel_path)


def test_missing_model_raises(tmp_path) -> None:
    panel_path = tmp_path / "panel.parquet"
    _panel().write_parquet(panel_path)
    with pytest.raises(NotReadyError):
        Predictor(tmp_path / "empty", panel_path)


def test_missing_panel_raises(tmp_path) -> None:
    with pytest.raises(NotReadyError):
        Predictor(tmp_path / "models", tmp_path / "nope.parquet")


def test_model_info_carries_the_metrics(predictor: Predictor) -> None:
    info = predictor.model_info()
    assert info.version == "v-test"
    assert info.alias == "production"
    assert "wape" in info.metrics
    assert info.features


def test_forecast_returns_a_dense_day(predictor: Predictor) -> None:
    points = predictor.forecast(zone_id=2, slots=48)
    assert len(points) == 48
    assert all(point.zone_id == 2 for point in points)
    assert points[0].ts < points[-1].ts
    assert all(point.trips_p50 >= 0 for point in points)
    assert points[0].trips_p10 is not None
    assert points[0].trips_p10 <= points[0].trips_p50 <= points[0].trips_p90  # type: ignore[operator]


def test_forecast_can_drop_the_band(predictor: Predictor) -> None:
    points = predictor.forecast(zone_id=1, slots=2, quantiles=False)
    assert points[0].trips_p10 is None
    assert points[0].trips_p90 is None


def test_forecast_honours_a_start_inside_the_window(predictor: Predictor) -> None:
    start = predictor.forecast_start + timedelta(hours=6)
    points = predictor.forecast(zone_id=1, slots=2, start=start)
    assert points[0].ts == start


def test_forecast_rejects_a_start_outside_the_window(predictor: Predictor) -> None:
    with pytest.raises(ValueError):
        predictor.forecast(zone_id=1, start=predictor.forecast_end + timedelta(hours=1))
    with pytest.raises(ValueError):
        predictor.forecast(zone_id=1, start=predictor.data_through.replace(year=2020))


def test_forecast_unknown_zone_is_a_lookup_error(predictor: Predictor) -> None:
    with pytest.raises(LookupError):
        predictor.forecast(zone_id=999)


def test_batch_covers_every_zone(predictor: Predictor) -> None:
    forecasts = predictor.forecast_many(slots=2)
    assert len(forecasts) == 3
    assert all(len(points) == 2 for points in forecasts.values())


def test_batch_cap_is_enforced(predictor: Predictor) -> None:
    with pytest.raises(ValueError):
        predictor.forecast_many(zone_ids=list(range(1, 300)))


def test_replay_aligns_actuals_and_features(predictor: Predictor) -> None:
    window = predictor.replay(days=3)
    assert len(window.y) == len(window.y_hat) == len(window.ts) == len(window.zone)
    assert set(window.features) == {"hour", "lag_1d", "lag_7d", "roll_7d_slot", "roll_7d_zone"}
    assert window.y.min() >= 0
