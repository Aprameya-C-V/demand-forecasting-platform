from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from demand_forecast.config import DataConfig, MonitoringConfig
from demand_forecast.data.validate import validate_panel
from demand_forecast.monitor import classify, drift_report, metrics_report, monitor, psi

CFG = DataConfig(months=["2026-01"], base_url="", zone_lookup_url="", zone_max=2)


def _panel(days: int = 2, zones: int = 2) -> pl.DataFrame:
    rows = [
        {"ts": datetime(2026, 1, 5) + __import__("datetime").timedelta(minutes=30 * slot), "zone_id": zone, "trips": slot}
        for zone in range(1, zones + 1)
        for slot in range(days * 48)
    ]
    return pl.DataFrame(rows, schema={"ts": pl.Datetime("us"), "zone_id": pl.Int16, "trips": pl.Int32})


def test_dense_panel_passes() -> None:
    report = validate_panel(_panel(), CFG)
    assert report.passed, report.failures
    assert report.rows == 2 * 2 * 48
    assert report.gaps == 0


def test_missing_row_is_a_failure() -> None:
    panel = _panel()
    hole = panel.filter(~((pl.col("zone_id") == 1) & (pl.col("ts") == datetime(2026, 1, 5, 12, 0))))
    report = validate_panel(hole, CFG)
    assert not report.passed
    assert any("missing" in failure for failure in report.failures)


def test_missing_zone_is_a_failure() -> None:
    report = validate_panel(_panel().head(95), CFG)
    assert not report.passed
    assert any("zones" in failure for failure in report.failures)


def test_negative_and_duplicate_rows_fail() -> None:
    panel = _panel()
    extra = pl.DataFrame(
        {"ts": [panel["ts"][0]], "zone_id": [1], "trips": [-3]},
        schema={"ts": pl.Datetime("us"), "zone_id": pl.Int16, "trips": pl.Int32},
    )
    broken = pl.concat([panel, extra])
    report = validate_panel(broken, CFG)
    assert any("negative" in failure for failure in report.failures)


def test_psi_is_zero_for_identical_samples() -> None:
    values = np.linspace(0, 10, 1000)
    edges = np.linspace(0, 10, 11).tolist()
    counts, _ = np.histogram(values, bins=edges)
    proportions = (counts / counts.sum()).tolist()
    assert psi(edges, proportions, values) == pytest.approx(0.0, abs=1e-6)


def test_psi_grows_when_the_sample_shifts() -> None:
    reference = np.linspace(0, 10, 1000)
    edges = np.linspace(0, 10, 11).tolist()
    counts, _ = np.histogram(reference, bins=edges)
    proportions = (counts / counts.sum()).tolist()
    shifted = reference * 0.5 + 5.0  # compresses the sample into the upper half
    assert psi(edges, proportions, shifted) > 0.5


def test_classify_thresholds() -> None:
    assert classify(0.01, 0.2) == "stable"
    assert classify(0.15, 0.2) == "warning"
    assert classify(0.30, 0.2) == "drift"


def test_metrics_report_reports_by_hour() -> None:
    ts = np.array(
        [np.datetime64("2026-01-01T00:00"), np.datetime64("2026-01-01T00:30")], dtype="datetime64[us]"
    )
    report = metrics_report(ts, np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert report.rows == 2
    assert report.wape == pytest.approx(0.3333, abs=1e-4)
    assert set(report.by_hour) == {"0"}


def test_monitor_flags_a_regression() -> None:
    ts = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-01-01") + np.timedelta64(10, "h"), np.timedelta64(1, "h"))
    good = metrics_report(ts, np.ones(10), np.ones(10))
    drift = drift_report({}, {}, threshold=0.2, reference_window="a", current_window="b")
    clean = monitor(model_version="v1", metrics=good, drift=drift, cfg=MonitoringConfig(min_rows=1))
    assert clean.passed

    bad = metrics_report(ts, np.full(10, 10.0), np.ones(10))
    flagged = monitor(model_version="v1", metrics=bad, drift=drift, cfg=MonitoringConfig(min_rows=1))
    assert not flagged.passed
    assert any("WAPE" in alert for alert in flagged.alerts)
