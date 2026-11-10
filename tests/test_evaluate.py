from __future__ import annotations

import numpy as np
import pytest

from demand_forecast.evaluate import bias, coverage, mae, pinball, rmse, summarize, wape


def test_point_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = np.array([1.0, 2.0, 3.0, 6.0])
    assert mae(y, y_hat) == pytest.approx(0.5)
    assert rmse(y, y_hat) == pytest.approx(1.0)
    assert wape(y, y_hat) == pytest.approx(2.0 / 10.0)


def test_bias_sign() -> None:
    y = np.array([10.0, 10.0])
    assert bias(y, np.array([11.0, 11.0])) == pytest.approx(0.1)
    assert bias(y, np.array([9.0, 9.0])) == pytest.approx(-0.1)


def test_wape_handles_zero_demand() -> None:
    assert wape(np.zeros(3), np.zeros(3)) == 0.0


def test_coverage_counts_inside_the_band() -> None:
    y = np.array([1.0, 5.0, 9.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([2.0, 4.0, 10.0])
    assert coverage(y, lower, upper) == pytest.approx(2 / 3)


def test_pinball_prefers_the_quantile() -> None:
    y = np.array([10.0, 10.0])
    good = pinball(y, np.array([10.0, 10.0]), 0.9)
    bad = pinball(y, np.array([2.0, 2.0]), 0.9)
    assert good == 0.0
    assert bad > good


def test_summarize_includes_interval_metrics_only_with_a_band() -> None:
    y = np.arange(10, dtype=float)
    without = summarize(y, y)
    with_band = summarize(y, y, lower=y - 1, upper=y + 1)
    assert "coverage_p10_p90" not in without
    assert with_band["coverage_p10_p90"] == 1.0
