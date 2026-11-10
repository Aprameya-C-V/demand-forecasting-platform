from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from demand_forecast.features import FEATURES, build_matrix, weekday_of

SLOTS = 48


def _panel(days: int = 10, zones: int = 2) -> pl.DataFrame:
    start = datetime(2026, 1, 5)  # a Monday
    rows = []
    for zone in range(1, zones + 1):
        for day in range(days):
            for slot in range(SLOTS):
                rows.append(
                    {
                        "ts": start + timedelta(days=day, minutes=30 * slot),
                        "zone_id": zone,
                        "trips": zone * 100 + day + slot,
                    }
                )
    return pl.DataFrame(
        rows, schema={"ts": pl.Datetime("us"), "zone_id": pl.Int16, "trips": pl.Int32}
    )


def test_weekday_of() -> None:
    days = np.array(["2026-01-05", "2026-01-06", "2026-01-10"], dtype="datetime64[D]")
    assert list(weekday_of(days)) == [0, 1, 5]


def test_matrix_drops_the_lookback_window() -> None:
    matrix = build_matrix(_panel(days=10))
    assert matrix.names == FEATURES
    # the first seven days cannot have lag_7d, so they are dropped entirely
    assert matrix.X.shape[0] == 2 * 3 * SLOTS
    assert np.isnan(matrix.X).sum() == 0


def test_lags_are_taken_from_the_right_day() -> None:
    panel = _panel(days=10, zones=1)
    matrix = build_matrix(panel)
    index = matrix.names.index("lag_1d")
    row = matrix.as_frame().row(0, named=True)
    expected_previous_day = row["y"] - 1  # the synthetic series increases by 1 per day
    assert row["lag_1d"] == expected_previous_day
    assert matrix.X[0, index] == expected_previous_day
    lag_7d = matrix.X[:, matrix.names.index("lag_7d")]
    assert lag_7d[0] == row["y"] - 7


def test_future_rows_carry_no_target_and_use_real_history() -> None:
    matrix = build_matrix(_panel(days=10, zones=2), horizon_days=1)
    future = matrix.future_mask
    assert future.sum() == 2 * SLOTS
    assert np.isnan(matrix.y[future]).all()
    assert np.isnan(matrix.X[future]).sum() == 0
    # a future row still has a one day lag pointing at real demand
    lag_1d = matrix.X[future, matrix.names.index("lag_1d")]
    assert (lag_1d > 0).all()


def test_zone_id_is_the_categorical_feature() -> None:
    matrix = build_matrix(_panel())
    assert matrix.categorical_indices == [matrix.names.index("zone_id")]


def test_rejects_a_panel_that_is_not_whole_days() -> None:
    panel = _panel(days=3).head(100)
    try:
        build_matrix(panel)
    except ValueError as exc:
        assert "rectangular" in str(exc) or "whole days" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
