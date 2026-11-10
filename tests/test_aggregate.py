from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from demand_forecast.data.aggregate import densify, month_bounds, slot_grid

ZONES = {"slot_minutes": 30, "zone_min": 1, "zone_max": 3}


def test_month_bounds_are_half_open() -> None:
    start, end = month_bounds("2026-02")
    assert start == datetime(2026, 2, 1)
    assert end == datetime(2026, 3, 1)


def test_slot_grid_is_dense() -> None:
    grid = slot_grid(datetime(2026, 1, 1), datetime(2026, 1, 2), **ZONES)
    assert grid.height == 3 * 48
    assert grid["ts"].n_unique() == 48
    assert grid["zone_id"].n_unique() == 3


def test_densify_fills_missing_slots_with_zero() -> None:
    grid = slot_grid(datetime(2026, 1, 1), datetime(2026, 1, 1, 2), **ZONES)
    counts = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 30)],
            "zone_id": [2],
            "trips": [7],
        },
        schema={"ts": pl.Datetime("us"), "zone_id": pl.Int16, "trips": pl.Int32},
    )
    panel = densify(counts, grid)
    assert panel.height == grid.height
    assert panel.filter(pl.col("trips") > 0).height == 1
    assert panel.filter(pl.col("trips") > 0)["trips"].item() == 7
    assert panel["trips"].sum() == 7


def test_aggregate_month_counts_real_pickups(tmp_path) -> None:
    from demand_forecast.config import DataConfig
    from demand_forecast.data.aggregate import aggregate_month

    timestamps = [datetime(2026, 3, 1, 0, 5), datetime(2026, 3, 1, 0, 20), datetime(2026, 3, 1, 0, 45)]
    frame = pl.DataFrame(
        {
            "tpep_pickup_datetime": timestamps,
            "PULocationID": [1, 1, 999],
            "fare_amount": [10.0, 11.0, 12.0],
        }
    )
    path = tmp_path / "yellow_tripdata_2026-03.parquet"
    frame.write_parquet(path)
    cfg = DataConfig(months=["2026-03"], base_url="", zone_lookup_url="", zone_max=2)
    panel = aggregate_month(path, "2026-03", cfg)

    assert panel.height == 2 * 31 * 48
    first_slot = panel.filter((pl.col("zone_id") == 1) & (pl.col("ts") == datetime(2026, 3, 1, 0, 0)))
    assert first_slot["trips"].item() == 2
    # the out of range zone is dropped, not counted anywhere
    assert panel["trips"].sum() == 2
    assert panel["ts"].max() == datetime(2026, 3, 31, 23, 30)


def test_aggregate_handles_the_month_boundary() -> None:
    start, end = month_bounds("2026-01")
    grid = slot_grid(start, end, **ZONES)
    assert grid["ts"].min() == start
    assert grid["ts"].max() == end - timedelta(minutes=30)
