"""Aggregation: raw trips become a dense zone x slot panel.

A row exists for every zone and every 30 minute slot in the window, with zero trips where the
TLC data has no pickups. Lags are only meaningful on a dense panel, so this step is not optional.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from demand_forecast.config import DataConfig

PANEL_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us"),
    "zone_id": pl.Int16(),
    "trips": pl.Int32(),
}


def month_bounds(month: str) -> tuple[datetime, datetime]:
    """Return the half open interval covered by ``YYYY-MM``."""
    year, month_number = (int(part) for part in month.split("-"))
    start = datetime(year, month_number, 1)
    days = calendar.monthrange(year, month_number)[1]
    return start, start + timedelta(days=days)


def slot_grid(
    start: datetime, end: datetime, *, slot_minutes: int, zone_min: int, zone_max: int
) -> pl.DataFrame:
    """Every (zone, slot) pair in the half open window ``[start, end)``."""
    if end <= start:
        raise ValueError("end must be after start")
    step = timedelta(minutes=slot_minutes)
    slots = pl.datetime_range(start, end - step, interval=f"{slot_minutes}m", eager=True)
    zones = pl.Series("zone_id", range(zone_min, zone_max + 1), dtype=pl.Int16)
    return pl.DataFrame({"ts": slots}).join(pl.DataFrame({"zone_id": zones}), how="cross")


def densify(counts: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Left join observed counts onto the grid and fill the gaps with zero."""
    return (
        grid.join(counts, on=["ts", "zone_id"], how="left")
        .with_columns(pl.col("trips").fill_null(0).cast(pl.Int32))
        .select(["ts", "zone_id", "trips"])
        .sort(["zone_id", "ts"])
    )


def aggregate_month(path: Path, month: str, cfg: DataConfig) -> pl.DataFrame:
    """Aggregate one monthly trip file into the dense panel."""
    start, end = month_bounds(month)
    counts = (
        pl.scan_parquet(path)
        .select(
            pl.col("tpep_pickup_datetime").alias("ts"),
            pl.col("PULocationID").cast(pl.Int16).alias("zone_id"),
        )
        .filter(
            pl.col("ts").is_between(start, end, closed="left")
            & pl.col("zone_id").is_between(cfg.zone_min, cfg.zone_max)
        )
        .with_columns(pl.col("ts").dt.truncate(f"{cfg.slot_minutes}m"))
        .group_by(["ts", "zone_id"])
        .len()
        .rename({"len": "trips"})
        .with_columns(pl.col("trips").cast(pl.Int32))
        .collect()
    )
    grid = slot_grid(
        start, end, slot_minutes=cfg.slot_minutes, zone_min=cfg.zone_min, zone_max=cfg.zone_max
    )
    return densify(counts, grid)


def build_panel(raw_dir: Path, cfg: DataConfig, *, months: list[str] | None = None) -> pl.DataFrame:
    """Aggregate every configured month and concatenate them into one panel."""
    from demand_forecast.data.fetch import trip_file_name

    selected = months or cfg.months
    frames = [aggregate_month(raw_dir / trip_file_name(month), month, cfg) for month in selected]
    panel = pl.concat(frames).sort(["zone_id", "ts"])
    return panel.with_columns(
        pl.col("ts").cast(PANEL_SCHEMA["ts"]),
        pl.col("zone_id").cast(PANEL_SCHEMA["zone_id"]),
        pl.col("trips").cast(PANEL_SCHEMA["trips"]),
    )


def write_panel(panel: pl.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(destination, compression="zstd", statistics=True)
    return destination
