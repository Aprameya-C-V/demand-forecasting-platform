"""Feature engineering on a dense zone x day x slot cube.

The panel is dense and sorted by ``(zone_id, ts)``, so it reshapes exactly into a cube. Lags are
then shifts along the day axis, which keeps training and serving on one code path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

FEATURES: list[str] = [
    "zone_id",
    "hour",
    "dow",
    "slot",
    "is_weekend",
    "lag_1d",
    "lag_7d",
    "roll_7d_slot",
    "roll_7d_zone",
]
CATEGORICAL_FEATURES: list[str] = ["zone_id"]
LOOKBACK_DAYS = 7


@dataclass
class FeatureMatrix:
    """Flat, model ready matrix plus the metadata needed to serve and to explain it."""

    X: np.ndarray
    y: np.ndarray
    ts: np.ndarray
    zone: np.ndarray
    names: list[str]
    slots_per_day: int
    zone_ids: np.ndarray
    last_ts: datetime

    @property
    def categorical_indices(self) -> list[int]:
        return [self.names.index(name) for name in CATEGORICAL_FEATURES]

    @property
    def future_mask(self) -> np.ndarray:
        return np.isnan(self.y)

    def as_frame(self) -> pl.DataFrame:
        frame = pl.DataFrame({name: self.X[:, index] for index, name in enumerate(self.names)})
        return frame.with_columns(
            pl.Series("y", self.y),
            pl.Series("ts", self.ts),
            pl.Series("zone_id", self.zone.astype(np.int16)),
        )


def _lag(cube: np.ndarray, days: int) -> np.ndarray:
    """Shift the cube forward by ``days`` along the day axis, padding with NaN."""
    out = np.full_like(cube, np.nan, dtype=float)
    if days < cube.shape[1]:
        out[:, days:, :] = cube[:, :-days, :]
    return out


def _rolling(cube: np.ndarray, window: int) -> np.ndarray:
    """Mean of the previous ``window`` days, excluding the current day. NaN until available."""
    stacked = np.stack([_lag(cube, offset) for offset in range(1, window + 1)], axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(stacked, axis=0)
    result[np.isnan(stacked).all(axis=0)] = np.nan
    return result


def weekday_of(days: np.ndarray) -> np.ndarray:
    """Monday=0 weekday for an array of ``datetime64[D]`` values."""
    epoch_days = days.astype("datetime64[D]").astype(int)
    return (epoch_days + 3) % 7


def build_matrix(
    panel: pl.DataFrame,
    *,
    slot_minutes: int = 30,
    horizon_days: int = 0,
) -> FeatureMatrix:
    """Turn the panel into a feature matrix, optionally appending unseen future days.

    With ``horizon_days > 0`` the future rows carry a NaN target: they are the slots to forecast,
    and their features are built from real history. Callers split them with ``future_mask``.
    """
    if panel.height == 0:
        raise ValueError("panel is empty")

    zone_ids = np.sort(panel["zone_id"].unique().to_numpy())
    n_zones = len(zone_ids)
    if panel.height % n_zones:
        raise ValueError("panel is not rectangular: rows do not divide by the zone count")

    slots_per_day = 24 * 60 // slot_minutes
    rows_per_zone = panel.height // n_zones
    if rows_per_zone % slots_per_day:
        raise ValueError("panel does not cover whole days")
    n_days = rows_per_zone // slots_per_day

    trips = panel["trips"].to_numpy().reshape(n_zones, n_days, slots_per_day).astype(float)
    day_ts = np.unique(panel["ts"].to_numpy())[::slots_per_day]
    last_ts = panel["ts"].max()

    if horizon_days:
        last_day = np.datetime64(str(last_ts), "D")
        future = np.array([last_day + np.timedelta64(day + 1, "D") for day in range(horizon_days)])
        trips = np.concatenate(
            [trips, np.full((n_zones, horizon_days, slots_per_day), np.nan)], axis=1
        )
        day_ts = np.concatenate([day_ts, future])
    n_days = trips.shape[1]

    hour = (np.arange(slots_per_day) * slot_minutes // 60).astype(float)
    dow = weekday_of(day_ts).astype(float)
    is_weekend = (dow >= 5).astype(float)

    lag_1d = _lag(trips, 1)
    lag_7d = _lag(trips, LOOKBACK_DAYS)
    roll_7d_slot = _rolling(trips, LOOKBACK_DAYS)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        zone_level = np.nanmean(roll_7d_slot, axis=2, keepdims=True)
    roll_7d_zone = np.broadcast_to(zone_level, trips.shape)

    shape = trips.shape

    def tile(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values.astype(float), shape)

    columns = {
        "zone_id": tile(zone_ids.reshape(-1, 1, 1)),
        "hour": tile(hour.reshape(1, 1, -1)),
        "dow": tile(dow.reshape(1, -1, 1)),
        "slot": tile(np.arange(slots_per_day).reshape(1, 1, -1)),
        "is_weekend": tile(is_weekend.reshape(1, -1, 1)),
        "lag_1d": lag_1d,
        "lag_7d": lag_7d,
        "roll_7d_slot": roll_7d_slot,
        "roll_7d_zone": roll_7d_zone,
    }
    X = np.stack([columns[name].reshape(-1) for name in FEATURES], axis=1)

    step = np.timedelta64(slot_minutes, "m")
    slot_offsets = np.tile(np.arange(slots_per_day), n_days) * step
    base = np.repeat(day_ts, slots_per_day) + slot_offsets
    ts = np.tile(base, n_zones)
    zone = np.repeat(zone_ids, n_days * slots_per_day)

    keep = ~np.isnan(X).any(axis=1)

    return FeatureMatrix(
        X=X[keep],
        y=trips.reshape(-1)[keep],
        ts=ts[keep],
        zone=zone[keep],
        names=list(FEATURES),
        slots_per_day=slots_per_day,
        zone_ids=zone_ids,
        last_ts=np.datetime64(str(last_ts)).astype("datetime64[us]").astype(datetime),
    )
