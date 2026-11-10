from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from demand_forecast.config import ModelConfig, PipelineConfig, SplitConfig
from demand_forecast.features import build_matrix
from demand_forecast.train import reference_profile, split_masks, train

SLOTS = 48


def _panel(days: int = 40, zones: int = 2) -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    rows = [
        {
            "ts": start + timedelta(days=day, minutes=30 * slot),
            "zone_id": zone,
            "trips": (slot % 12) + zone,
        }
        for zone in range(1, zones + 1)
        for day in range(days)
        for slot in range(SLOTS)
    ]
    return pl.DataFrame(rows, schema={"ts": pl.Datetime("us"), "zone_id": pl.Int16, "trips": pl.Int32})


def _config() -> PipelineConfig:
    return PipelineConfig(
        data={
            "months": ["2026-01"],
            "base_url": "",
            "zone_lookup_url": "",
            "zone_max": 2,
        },
        split=SplitConfig(validation_days=7, test_days=7),
        model=ModelConfig(params={"n_estimators": 25, "num_leaves": 8, "verbosity": -1}),
    )


def test_split_masks_are_ordered_and_disjoint() -> None:
    ts = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-01-21")).astype("datetime64[us]")
    train_mask, validation_mask, test_mask = split_masks(ts, validation_days=5, test_days=5)
    assert not (train_mask & validation_mask).any()
    assert not (validation_mask & test_mask).any()
    assert ts[train_mask].max() < ts[validation_mask].min()
    assert ts[validation_mask].max() < ts[test_mask].min()
    assert len(ts[test_mask]) == 5


def test_split_rejects_a_window_that_is_too_short() -> None:
    ts = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-01-06")).astype("datetime64[us]")
    try:
        split_masks(ts, validation_days=7, test_days=7)
    except ValueError as exc:
        assert "empty part" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_reference_profile_has_bins_and_proportions() -> None:
    matrix = build_matrix(_panel(days=20))
    profile = reference_profile(matrix)
    assert set(profile) == {"hour", "lag_1d", "lag_7d", "roll_7d_slot", "roll_7d_zone"}
    for stats in profile.values():
        assert len(stats["edges"]) == len(stats["proportions"]) + 1
        assert abs(sum(stats["proportions"]) - 1.0) < 1e-9


def test_train_registers_artefacts_and_metrics(tmp_path) -> None:
    matrix = build_matrix(_panel(days=40))
    result = train(matrix, _config(), artefact_root=tmp_path, version="test-version")

    assert result.version == "test-version"
    assert result.metrics["wape"] >= 0
    assert (tmp_path / "test-version" / "model_mean.txt").exists()
    assert (tmp_path / "test-version" / "metadata.json").exists()
    assert (tmp_path / "test-version" / "reference.json").exists()
    assert result.train_rows > result.test_rows > 0
