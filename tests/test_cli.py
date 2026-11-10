from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from demand_forecast.cli import build_parser, main

SLOTS = 48


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
data:
  months: ["2026-03"]
  base_url: ""
  zone_lookup_url: ""
  slot_minutes: 30
  zone_min: 1
  zone_max: 2
split:
  validation_days: 4
  test_days: 3
model:
  objective: poisson
  quantiles: [0.1, 0.9]
  params:
    n_estimators: 15
    num_leaves: 8
    verbosity: -1
  early_stopping_rounds: 10
registry:
  min_improvement: 0.02
  alias: production
monitoring:
  wape_threshold: 5.0
  bias_threshold: 1.0
  psi_threshold: 100.0
  min_rows: 10
"""
    )
    return config


def _write_raw(tmp_path: Path) -> None:
    """A full synthetic month with varying counts, so the model cannot fit it exactly.

    The panel is zero filled to the month, so a fixture that stops early would leave the test
    window empty and every metric would be a degenerate zero.
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 3, 1)
    rng = random.Random(7)
    timestamps = []
    zones = []
    for day in range(31):
        for slot in range(SLOTS):
            for zone in (1, 2):
                hour = slot // 2
                base = 3 if 7 <= hour <= 20 else 1
                count = max(0, base + rng.choice([-1, 0, 1]) + (zone % 2))
                for _ in range(count):
                    timestamps.append(start + timedelta(days=day, minutes=30 * slot))
                    zones.append(zone)
    pl.DataFrame({"tpep_pickup_datetime": timestamps, "PULocationID": zones}).write_parquet(
        raw / "yellow_tripdata_2026-03.parquet"
    )


def test_parser_wires_every_command() -> None:
    parser = build_parser()
    for argv, expected in [
        (["fetch"], "fetch"),
        (["panel"], "panel"),
        (["train"], "train"),
        (["monitor"], "monitor"),
        (["serve"], "serve"),
        (["pipeline"], "pipeline"),
    ]:
        args = parser.parse_args(argv)
        assert args.command == expected
        assert args.config == "configs/base.yaml"
        assert callable(args.func)


def test_panel_train_monitor_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DF_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DF_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DF_DASHBOARD_DIR", str(tmp_path / "dashboard"))

    config = _write_config(tmp_path)
    _write_raw(tmp_path)

    assert main(["panel", "--config", str(config)]) == 0
    assert (tmp_path / "data" / "processed" / "panel.parquet").exists()

    assert main(["train", "--config", str(config)]) == 0
    summary = json.loads((tmp_path / "reports" / "train_summary.json").read_text())
    assert summary["promoted"] is True
    assert summary["reason"] == "no champion registered"
    assert (tmp_path / "models" / "aliases.json").exists()

    assert main(["monitor", "--config", str(config), "--days", "3"]) == 0
    report = json.loads((tmp_path / "reports" / "monitor.json").read_text())
    assert report["passed"] is True
    dashboard = json.loads((tmp_path / "dashboard" / "data.json").read_text())
    assert dashboard["series"]
    assert dashboard["zones"]


def test_training_twice_keeps_the_champion_when_the_challenger_is_not_better(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DF_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DF_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DF_DASHBOARD_DIR", str(tmp_path / "dashboard"))

    config = _write_config(tmp_path)
    _write_raw(tmp_path)
    main(["panel", "--config", str(config)])
    main(["train", "--config", str(config)])

    first = json.loads((tmp_path / "models" / "aliases.json").read_text())["production"]
    main(["train", "--config", str(config)])
    summary = json.loads((tmp_path / "reports" / "train_summary.json").read_text())
    second = json.loads((tmp_path / "models" / "aliases.json").read_text())["production"]

    # an identical retrain should not beat the 2% gate, so the alias must not move
    assert summary["promoted"] is False
    assert second == first
