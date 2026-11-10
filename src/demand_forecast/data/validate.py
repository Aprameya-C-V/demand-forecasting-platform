"""Data contracts: the pipeline refuses to train on a panel that fails these checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from demand_forecast.config import DataConfig


class DataValidationError(RuntimeError):
    """Raised when a panel does not satisfy the contract."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(report.failures))


@dataclass
class ValidationReport:
    rows: int
    zones: int
    slots: int
    start: datetime | None
    end: datetime | None
    nulls: int
    negatives: int
    duplicates: int
    gaps: int
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "zones": self.zones,
            "slots": self.slots,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "nulls": self.nulls,
            "negatives": self.negatives,
            "duplicates": self.duplicates,
            "gaps": self.gaps,
            "failures": self.failures,
            "passed": self.passed,
        }


def validate_panel(panel: pl.DataFrame, cfg: DataConfig) -> ValidationReport:
    """Check schema, uniqueness, non negativity and panel density.

    A gap is a (zone, slot) pair that should exist but does not, computed against the global
    slot count. Gaps mean lags would silently span a missing row, so they are a failure.
    """
    report = ValidationReport(
        rows=panel.height,
        zones=panel.select(pl.col("zone_id").n_unique()).item(),
        slots=panel.select(pl.col("ts").n_unique()).item(),
        start=panel.select(pl.col("ts").min()).item(),
        end=panel.select(pl.col("ts").max()).item(),
        nulls=panel.null_count().select(pl.sum_horizontal(pl.all())).item(),
        negatives=panel.filter(pl.col("trips") < 0).height,
        duplicates=panel.height - panel.unique(subset=["ts", "zone_id"]).height,
        gaps=0,
    )

    if set(panel.columns) != {"ts", "zone_id", "trips"}:
        report.failures.append(f"unexpected columns: {sorted(panel.columns)}")
    if report.rows == 0:
        report.failures.append("panel is empty")
    if report.nulls:
        report.failures.append(f"{report.nulls} null values")
    if report.negatives:
        report.failures.append(f"{report.negatives} negative trip counts")
    if report.duplicates:
        report.failures.append(f"{report.duplicates} duplicate (ts, zone_id) rows")
    if report.zones != cfg.zone_max - cfg.zone_min + 1:
        report.failures.append(f"expected {cfg.zone_max - cfg.zone_min + 1} zones, found {report.zones}")
    if report.rows and report.slots:
        expected = report.zones * report.slots
        report.gaps = expected - report.rows
        if report.gaps:
            report.failures.append(f"{report.gaps} missing (zone, slot) rows")
    return report
