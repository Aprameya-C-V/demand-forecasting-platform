"""Command line entry point: the same commands run locally and in GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from demand_forecast.config import PipelineConfig, Settings, load_config
from demand_forecast.data.aggregate import build_panel, write_panel
from demand_forecast.data.fetch import fetch_all
from demand_forecast.data.validate import DataValidationError, validate_panel
from demand_forecast.features import build_matrix
from demand_forecast.monitor import drift_report, metrics_report, monitor
from demand_forecast.registry import ModelRegistry
from demand_forecast.train import train


def _settings(config: str) -> tuple[Settings, PipelineConfig]:
    settings = load_config(config)
    settings.paths.ensure()
    return settings, settings.pipeline


def cmd_fetch(args: argparse.Namespace) -> int:
    settings, cfg = _settings(args.config)
    files = fetch_all(cfg.data, settings.paths.raw_dir)
    print(json.dumps({"files": [item.name for item in files]}, indent=2))
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    settings, cfg = _settings(args.config)
    panel = build_panel(settings.paths.raw_dir, cfg.data)
    report = validate_panel(panel, cfg.data)
    print(json.dumps(report.as_dict(), indent=2))
    if not report.passed:
        return 1
    write_panel(panel, settings.paths.panel_path)
    print(f"panel: {panel.height} rows -> {settings.paths.panel_path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    settings, cfg = _settings(args.config)
    panel = pl.read_parquet(settings.paths.panel_path)
    matrix = build_matrix(panel, slot_minutes=cfg.data.slot_minutes)
    staging = settings.paths.model_dir / "_staging"
    result = train(
        matrix,
        cfg,
        artefact_root=staging,
        version=args.version,
        config_hash=settings.config_hash,
    )

    registry = ModelRegistry(settings.paths.model_dir)
    champion = registry.resolve(cfg.registry.alias)
    promote, reason = registry.should_promote(
        result.metrics,
        champion.metrics if champion else None,
        min_improvement=cfg.registry.min_improvement,
    )
    registry.register(
        version=result.version,
        artefact_dir=result.artefact_dir,
        metrics=result.metrics,
        metadata=json.loads((result.artefact_dir / "metadata.json").read_text()),
    )
    shutil.rmtree(staging, ignore_errors=True)

    if promote or args.force_promote:
        registry.set_alias(cfg.registry.alias, result.version)

    summary = {
        "promoted": bool(promote or args.force_promote),
        "reason": reason if not args.force_promote else "forced",
        "champion_before": champion.version if champion else None,
        "challenger": result.as_dict(),
    }
    (settings.paths.report_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from demand_forecast.serve.predictor import Predictor

    settings, cfg = _settings(args.config)
    predictor = Predictor(settings.paths.model_dir, settings.paths.panel_path, alias=cfg.registry.alias)
    window = predictor.replay(args.days)
    metrics = metrics_report(window.ts, window.y, window.y_hat, lower=window.lower, upper=window.upper)
    drift = drift_report(
        predictor.reference,
        window.features,
        threshold=cfg.monitoring.psi_threshold,
        reference_window=f"train up to {window.ts.min()}",
        current_window=f"last {args.days} days",
    )
    report = monitor(
        model_version=predictor.version.version if predictor.version else "unknown",
        metrics=metrics,
        drift=drift,
        cfg=cfg.monitoring,
    )
    payload = report.model_dump(mode="json")
    settings.paths.report_dir.mkdir(parents=True, exist_ok=True)
    (settings.paths.report_dir / "monitor.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_dashboard_data(payload, predictor.model_info().model_dump(mode="json"))

    print(json.dumps({"passed": report.passed, "alerts": report.alerts}, indent=2))
    if not report.passed and not args.exit_zero:
        return 1
    return 0


def _write_dashboard_data(report: dict[str, object], model: dict[str, object]) -> None:
    dashboard_dir = Path(os.environ.get("DF_DASHBOARD_DIR", "docs/dashboard"))
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "report": report,
        "source": "monitor replay over realised demand",
    }
    (dashboard_dir / "data.json").write_text(json.dumps(payload, indent=2) + "\n")


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "demand_forecast.serve.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """fetch -> panel -> train -> monitor, the same sequence the workflows run."""
    for step in (cmd_fetch, cmd_panel, cmd_train):
        code = step(args)
        if code:
            return code
    return cmd_monitor(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/base.yaml", help="pipeline config file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", parents=[common], help="download the published TLC files").set_defaults(
        func=cmd_fetch
    )
    sub.add_parser("panel", parents=[common], help="aggregate and validate the zone-slot panel").set_defaults(
        func=cmd_panel
    )

    train_parser = sub.add_parser(
        "train", parents=[common], help="train, register and maybe promote a model"
    )
    train_parser.add_argument("--version", default=None)
    train_parser.add_argument("--force-promote", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    monitor_parser = sub.add_parser(
        "monitor", parents=[common], help="replay realised demand and check thresholds"
    )
    monitor_parser.add_argument("--days", type=int, default=14)
    monitor_parser.add_argument("--exit-zero", action="store_true", help="never fail the step")
    monitor_parser.set_defaults(func=cmd_monitor)

    serve_parser = sub.add_parser("serve", parents=[common], help="run the forecast API")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--log-level", default="info")
    serve_parser.set_defaults(func=cmd_serve)

    pipeline_parser = sub.add_parser(
        "pipeline", parents=[common], help="fetch, panel, train and monitor"
    )
    pipeline_parser.add_argument("--version", default=None)
    pipeline_parser.add_argument("--force-promote", action="store_true")
    pipeline_parser.add_argument("--days", type=int, default=14)
    pipeline_parser.add_argument("--exit-zero", action="store_true")
    pipeline_parser.set_defaults(func=cmd_pipeline)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DataValidationError as exc:
        print(json.dumps(exc.report.as_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
