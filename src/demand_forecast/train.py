"""Training: time based split, LightGBM point and quantile models, artefacts, metadata."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from demand_forecast.config import PipelineConfig
from demand_forecast.evaluate import summarize
from demand_forecast.features import FeatureMatrix

DRIFT_FEATURES = ["hour", "lag_1d", "lag_7d", "roll_7d_slot", "roll_7d_zone"]
PSI_BINS = 10


@dataclass
class TrainResult:
    version: str
    artefact_dir: Path
    metrics: dict[str, float]
    quantile_metrics: dict[str, dict[str, float]]
    data_window: dict[str, str]
    train_rows: int
    validation_rows: int
    test_rows: int
    best_iterations: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metrics": self.metrics,
            "quantile_metrics": self.quantile_metrics,
            "data_window": self.data_window,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "best_iterations": self.best_iterations,
        }


def _version_tag(config_hash: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{config_hash}"


def split_masks(
    ts: np.ndarray, *, validation_days: int, test_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out the last ``test_days`` and the ``validation_days`` before them."""
    days = ts.astype("datetime64[D]")
    last_day = days.max()
    test_start = last_day - np.timedelta64(test_days - 1, "D")
    validation_start = test_start - np.timedelta64(validation_days, "D")
    train = days < validation_start
    validation = (days >= validation_start) & (days < test_start)
    test = days >= test_start
    if not (train.any() and validation.any() and test.any()):
        raise ValueError("split produced an empty part; the panel is too short for this config")
    return train, validation, test


def reference_profile(matrix: FeatureMatrix) -> dict[str, dict[str, list[float]]]:
    """Bin edges and reference proportions per drift feature, frozen at training time."""
    profile: dict[str, dict[str, list[float]]] = {}
    for feature in DRIFT_FEATURES:
        column = matrix.X[:, matrix.names.index(feature)]
        edges = np.unique(np.quantile(column, np.linspace(0, 1, PSI_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(column, bins=edges)
        proportions = (counts / counts.sum()).tolist()
        profile[feature] = {"edges": edges.tolist(), "proportions": proportions}
    return profile


def _git_sha() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha[:12]
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _train_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    cfg: PipelineConfig,
    matrix: FeatureMatrix,
    *,
    objective: str,
    extra_params: dict[str, Any] | None = None,
) -> lgb.Booster:
    """Train one booster with early stopping on the validation window."""
    rounds = int(cfg.model.params.get("n_estimators", 1000))
    params: dict[str, Any] = {
        "objective": objective,
        "metric": "l2",
        "seed": 7,
        "verbosity": -1,
        **{key: value for key, value in cfg.model.params.items() if key != "n_estimators"},
        **(extra_params or {}),
    }
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=matrix.categorical_indices,
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        X_validation,
        label=y_validation,
        reference=train_set,
        free_raw_data=False,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[validation_set],
        callbacks=[
            lgb.early_stopping(cfg.model.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )


def train(
    matrix: FeatureMatrix,
    cfg: PipelineConfig,
    *,
    artefact_root: Path,
    version: str | None = None,
    config_hash: str = "local",
) -> TrainResult:
    """Train the point model and the quantile band, then evaluate on the held out window."""
    history = ~np.isnan(matrix.y)
    if not history.any():
        raise ValueError("no labelled rows in the feature matrix")

    version = version or _version_tag(config_hash)
    train_mask, validation_mask, test_mask = split_masks(
        matrix.ts,
        validation_days=cfg.split.validation_days,
        test_days=cfg.split.test_days,
    )
    train_mask &= history
    validation_mask &= history
    test_mask &= history

    X, y = matrix.X, matrix.y
    point = _train_booster(
        X[train_mask],
        y[train_mask],
        X[validation_mask],
        y[validation_mask],
        cfg,
        matrix,
        objective=cfg.model.objective,
    )
    prediction = np.asarray(point.predict(X[test_mask]), dtype=float)

    artefact_dir = artefact_root / version
    artefact_dir.mkdir(parents=True, exist_ok=True)
    point.save_model(str(artefact_dir / "model_mean.txt"))

    bands: dict[str, np.ndarray] = {}
    quantile_metrics: dict[str, dict[str, float]] = {}
    for quantile in cfg.model.quantiles:
        model = _train_booster(
            X[train_mask],
            y[train_mask],
            X[validation_mask],
            y[validation_mask],
            cfg,
            matrix,
            objective="quantile",
            extra_params={"alpha": quantile, "metric": "quantile"},
        )
        name = f"model_q{int(quantile * 100):02d}"
        model.save_model(str(artefact_dir / f"{name}.txt"))
        bands[name] = np.asarray(model.predict(X[test_mask]), dtype=float)
        quantile_metrics[name] = {"alpha": quantile, "best_iteration": int(model.best_iteration or 0)}

    lower = bands.get("model_q10")
    upper = bands.get("model_q90")
    metrics = summarize(y[test_mask], prediction, lower=lower, upper=upper)

    test_ts = matrix.ts[test_mask]
    data_window = {
        "train_start": str(matrix.ts[train_mask].min()),
        "train_end": str(matrix.ts[validation_mask].min()),
        "validation_start": str(matrix.ts[validation_mask].min()),
        "validation_end": str(test_ts.min()),
        "test_start": str(test_ts.min()),
        "test_end": str(test_ts.max()),
    }

    importance = point.feature_importance(importance_type="gain")
    total = float(importance.sum()) or 1.0
    feature_importance = {
        name: round(float(gain) / total, 4) for name, gain in zip(matrix.names, importance, strict=True)
    }

    metadata: dict[str, Any] = {
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": matrix.names,
        "categorical_features": [matrix.names[i] for i in matrix.categorical_indices],
        "params": cfg.model.params,
        "objective": cfg.model.objective,
        "quantiles": cfg.model.quantiles,
        "data_window": data_window,
        "rows": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "metrics": metrics,
        "quantile_metrics": quantile_metrics,
        "feature_importance": feature_importance,
        "best_iterations": {
            "model_mean": int(point.best_iteration or 0),
            **{name: int(model["best_iteration"]) for name, model in quantile_metrics.items()},
        },
        "last_ts": str(matrix.last_ts),
        "git_sha": _git_sha(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
        },
        "config_hash": config_hash,
    }
    (artefact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (artefact_dir / "reference.json").write_text(
        json.dumps(reference_profile(_subset(matrix, train_mask)), indent=2) + "\n"
    )

    return TrainResult(
        version=version,
        artefact_dir=artefact_dir,
        metrics=metrics,
        quantile_metrics=quantile_metrics,
        data_window=data_window,
        train_rows=int(train_mask.sum()),
        validation_rows=int(validation_mask.sum()),
        test_rows=int(test_mask.sum()),
        best_iterations={"model_mean": int(point.best_iteration or 0)},
    )


def _subset(matrix: FeatureMatrix, mask: np.ndarray) -> FeatureMatrix:
    return FeatureMatrix(
        X=matrix.X[mask],
        y=matrix.y[mask],
        ts=matrix.ts[mask],
        zone=matrix.zone[mask],
        names=matrix.names,
        slots_per_day=matrix.slots_per_day,
        zone_ids=matrix.zone_ids,
        last_ts=matrix.last_ts,
    )
