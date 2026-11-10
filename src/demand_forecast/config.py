"""Typed configuration for the demand forecasting platform.

The pipeline reads one YAML file (``configs/base.yaml`` by default) and overrides paths from
``DF_*`` environment variables so the same code runs locally, in CI and in a container.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


class DataConfig(BaseModel):
    months: list[str] = Field(min_length=1)
    base_url: str
    zone_lookup_url: str
    slot_minutes: int = 30
    zone_min: int = 1
    zone_max: int = 263


class SplitConfig(BaseModel):
    validation_days: int = 21
    test_days: int = 14


class ModelConfig(BaseModel):
    objective: str = "poisson"
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.9])
    params: dict[str, Any] = Field(default_factory=dict)
    early_stopping_rounds: int = 50


class RegistryConfig(BaseModel):
    min_improvement: float = 0.02
    alias: str = "production"


class MonitoringConfig(BaseModel):
    wape_threshold: float = 0.38
    bias_threshold: float = 0.10
    psi_threshold: float = 0.20
    min_rows: int = 2000


class PipelineConfig(BaseModel):
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


class Paths(BaseSettings):
    """Filesystem layout. Override with ``DF_DATA_DIR``, ``DF_MODEL_DIR`` and ``DF_REPORT_DIR``."""

    model_config = SettingsConfigDict(env_prefix="DF_", extra="ignore")

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def panel_path(self) -> Path:
        return self.processed_dir / "panel.parquet"

    def ensure(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.model_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    pipeline: PipelineConfig
    paths: Paths = Field(default_factory=Paths)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.pipeline.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    """Load the pipeline config, applying ``DF_*`` environment overrides to the paths."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} is not a mapping")
    return Settings(pipeline=PipelineConfig.model_validate(raw), paths=Paths())
