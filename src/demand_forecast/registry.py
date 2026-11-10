"""Model registry: versioned artefacts on disk, aliases, and a champion/challenger gate."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALIASES_FILE = "aliases.json"


@dataclass(frozen=True)
class ModelVersion:
    version: str
    path: Path
    metrics: dict[str, float]
    trained_at: str
    metadata: dict[str, Any]


class ModelRegistry:
    """A directory of versions plus a small JSON index.

    Layout::

        models/
          registry.json
          aliases.json
          v/<version>/model_mean.txt
          v/<version>/model_p10.txt
          v/<version>/model_p90.txt
          v/<version>/metadata.json
          v/<version>/reference.json
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.root / "registry.json"

    @property
    def aliases_path(self) -> Path:
        return self.root / ALIASES_FILE

    def versions_dir(self, version: str) -> Path:
        return self.root / "v" / version

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"versions": {}}
        payload: dict[str, Any] = json.loads(self.index_path.read_text())
        return payload

    def _write_index(self, payload: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def register(
        self,
        *,
        version: str,
        artefact_dir: Path,
        metrics: dict[str, float],
        metadata: dict[str, Any],
    ) -> ModelVersion:
        """Copy a trained artefact directory into the registry under ``version``."""
        destination = self.versions_dir(version)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(artefact_dir, destination)
        payload = self._read_index()
        payload["versions"][version] = {
            "version": version,
            "metrics": metrics,
            "trained_at": metadata.get("trained_at", datetime.now(UTC).isoformat()),
            "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "path": str(destination.relative_to(self.root)),
        }
        self._write_index(payload)
        return ModelVersion(
            version=version,
            path=destination,
            metrics=metrics,
            trained_at=str(payload["versions"][version]["trained_at"]),
            metadata=metadata,
        )

    def aliases(self) -> dict[str, str]:
        if not self.aliases_path.exists():
            return {}
        return dict(json.loads(self.aliases_path.read_text()))

    def set_alias(self, alias: str, version: str) -> None:
        if not self.versions_dir(version).exists():
            raise KeyError(f"version {version} is not registered")
        aliases = self.aliases()
        aliases[alias] = version
        self.aliases_path.write_text(json.dumps(aliases, indent=2, sort_keys=True) + "\n")

    def resolve(self, alias: str) -> ModelVersion | None:
        version = self.aliases().get(alias)
        if version is None:
            return None
        entry = self._read_index()["versions"].get(version)
        if entry is None:
            return None
        metadata_path = self.versions_dir(version) / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return ModelVersion(
            version=version,
            path=self.versions_dir(version),
            metrics=dict(entry.get("metrics", {})),
            trained_at=str(entry.get("trained_at")),
            metadata=metadata,
        )

    def should_promote(
        self,
        challenger: dict[str, float],
        champion: dict[str, float] | None,
        *,
        min_improvement: float,
        metric: str = "wape",
    ) -> tuple[bool, str]:
        """A challenger wins only if it is better by at least ``min_improvement`` relative."""
        if champion is None:
            return True, "no champion registered"
        challenger_score = challenger.get(metric)
        champion_score = champion.get(metric)
        if challenger_score is None or champion_score is None or champion_score == 0:
            return True, "champion has no comparable metric"
        relative = (champion_score - challenger_score) / champion_score
        if relative >= min_improvement:
            return True, f"{metric} improved by {relative:.1%}"
        return False, f"{metric} improvement {relative:.1%} below the {min_improvement:.0%} gate"
