"""Ingestion: download the published TLC files, verify them, record a manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from demand_forecast.config import DataConfig


@dataclass(frozen=True)
class FetchedFile:
    name: str
    url: str
    sha256: str
    bytes: int
    fetched_at: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, timeout: float = 120.0) -> FetchedFile:
    temp = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    if temp.stat().st_size < 1024:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"download too small, probably an error page: {url}")
    shutil.move(temp, destination)
    return FetchedFile(
        name=destination.name,
        url=url,
        sha256=sha256_of(destination),
        bytes=destination.stat().st_size,
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def trip_file_name(month: str) -> str:
    return f"yellow_tripdata_{month}.parquet"


def fetch_month(month: str, cfg: DataConfig, raw_dir: Path) -> FetchedFile:
    """Fetch one monthly trip file. Cached: an existing file is hashed, not re-downloaded."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / trip_file_name(month)
    url = f"{cfg.base_url}/{trip_file_name(month)}"
    if destination.exists() and destination.stat().st_size > 1024:
        return FetchedFile(
            name=destination.name,
            url=url,
            sha256=sha256_of(destination),
            bytes=destination.stat().st_size,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    return _download(url, destination)


def fetch_zone_lookup(cfg: DataConfig, raw_dir: Path) -> FetchedFile:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "taxi_zone_lookup.csv"
    if destination.exists() and destination.stat().st_size > 1024:
        return FetchedFile(
            name=destination.name,
            url=cfg.zone_lookup_url,
            sha256=sha256_of(destination),
            bytes=destination.stat().st_size,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    return _download(cfg.zone_lookup_url, destination)


def fetch_all(cfg: DataConfig, raw_dir: Path) -> list[FetchedFile]:
    """Fetch every configured month plus the zone lookup and write ``manifest.json``."""
    files = [fetch_month(month, cfg, raw_dir) for month in cfg.months]
    files.append(fetch_zone_lookup(cfg, raw_dir))
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "months": cfg.months,
        "files": [asdict(item) for item in files],
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return files
