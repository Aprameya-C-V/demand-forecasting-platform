from __future__ import annotations

import json

import pytest

from demand_forecast.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "models")


def _artefacts(tmp_path, name: str = "v1", payload: str = "model") -> object:
    directory = tmp_path / f"artefacts-{name}"
    directory.mkdir()
    (directory / "model_mean.txt").write_text(payload)
    (directory / "metadata.json").write_text(json.dumps({"trained_at": "2026-01-01T00:00:00Z"}))
    return directory


def test_register_and_resolve(registry, tmp_path) -> None:
    registry.register(
        version="v1",
        artefact_dir=_artefacts(tmp_path),
        metrics={"wape": 0.2},
        metadata={"trained_at": "2026-01-01T00:00:00Z"},
    )
    assert registry.resolve("production") is None
    registry.set_alias("production", "v1")
    resolved = registry.resolve("production")
    assert resolved is not None
    assert resolved.version == "v1"
    assert resolved.metrics["wape"] == 0.2
    assert (resolved.path / "model_mean.txt").read_text() == "model"


def test_set_alias_rejects_unknown_version(registry) -> None:
    with pytest.raises(KeyError):
        registry.set_alias("production", "missing")


def test_promotion_without_a_champion(registry) -> None:
    promoted, reason = registry.should_promote({"wape": 0.2}, None, min_improvement=0.02)
    assert promoted
    assert "no champion" in reason


def test_promotion_needs_the_minimum_improvement(registry) -> None:
    champion = {"wape": 0.20}
    assert registry.should_promote({"wape": 0.19}, champion, min_improvement=0.02)[0]
    promoted, reason = registry.should_promote({"wape": 0.199}, champion, min_improvement=0.02)
    assert not promoted
    assert "below the 2% gate" in reason


def test_a_worse_challenger_never_promotes(registry) -> None:
    promoted, _ = registry.should_promote({"wape": 0.30}, {"wape": 0.20}, min_improvement=0.02)
    assert not promoted


def test_missing_metrics_fall_back_to_promote(registry) -> None:
    promoted, reason = registry.should_promote({"mae": 1.0}, {"wape": 0.2}, min_improvement=0.02)
    assert promoted
    assert "no comparable metric" in reason
