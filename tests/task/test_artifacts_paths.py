"""Tests for artifact directory naming behavior."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from xax.core.conf import _load_user_config_cached
from xax.task.base import BaseTask
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin


@dataclass
class DummyArtifactsConfig(ArtifactsConfig):
    pass


class DummyArtifactsTask(ArtifactsMixin[DummyArtifactsConfig], BaseTask[DummyArtifactsConfig]):
    pass


def _configure_user_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()


def test_artifacts_uses_runs_dir_hierarchy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("RUNS_DIR", str(runs_root))
    monkeypatch.delenv("RUN_DIR", raising=False)
    _load_user_config_cached.cache_clear()

    first_task = DummyArtifactsTask(DummyArtifactsConfig())
    second_task = DummyArtifactsTask(DummyArtifactsConfig())

    first_run_dir = first_task.run_dir
    second_run_dir = second_task.run_dir

    assert first_run_dir == runs_root / "dummy_artifacts_task" / "run_000"
    assert second_run_dir == runs_root / "dummy_artifacts_task" / "run_001"
    assert first_run_dir.exists()
    assert second_run_dir.exists()
    assert first_task.exp_dir == first_run_dir


def test_artifacts_respects_fixed_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)

    fixed_run_dir = tmp_path / "fixed_run"
    task = DummyArtifactsTask(DummyArtifactsConfig(run_dir=str(fixed_run_dir)))

    assert task.run_dir == fixed_run_dir.resolve()
    assert task.exp_dir == fixed_run_dir.resolve()
    assert fixed_run_dir.exists()


def test_artifacts_exp_dir_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)

    legacy_exp_dir = tmp_path / "legacy_exp"
    task = DummyArtifactsTask(DummyArtifactsConfig(exp_dir=str(legacy_exp_dir)))

    assert task.run_dir == legacy_exp_dir.resolve()
    assert task.exp_dir == legacy_exp_dir.resolve()
    assert legacy_exp_dir.exists()
