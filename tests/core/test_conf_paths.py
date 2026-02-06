"""Tests for user config path resolution."""

from pathlib import Path

import pytest

from xax.core.conf import (
    _load_user_config_cached,
    get_experiments_dir,
    get_run_dir,
    get_runs_dir,
    get_user_global_dir,
    load_user_config,
    user_config_path,
)


def test_user_config_defaults_to_xax_home_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()

    expected_path = (tmp_path / ".xax" / "config.yml").resolve()
    assert user_config_path().resolve() == expected_path
    assert get_user_global_dir().resolve() == (tmp_path / ".xax").resolve()

    load_user_config()
    assert expected_path.exists()


def test_user_config_path_respects_xaxrc_path_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAXRC_PATH", str(tmp_path / "custom.yml"))
    monkeypatch.delenv("XAX_HOME", raising=False)
    _load_user_config_cached.cache_clear()

    assert user_config_path().resolve() == (tmp_path / "custom.yml").resolve()
    assert get_user_global_dir().resolve() == tmp_path.resolve()


def test_user_config_path_respects_xaxrc_path_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAXRC_PATH", str(tmp_path / "custom_dir"))
    monkeypatch.delenv("XAX_HOME", raising=False)
    _load_user_config_cached.cache_clear()

    assert user_config_path().resolve() == (tmp_path / "custom_dir" / "config.yml").resolve()
    assert get_user_global_dir().resolve() == (tmp_path / "custom_dir").resolve()


def test_get_runs_dir_prefers_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs_root"))
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "legacy_runs_root"))
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()

    expected_runs_dir = (tmp_path / "runs_root").resolve()
    assert get_runs_dir() == expected_runs_dir
    assert get_run_dir() == expected_runs_dir
    assert expected_runs_dir.exists()


def test_get_runs_dir_falls_back_to_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RUNS_DIR", raising=False)
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "legacy_runs_root"))
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()

    expected_runs_dir = (tmp_path / "legacy_runs_root").resolve()
    assert get_runs_dir() == expected_runs_dir
    assert expected_runs_dir.exists()


def test_get_experiments_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path / "experiments_root"))
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()

    experiments_dir = get_experiments_dir()
    assert experiments_dir is not None
    assert experiments_dir == (tmp_path / "experiments_root").resolve()
    assert experiments_dir.exists()
