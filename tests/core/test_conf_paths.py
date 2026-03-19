"""Tests for user config path resolution."""

from pathlib import Path

import pytest

from xax.core.conf import (
    _load_user_config_cached,
    get_experiments_dir,
    get_runs_dir,
    get_user_global_dir,
    load_user_config,
    user_config_path,
)


def _write_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_text: str) -> Path:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    config_path = (tmp_path / ".xax" / "config.yml").resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")
    _load_user_config_cached.cache_clear()
    return config_path


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


def test_get_runs_dir_uses_config_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs_root = (tmp_path / "runs_root").resolve()
    _write_user_config(
        monkeypatch,
        tmp_path,
        f"""
directories:
  runs: {runs_root}
""".strip()
        + "\n",
    )

    expected_runs_dir = runs_root
    assert get_runs_dir() == expected_runs_dir
    assert expected_runs_dir.exists()


def test_get_runs_dir_returns_none_without_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()

    assert get_runs_dir() is None


def test_get_experiments_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    experiments_root = (tmp_path / "experiments_root").resolve()
    _write_user_config(
        monkeypatch,
        tmp_path,
        f"""
directories:
  experiments: {experiments_root}
""".strip()
        + "\n",
    )

    experiments_dir = get_experiments_dir()
    assert experiments_dir is not None
    assert experiments_dir == experiments_root
    assert experiments_dir.exists()


def test_runs_dir_env_does_not_override_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured_runs = (tmp_path / "configured_runs").resolve()
    env_runs = (tmp_path / "env_runs").resolve()
    _write_user_config(
        monkeypatch,
        tmp_path,
        f"""
directories:
  runs: {configured_runs}
""".strip()
        + "\n",
    )
    monkeypatch.setenv("RUNS_DIR", str(env_runs))
    _load_user_config_cached.cache_clear()

    assert get_runs_dir() == configured_runs
    assert not env_runs.exists()


def test_invalid_user_config_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_user_config(
        monkeypatch,
        tmp_path,
        """
directories:
  run: /tmp/legacy
""".strip()
        + "\n",
    )

    with pytest.raises(ValueError, match="Unknown config keys"):
        load_user_config()
