"""Tests for experiment-monitor CLI flows."""

import json
from pathlib import Path

import pytest

from xax.cli.experiment import main
from xax.core.conf import _load_user_config_cached


@pytest.fixture(autouse=True)
def _configure_user_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()


def test_start_experiment_session_new(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["session", "--experiments-dir", str(tmp_path), "--new", "--name", "exp-a"])

    assert system_exit.value.code == 0
    assert (tmp_path / "exp-a" / "experiment_log.csv").exists()
    assert (tmp_path / "exp-a" / "experiment_report.md").exists()


def test_start_experiment_session_resume_requires_existing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["session", "--experiments-dir", str(tmp_path), "--resume", "--name", "missing-exp"])

    assert system_exit.value.code == 1


def test_sync_queue_status_updates_log(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["session", "--experiments-dir", str(tmp_path), "--new", "--name", "exp-a"])
    assert system_exit.value.code == 0

    status_json_path = tmp_path / "queue_status.json"
    status_json_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "job-0000001",
                        "status": "completed",
                        "task_key": "tests.task.DummyTask",
                        "config_path": "/tmp/queued_config.yaml",
                        "run_dir": "/tmp/run_001",
                        "started_at": 1.0,
                        "ended_at": 2.0,
                        "return_code": 0,
                        "launcher": "single",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as sync_exit:
        main(
            [
                "sync-queue-status",
                "--status-json",
                str(status_json_path),
                "--experiments-dir",
                str(tmp_path),
                "--experiment-name",
                "exp-a",
            ]
        )
    assert sync_exit.value.code == 0

    log_text = (tmp_path / "exp-a" / "experiment_log.csv").read_text(encoding="utf-8")
    assert "job-0000001" in log_text
    assert "/tmp/run_001" in log_text
