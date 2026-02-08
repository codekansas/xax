"""High-value integration tests for experiment-monitor skill scripts."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "xax" / "agents" / "skills" / "experiment-monitor" / "scripts"


def _run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script_path = SCRIPTS_DIR / script_name
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_start_experiment_session_creates_templates(tmp_path: Path) -> None:
    result = _run_script(
        "start_experiment_session.py",
        "--experiments-dir",
        str(tmp_path),
        "--new",
        "--name",
        "exp-a",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["experiment_name"] == "exp-a"
    assert (tmp_path / "exp-a" / "experiment_log.csv").exists()
    assert (tmp_path / "exp-a" / "experiment_report.md").exists()


def test_experiment_scripts_sync_upsert_and_render_report(tmp_path: Path) -> None:
    start_result = _run_script(
        "start_experiment_session.py",
        "--experiments-dir",
        str(tmp_path),
        "--new",
        "--name",
        "exp-a",
    )
    assert start_result.returncode == 0, start_result.stderr

    status_json_path = tmp_path / "queue_status.json"
    status_json_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "job-0000001",
                        "status": "completed",
                        "task_key": "tests.task.DummyTask",
                        "config_path": "/tmp/config.yaml",
                        "run_dir": "/tmp/run_001",
                        "started_at": 1.0,
                        "ended_at": 2.0,
                        "return_code": 0,
                        "launcher": "queued",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sync_result = _run_script(
        "sync_queue_status.py",
        "--status-json",
        str(status_json_path),
        "--experiments-dir",
        str(tmp_path),
        "--experiment-name",
        "exp-a",
    )
    assert sync_result.returncode == 0, sync_result.stderr

    upsert_result = _run_script(
        "upsert_experiment_log.py",
        "--experiment-id",
        "job-0000001",
        "--experiments-dir",
        str(tmp_path),
        "--experiment-name",
        "exp-a",
        "--objective-metric",
        "val/loss",
        "--objective-mode",
        "min",
        "--objective-value",
        "0.42",
        "--next-action",
        "try larger hidden size",
    )
    assert upsert_result.returncode == 0, upsert_result.stderr

    report_result = _run_script(
        "render_experiment_report.py",
        "--experiments-dir",
        str(tmp_path),
        "--experiment-name",
        "exp-a",
    )
    assert report_result.returncode == 0, report_result.stderr

    log_text = (tmp_path / "exp-a" / "experiment_log.csv").read_text(encoding="utf-8")
    assert "job-0000001" in log_text
    assert "val/loss" in log_text
    assert "0.42" in log_text

    report_text = (tmp_path / "exp-a" / "experiment_report.md").read_text(encoding="utf-8")
    assert "Experiment Monitor Report" in report_text
    assert "job-0000001" in report_text
