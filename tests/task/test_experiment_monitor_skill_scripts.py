"""Tests for experiment-monitor helper scripts."""

import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path("xax/agents/skills/experiment-monitor/scripts")


def test_start_experiment_session_new(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(SCRIPT_ROOT / "start_experiment_session.py"),
        "--experiments-dir",
        str(tmp_path),
        "--new",
        "--name",
        "exp-a",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert (tmp_path / "exp-a" / "experiment_log.csv").exists()
    assert (tmp_path / "exp-a" / "experiment_report.md").exists()


def test_start_experiment_session_resume_requires_existing(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(SCRIPT_ROOT / "start_experiment_session.py"),
        "--experiments-dir",
        str(tmp_path),
        "--resume",
        "--name",
        "missing-exp",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode != 0
