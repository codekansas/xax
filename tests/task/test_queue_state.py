"""Tests for queue state persistence and process metadata."""

import json
from pathlib import Path

import pytest

from xax.core.conf import _load_user_config_cached
from xax.utils.launcher.queue_state import (
    claim_next_job,
    clear_job_process_tracking,
    enqueue_job,
    finish_running_job,
    get_job,
    get_queue_paths,
    get_running_job,
    read_queue_state,
    set_running_job_children,
    set_running_job_pid,
)


def _configure_user_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()


def _enqueue_dummy_job(tmp_path: Path) -> str:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "code"
    config_path = run_dir / "queued_config.yaml"
    observer_log_path = run_dir / "queue_observer.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("test: true\n", encoding="utf-8")
    stage_dir.mkdir(parents=True, exist_ok=True)
    return enqueue_job(
        task_key="tests.task.test_queue_state.DummyTask",
        launcher="multi",
        run_dir=run_dir,
        stage_dir=stage_dir,
        config_path=config_path,
        observer_log_path=observer_log_path,
    )


def test_queue_state_normalizes_missing_process_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    paths = get_queue_paths()
    paths.state_path.parent.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "version": 1,
        "next_job_idx": 2,
        "queue": ["job-0000001"],
        "running_job_id": None,
        "jobs": {
            "job-0000001": {
                "job_id": "job-0000001",
                "task_key": "tests.task.test_queue_state.DummyTask",
                "launcher": "multi",
                "status": "queued",
                "run_dir": str(tmp_path / "run"),
                "stage_dir": str(tmp_path / "run" / "code"),
                "config_path": str(tmp_path / "run" / "queued_config.yaml"),
                "observer_log_path": str(tmp_path / "run" / "queue_observer.log"),
                "enqueued_at": 1.0,
                "started_at": None,
                "ended_at": None,
                "pid": None,
                "return_code": None,
                "error": None,
            }
        },
    }
    paths.state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    state = read_queue_state()
    job = state["jobs"]["job-0000001"]
    assert job["process_group_id"] is None
    assert job["child_pids"] == []
    assert job["oom_detected"] is False


def test_process_tracking_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    job_id = _enqueue_dummy_job(tmp_path)
    claimed = claim_next_job()
    assert claimed is not None
    assert claimed["job_id"] == job_id

    set_running_job_pid(job_id, pid=12345, process_group_id=54321)
    set_running_job_children(job_id, [12346, 12347, 12346])

    running_job = get_running_job()
    assert running_job is not None
    assert running_job["pid"] == 12345
    assert running_job["process_group_id"] == 54321
    assert running_job["child_pids"] == [12346, 12347]

    clear_job_process_tracking(job_id)
    updated_job = get_job(job_id)
    assert updated_job is not None
    assert updated_job["pid"] is None
    assert updated_job["process_group_id"] is None
    assert updated_job["child_pids"] == []


def test_finish_running_job_marks_oom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    job_id = _enqueue_dummy_job(tmp_path)
    claimed = claim_next_job()
    assert claimed is not None

    set_running_job_pid(job_id, pid=12345, process_group_id=12345)
    finished_job = finish_running_job(
        job_id,
        return_code=1,
        error="CUDA OOM detected: out of memory",
        oom_detected=True,
    )
    assert finished_job["status"] == "failed"
    assert finished_job["oom_detected"] is True
    assert finished_job["process_group_id"] is None
    assert finished_job["child_pids"] == []
