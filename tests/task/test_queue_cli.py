"""Tests for queue CLI helpers."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from xax.cli.queue import (
    _resolve_existing_run_dir,
    _resolve_queue_gpu_devices,
    _service_unit_text,
    main,
)
from xax.core.conf import _load_user_config_cached
from xax.utils.cli_output import CliOutput
from xax.utils.launcher.queue_state import ObserverInfo, QueuedJob, QueueState, edit_queue_state, read_queue_state


@pytest.fixture(autouse=True)
def _configure_user_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()


def test_service_unit_has_robust_process_controls() -> None:
    unit_text = _service_unit_text("xax-queue-observer")
    assert "-m xax.cli.main queue _observer" in unit_text
    assert "KillMode=control-group" in unit_text
    assert "OOMPolicy=continue" in unit_text
    assert "Restart=always" in unit_text


def test_service_unit_supports_queue_gpu_reservation() -> None:
    unit_text = _service_unit_text("xax-queue-observer", queue_gpus="0,2")
    assert "Environment=XAX_QUEUE_GPUS=0,2" in unit_text
    assert "Environment=XAX_QUEUE_NUM_GPUS=" not in unit_text


def test_service_unit_supports_queue_gpu_count() -> None:
    unit_text = _service_unit_text("xax-queue-observer", queue_num_gpus=3)
    assert "Environment=XAX_QUEUE_NUM_GPUS=3" in unit_text
    assert "Environment=XAX_QUEUE_GPUS=" not in unit_text


def test_resolve_queue_gpu_devices_from_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_utils.shutil.which",
        lambda binary: "/usr/bin/nvidia-smi" if binary == "nvidia-smi" else None,
    )

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        assert not check
        assert capture_output
        assert text
        return subprocess.CompletedProcess(command, 0, stdout="0\n1\n2\n3\n", stderr="")

    monkeypatch.setattr("xax.utils.launcher.gpu_utils.subprocess.run", fake_run)

    assert _resolve_queue_gpu_devices(None, 2, use_env_defaults=False) == "0,1"


def test_resolve_queue_gpu_devices_rejects_conflicting_options() -> None:
    with pytest.raises(ValueError, match="Pass at most one of --queue-gpus and --queue-num-gpus"):
        _resolve_queue_gpu_devices("0,1", 2, use_env_defaults=False)


def test_queue_start_uses_systemd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service_dir = tmp_path / "systemd_user"
    service_dir.mkdir(parents=True, exist_ok=True)
    calls: list[list[str]] = []

    monkeypatch.setattr("xax.cli.queue.shutil.which", lambda _: "/usr/bin/systemctl")
    monkeypatch.setattr("xax.cli.queue._service_dir", lambda: service_dir)

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert not check
        assert capture_output
        assert text
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("xax.cli.queue.subprocess.run", fake_run)

    with pytest.raises(SystemExit) as system_exit:
        main(["start"])

    assert system_exit.value.code == 0
    assert (service_dir / "xax-queue-observer.service").exists()
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "start", "xax-queue-observer.service"],
    ]


def test_queue_start_fails_without_systemd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("xax.cli.queue.shutil.which", lambda _: None)

    with pytest.raises(SystemExit) as system_exit:
        main(["start"])

    assert system_exit.value.code == 1


def test_queue_top_level_help_lists_command_descriptions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["--help"])

    assert system_exit.value.code == 0
    output = capsys.readouterr().out
    assert "status" in output
    assert "wait" in output
    assert "Show observer status" in output
    assert "install-service" not in output
    assert "uninstall-service" not in output


def test_queue_command_help_uses_field_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["status", "--help"])

    assert system_exit.value.code == 0
    output = capsys.readouterr().out
    assert "Usage: xax queue status" in output
    assert "Emit queue state as JSON." in output


@pytest.mark.parametrize("command", ["systemd", "list", "cleanup", "install-service", "uninstall-service"])
def test_queue_unknown_removed_command(capsys: pytest.CaptureFixture[str], command: str) -> None:
    with pytest.raises(SystemExit) as system_exit:
        main([command])

    assert system_exit.value.code == 2
    captured = capsys.readouterr()
    assert "Unknown queue command" in captured.err
    assert command in captured.err
    assert "Usage: xax queue <command> [args]" in captured.out


def test_queue_stop_removes_unit_and_resets_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service_dir = tmp_path / "systemd_user"
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / "xax-queue-observer.service"
    service_path.write_text("[Unit]\nDescription=test\n", encoding="utf-8")

    calls: list[list[str]] = []

    monkeypatch.setattr("xax.cli.queue._service_dir", lambda: service_dir)
    monkeypatch.setattr("xax.cli.queue.shutil.which", lambda _: "/usr/bin/systemctl")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert not check
        assert capture_output
        assert text
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("xax.cli.queue.subprocess.run", fake_run)

    running_job = QueuedJob(
        job_id="job-0000009",
        task_key="tests.task.DummyTask",
        launcher="multi",
        python_executable=sys.executable,
        status="running",
        run_dir=str(tmp_path / "run"),
        stage_dir=str(tmp_path / "stage"),
        config_path=str(tmp_path / "cfg.yaml"),
        observer_log_path=str(tmp_path / "observer.log"),
        enqueued_at=1.0,
        started_at=2.0,
        ended_at=None,
        pid=None,
        process_group_id=None,
        child_pids=[],
        oom_detected=False,
        return_code=None,
        error=None,
    )
    with edit_queue_state() as state:
        state.jobs[running_job.job_id] = running_job
        state.running_job_id = running_job.job_id

    cleanup_called = {"value": False}

    def _fake_cleanup_stale_processes(_out: CliOutput) -> None:
        cleanup_called["value"] = True

    monkeypatch.setattr("xax.cli.queue._cleanup_stale_processes", _fake_cleanup_stale_processes)

    with pytest.raises(SystemExit) as system_exit:
        main(["stop"])

    assert system_exit.value.code == 0
    assert not service_path.exists()
    assert calls == [
        ["systemctl", "--user", "stop", "xax-queue-observer.service"],
        ["systemctl", "--user", "disable", "xax-queue-observer.service"],
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "reset-failed", "xax-queue-observer.service"],
    ]

    state_after = read_queue_state()
    assert state_after.running_job_id is None
    stopped_job = state_after.jobs["job-0000009"]
    assert stopped_job.status == "failed"
    assert stopped_job.error is not None
    assert "Queue stopped while job was running" in stopped_job.error
    assert cleanup_called["value"] is True


def test_resolve_existing_run_dir_falls_back_under_runs_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logs_root = (tmp_path / "logs").resolve()
    runs_root = (logs_root / "today").resolve()
    actual_run_dir = (runs_root / "queue_mnist_flow2" / "mnist_classification" / "run_000").resolve()
    actual_run_dir.mkdir(parents=True, exist_ok=True)

    stale_run_dir = logs_root / "queue_mnist_flow2" / "mnist_classification" / "run_000"
    monkeypatch.setattr("xax.cli.queue.xax.get_runs_dir", lambda: runs_root)

    assert _resolve_existing_run_dir(str(stale_run_dir)) == actual_run_dir


def test_resolve_existing_run_dir_falls_back_to_legacy_run_dir_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logs_root = (tmp_path / "logs").resolve()
    runs_root = (logs_root / "today").resolve()
    actual_run_dir = (runs_root / "queue_mnist_flow2" / "mnist_classification" / "run_000").resolve()
    actual_run_dir.mkdir(parents=True, exist_ok=True)

    stale_run_dir = logs_root / "queue_mnist_flow2" / "mnist_classification" / "run_000"
    monkeypatch.setattr("xax.cli.queue.xax.get_runs_dir", lambda: None)
    monkeypatch.setenv("RUN_DIR", str(runs_root))

    assert _resolve_existing_run_dir(str(stale_run_dir)) == actual_run_dir


def test_status_formats_recent_jobs_as_table_and_shows_observer_gpus(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job = QueuedJob(
        job_id="job-0000001",
        task_key="tests.task.DummyTask",
        launcher="single",
        python_executable=sys.executable,
        status="completed",
        run_dir="/tmp/run_001",
        stage_dir="/tmp/run_001/code",
        config_path="/tmp/run_001/config.yaml",
        observer_log_path="/tmp/run_001/queue_observer.log",
        enqueued_at=1.0,
        started_at=2.0,
        ended_at=3.0,
        pid=None,
        process_group_id=None,
        child_pids=[],
        oom_detected=False,
        return_code=0,
        error=None,
    )
    observer_info = ObserverInfo(
        pid=12345,
        hostname="test-host",
        started_at=1.0,
        updated_at=time.time(),
        status="idle",
    )
    state = QueueState(
        version=1,
        next_job_idx=2,
        queue=[],
        running_job_id=None,
        jobs={job.job_id: job},
    )

    monkeypatch.setattr("xax.cli.queue.xax.read_queue_state", lambda: state)
    monkeypatch.setattr("xax.cli.queue.xax.read_observer_info", lambda: observer_info)
    monkeypatch.setattr("xax.cli.queue.xax.list_jobs", lambda: [job])
    monkeypatch.setattr("xax.cli.queue.xax.is_observer_active", lambda **_kwargs: True)
    monkeypatch.setattr("xax.cli.queue._observer_queue_gpu_devices", lambda _info: "0,1")

    with pytest.raises(SystemExit) as system_exit:
        main(["status", "--recent", "1"])

    assert system_exit.value.code == 0
    output = capsys.readouterr().out
    assert "[queue]" not in output
    assert "Observer" in output
    assert "Running Job" in output
    assert "Queued Jobs" in output
    assert "Recent Jobs" in output
    assert any(line.startswith("┌") and "Observer" in line for line in output.splitlines())
    assert "│ Observer │" not in output
    assert "│ state" in output
    assert "│ idx" in output
    assert "queue_gpus" in output
    assert "0,1" in output
    assert "│ job_id" in output
    assert "│ status" in output
    assert "job-0000001" in output


def test_tail_defaults_to_running_job_when_job_id_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running_job = QueuedJob(
        job_id="job-0000002",
        task_key="tests.task.DummyTask",
        launcher="single",
        python_executable=sys.executable,
        status="running",
        run_dir="/tmp/run_002",
        stage_dir="/tmp/run_002/code",
        config_path="/tmp/run_002/config.yaml",
        observer_log_path="/tmp/run_002/queue_observer.log",
        enqueued_at=1.0,
        started_at=2.0,
        ended_at=None,
        pid=777,
        process_group_id=777,
        child_pids=[],
        oom_detected=False,
        return_code=None,
        error=None,
    )

    seen_path: Path | None = None
    seen_lines: int | None = None
    seen_follow: bool | None = None

    monkeypatch.setattr("xax.cli.queue.xax.get_running_job", lambda: running_job)
    monkeypatch.setattr("xax.cli.queue.xax.list_jobs", lambda: [])
    monkeypatch.setattr("xax.cli.queue.xax.get_job", lambda _job_id: None)
    monkeypatch.setattr("xax.cli.queue._resolve_log_file", lambda job, kind: Path(f"/tmp/{job.job_id}-{kind}.log"))

    def fake_tail_file(path: Path, lines: int, follow: bool) -> int:
        nonlocal seen_path, seen_lines, seen_follow
        seen_path = path
        seen_lines = lines
        seen_follow = follow
        return 0

    monkeypatch.setattr("xax.cli.queue._tail_file", fake_tail_file)

    with pytest.raises(SystemExit) as system_exit:
        main(["tail", "--kind", "task", "--lines", "7", "--follow"])

    assert system_exit.value.code == 0
    assert seen_path == Path("/tmp/job-0000002-task.log")
    assert seen_lines == 7
    assert seen_follow is True


def test_tail_without_job_id_uses_most_recent_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older_job = QueuedJob(
        job_id="job-0000002",
        task_key="tests.task.DummyTask",
        launcher="single",
        python_executable=sys.executable,
        status="completed",
        run_dir="/tmp/run_002",
        stage_dir="/tmp/run_002/code",
        config_path="/tmp/run_002/config.yaml",
        observer_log_path="/tmp/run_002/queue_observer.log",
        enqueued_at=1.0,
        started_at=2.0,
        ended_at=3.0,
        pid=None,
        process_group_id=None,
        child_pids=[],
        oom_detected=False,
        return_code=0,
        error=None,
    )
    newest_job = QueuedJob(
        job_id="job-0000003",
        task_key="tests.task.DummyTask",
        launcher="single",
        python_executable=sys.executable,
        status="queued",
        run_dir="/tmp/run_003",
        stage_dir="/tmp/run_003/code",
        config_path="/tmp/run_003/config.yaml",
        observer_log_path="/tmp/run_003/queue_observer.log",
        enqueued_at=4.0,
        started_at=None,
        ended_at=None,
        pid=None,
        process_group_id=None,
        child_pids=[],
        oom_detected=False,
        return_code=None,
        error=None,
    )

    seen_path: Path | None = None
    seen_lines: int | None = None
    seen_follow: bool | None = None

    monkeypatch.setattr("xax.cli.queue.xax.get_running_job", lambda: None)
    monkeypatch.setattr("xax.cli.queue.xax.list_jobs", lambda: [older_job, newest_job])
    monkeypatch.setattr("xax.cli.queue.xax.get_job", lambda _job_id: None)
    monkeypatch.setattr("xax.cli.queue._resolve_log_file", lambda job, kind: Path(f"/tmp/{job.job_id}-{kind}.log"))

    def fake_tail_file(path: Path, lines: int, follow: bool) -> int:
        nonlocal seen_path, seen_lines, seen_follow
        seen_path = path
        seen_lines = lines
        seen_follow = follow
        return 0

    monkeypatch.setattr("xax.cli.queue._tail_file", fake_tail_file)

    with pytest.raises(SystemExit) as system_exit:
        main(["tail", "--kind", "task", "--lines", "9"])

    assert system_exit.value.code == 0
    assert seen_path == Path("/tmp/job-0000003-task.log")
    assert seen_lines == 9
    assert seen_follow is False


def test_tail_without_job_id_errors_when_queue_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("xax.cli.queue.xax.get_running_job", lambda: None)
    monkeypatch.setattr("xax.cli.queue.xax.list_jobs", lambda: [])

    with pytest.raises(SystemExit) as system_exit:
        main(["tail"])

    assert system_exit.value.code == 1
    assert "No queued jobs to tail" in capsys.readouterr().err


def test_wait_returns_immediately_when_no_running_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("xax.cli.queue.xax.get_running_job", lambda: None)

    def fail_sleep(_seconds: float) -> None:
        raise AssertionError("wait should not sleep when there is no running job")

    monkeypatch.setattr("xax.cli.queue.time.sleep", fail_sleep)

    with pytest.raises(SystemExit) as system_exit:
        main(["wait"])

    assert system_exit.value.code == 0


def test_wait_polls_until_original_running_job_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    running_job = QueuedJob(
        job_id="job-0000010",
        task_key="tests.task.DummyTask",
        launcher="single",
        python_executable=sys.executable,
        status="running",
        run_dir="/tmp/run_010",
        stage_dir="/tmp/run_010/code",
        config_path="/tmp/run_010/config.yaml",
        observer_log_path="/tmp/run_010/queue_observer.log",
        enqueued_at=1.0,
        started_at=2.0,
        ended_at=None,
        pid=1010,
        process_group_id=1010,
        child_pids=[],
        oom_detected=False,
        return_code=None,
        error=None,
    )
    finished_job = QueuedJob(
        job_id="job-0000010",
        task_key="tests.task.DummyTask",
        launcher="single",
        python_executable=sys.executable,
        status="completed",
        run_dir="/tmp/run_010",
        stage_dir="/tmp/run_010/code",
        config_path="/tmp/run_010/config.yaml",
        observer_log_path="/tmp/run_010/queue_observer.log",
        enqueued_at=1.0,
        started_at=2.0,
        ended_at=3.0,
        pid=None,
        process_group_id=None,
        child_pids=[],
        oom_detected=False,
        return_code=0,
        error=None,
    )
    polled_states = iter(
        [
            QueueState(version=1, next_job_idx=2, queue=[], running_job_id="job-0000010", jobs={}),
            QueueState(version=1, next_job_idx=3, queue=[], running_job_id="job-0000099", jobs={}),
        ]
    )
    sleep_calls: list[float] = []

    monkeypatch.setattr("xax.cli.queue.xax.get_running_job", lambda: running_job)
    monkeypatch.setattr("xax.cli.queue.xax.read_queue_state", lambda: next(polled_states))
    monkeypatch.setattr("xax.cli.queue.xax.get_job", lambda job_id: finished_job if job_id == "job-0000010" else None)
    monkeypatch.setattr("xax.cli.queue.time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(SystemExit) as system_exit:
        main(["wait", "--poll-seconds", "0.5"])

    assert system_exit.value.code == 0
    assert sleep_calls == [0.5]
