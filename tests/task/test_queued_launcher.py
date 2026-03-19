"""Tests for the queued launcher."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from xax.core.conf import _load_user_config_cached
from xax.task.base import BaseConfig, BaseTask
from xax.task.launchers.queued import QueuedLauncher
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from xax.task.mixins.runnable import RunnableConfig, RunnableMixin
from xax.utils.launcher.queue_state import clear_observer, read_queue_state, register_observer


@dataclass
class DummyQueuedConfig(ArtifactsConfig, RunnableConfig, BaseConfig):
    pass


class DummyQueuedTask(
    ArtifactsMixin[DummyQueuedConfig],
    RunnableMixin[DummyQueuedConfig],
    BaseTask[DummyQueuedConfig],
):
    def run(self) -> None:
        return None


def _configure_user_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAX_HOME", str(tmp_path / ".xax"))
    monkeypatch.delenv("XAXRC_PATH", raising=False)
    _load_user_config_cached.cache_clear()


def test_queued_launcher_requires_active_observer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    launcher = QueuedLauncher()
    with pytest.raises(RuntimeError, match="observer"):
        launcher.launch(
            DummyQueuedTask,
            DummyQueuedConfig(run_dir=str(tmp_path / "run_a")),
            use_cli=False,
        )


def test_queued_launcher_enqueues_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    observer_pid = os.getpid()
    register_observer(observer_pid, status="idle")
    try:
        launcher = QueuedLauncher()
        run_dir = tmp_path / "run_b"
        launcher.launch(
            DummyQueuedTask,
            DummyQueuedConfig(run_dir=str(run_dir)),
            use_cli=False,
        )

        state = read_queue_state()
        assert len(state.queue) == 1
        job_id = state.queue[0]
        queued_job = state.jobs[job_id]

        assert queued_job.status == "queued"
        assert queued_job.task_key.endswith(".DummyQueuedTask")
        assert queued_job.launcher == "multi"
        assert queued_job.python_executable == sys.executable
        assert Path(queued_job.config_path).exists()
        assert Path(queued_job.stage_dir).exists()
    finally:
        clear_observer(observer_pid)


def test_queued_launcher_ignores_config_runs_dir_for_queue_run_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    observer_pid = os.getpid()
    register_observer(observer_pid, status="idle")
    canonical_runs_root = (tmp_path / "canonical_runs").resolve()
    monkeypatch.setattr(DummyQueuedTask, "_get_default_runs_dir", lambda self: canonical_runs_root)

    try:
        launcher = QueuedLauncher()
        provided_runs_dir = (tmp_path / "queue_mnist_flow").resolve()
        launcher.launch(
            DummyQueuedTask,
            DummyQueuedConfig(runs_dir=str(provided_runs_dir)),
            use_cli=False,
        )
        launcher.launch(
            DummyQueuedTask,
            DummyQueuedConfig(runs_dir=str(provided_runs_dir)),
            use_cli=False,
        )

        state = read_queue_state()
        assert len(state.queue) == 2
        queued_jobs = [state.jobs[job_id] for job_id in state.queue]
        queued_run_dirs = [Path(job.run_dir).resolve() for job in queued_jobs]

        expected_task_root = canonical_runs_root / "dummy_queued_task"
        assert queued_run_dirs[0] == expected_task_root / "run_000"
        assert queued_run_dirs[1] == expected_task_root / "run_001"
        assert all(
            provided_runs_dir not in run_dir.parents and run_dir != provided_runs_dir
            for run_dir in queued_run_dirs
        )
    finally:
        clear_observer(observer_pid)


def test_queued_launcher_allows_queue_reordering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    observer_pid = os.getpid()
    register_observer(observer_pid, status="idle")
    try:
        launcher = QueuedLauncher()
        launcher.launch(
            DummyQueuedTask,
            DummyQueuedConfig(run_dir=str(tmp_path / "run_1")),
            use_cli=False,
        )
        launcher.launch(
            DummyQueuedTask,
            DummyQueuedConfig(run_dir=str(tmp_path / "run_2")),
            use_cli=["--queue-position", "1"],
        )
        state = read_queue_state()
        assert len(state.queue) == 2
        first_job = state.jobs[state.queue[0]]
        second_job = state.jobs[state.queue[1]]
        assert first_job.run_dir.endswith("run_2")
        assert second_job.run_dir.endswith("run_1")
    finally:
        clear_observer(observer_pid)


def test_cli_launcher_supports_queued_choice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    observer_pid = os.getpid()
    register_observer(observer_pid, status="idle")
    try:
        DummyQueuedTask.launch(
            DummyQueuedConfig(run_dir=str(tmp_path / "run_cli")),
            use_cli=["--launcher", "queued"],
        )
        state = read_queue_state()
        assert len(state.queue) == 1
    finally:
        clear_observer(observer_pid)


def test_queued_launcher_rejects_removed_queued_launcher_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_user_dir(monkeypatch, tmp_path)
    observer_pid = os.getpid()
    register_observer(observer_pid, status="idle")
    try:
        launcher = QueuedLauncher()
        with pytest.raises(ValueError, match="no longer supported"):
            launcher.launch(
                DummyQueuedTask,
                DummyQueuedConfig(run_dir=str(tmp_path / "run_deprecated_flag")),
                use_cli=["--queued-launcher", "single"],
            )
    finally:
        clear_observer(observer_pid)
