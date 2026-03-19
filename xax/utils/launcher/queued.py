"""Helpers for queued launcher argument parsing and staging."""

import inspect
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from xax.task.interfaces.queued import QueuedArtifactsTask
from xax.utils.cli_args import ARGPARSE_DEST_METADATA_KEY, parse_known_args_as
from xax.utils.experiments import stage_environment
from xax.utils.run_dirs import next_available_run_dir, resolve_configured_run_dir


@dataclass(frozen=True)
class QueuedCliArgs:
    queue_position: int | None = field(default=None, metadata={ARGPARSE_DEST_METADATA_KEY: "queue_position"})


@dataclass(frozen=True)
class QueuedRunPaths:
    run_dir: Path
    config_path: Path
    stage_dir: Path
    observer_log_path: Path


def parse_queue_cli_args(cli_args: list[str]) -> tuple[QueuedCliArgs, list[str]]:
    """Parses queue launcher args and returns remaining task args."""
    if any(token == "--queued-launcher" or token.startswith("--queued-launcher=") for token in cli_args):
        raise ValueError("`--queued-launcher` is no longer supported; queued jobs always use MultiDeviceLauncher")
    queue_args, task_cli_args = parse_known_args_as(QueuedCliArgs, cli_args)
    return queue_args, task_cli_args


def resolve_queued_run_paths(task_obj: QueuedArtifactsTask) -> QueuedRunPaths:
    """Resolves queue staging paths with a single run-directory policy.

    Policy:
    - If `config.run_dir` is provided, queueing uses that exact run directory.
    - Otherwise, queueing allocates the next run id under `task_obj.get_queue_runs_dir()`.

    The queued launcher intentionally ignores `config.runs_dir` in this path
    so queued runs stay grouped under each task's canonical queue root.
    """
    run_dir = resolve_configured_run_dir(task_obj.config.run_dir, field_name="config.run_dir")
    if run_dir is None:
        run_dir = next_available_run_dir(task_obj.get_queue_runs_dir())
    task_obj.set_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return QueuedRunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        stage_dir=run_dir / "code",
        observer_log_path=run_dir / "queue_observer.log",
    )


def stage_task_environment(
    task_obj: QueuedArtifactsTask,
    staged_code_dir: Path,
) -> None:
    """Stages task source code for queued execution."""
    try:
        stage_environment(task_obj, staged_code_dir)
        return
    except Exception as stage_error:
        if (task_module := inspect.getmodule(task_obj.__class__)) is None:
            raise RuntimeError(f"Failed to stage environment for {task_obj.__class__}") from stage_error
        if (module_spec := task_module.__spec__) is None or module_spec.origin is None:
            raise RuntimeError(f"Failed to stage environment for {task_obj.__class__}") from stage_error

    source_path = Path(module_spec.origin).resolve()
    target_path = staged_code_dir / Path(*module_spec.name.split("."))
    if source_path.name == "__init__.py":
        target_dir = target_path
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_path.parent, target_dir)
    else:
        target_file = target_path.with_suffix(".py")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_file)
