"""Helpers for queued launcher argument parsing and staging."""

import inspect
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from xax.utils.cli_args import ARGPARSE_DEST_METADATA_KEY, parse_known_args_as
from xax.utils.experiments import stage_environment


@dataclass(frozen=True)
class QueuedCliArgs:
    queue_position: int | None = field(default=None, metadata={ARGPARSE_DEST_METADATA_KEY: "queue_position"})


def parse_queue_cli_args(cli_args: list[str]) -> tuple[QueuedCliArgs, list[str]]:
    """Parses queue launcher args and returns remaining task args."""
    if any(token == "--queued-launcher" or token.startswith("--queued-launcher=") for token in cli_args):
        raise ValueError("`--queued-launcher` is no longer supported; queued jobs always use MultiDeviceLauncher")
    queue_args, task_cli_args = parse_known_args_as(QueuedCliArgs, cli_args)
    return queue_args, task_cli_args


def _task_config_value(task_obj: object, key: str) -> object | None:
    task_config = getattr(task_obj, "config", None)
    if task_config is None:
        return None
    return getattr(task_config, key, None)


def _get_task_default_runs_dir(task_obj: object) -> Path | None:
    task_name = getattr(task_obj, "task_name", None)
    default_runs_root_fn = getattr(task_obj, "_get_default_runs_dir", None)
    if not isinstance(task_name, str) or not callable(default_runs_root_fn):
        return None
    runs_root = default_runs_root_fn()
    if not isinstance(runs_root, Path):
        return None
    return runs_root.expanduser().resolve() / task_name


def _next_run_dir(task_runs_dir: Path) -> Path:
    run_id = 0
    while (run_dir := task_runs_dir / f"run_{run_id:03d}").exists():
        run_id += 1
    return run_dir


def get_task_run_dir(task_obj: object) -> Path:
    """Returns queue run directory, normalizing queued runs into task-local run IDs.

    In queued mode, if a config provides only `runs_dir` (but not `run_dir`),
    we normalize to the task's canonical runs root and allocate the next
    `run_XXX` directory there. This prevents ad hoc queue labels from creating
    sibling top-level folders like `queue_mnist_flow/...`.
    """
    config_run_dir = _task_config_value(task_obj, "run_dir")
    config_runs_dir = _task_config_value(task_obj, "runs_dir")
    if config_run_dir is None and isinstance(config_runs_dir, str) and config_runs_dir.strip():
        if (default_task_runs_dir := _get_task_default_runs_dir(task_obj)) is not None:
            normalized_run_dir = _next_run_dir(default_task_runs_dir)
            set_run_dir = getattr(task_obj, "set_run_dir", None)
            if callable(set_run_dir):
                set_run_dir(normalized_run_dir)
            return normalized_run_dir.expanduser().resolve()

    # Falls back to standard task-provided run directory behavior.
    run_dir = getattr(task_obj, "run_dir", None)
    if run_dir is None or not isinstance(run_dir, Path):
        raise TypeError("Queued launcher requires task with `run_dir` support (use ArtifactsMixin-based tasks)")
    return run_dir.expanduser().resolve()


def stage_task_environment(task_obj: object, staged_code_dir: Path) -> None:
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
