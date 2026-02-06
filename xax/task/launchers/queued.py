"""Defines a launcher that queues jobs for a local observer process."""

import argparse
import inspect
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

from omegaconf import OmegaConf

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.task.launchers.queue_state import enqueue_job, is_observer_active
from xax.utils.experiments import stage_environment
from xax.utils.logging import LOG_STATUS, configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


QueuedTargetLauncher = Literal["single", "s", "multi", "m", "multi_cpu", "mc", "dataset", "d"]


def _canonical_launcher(choice: QueuedTargetLauncher) -> str:
    match choice:
        case "single" | "s":
            return "single"
        case "multi" | "m":
            return "multi"
        case "multi_cpu" | "mc":
            return "multi_cpu"
        case "dataset" | "d":
            return "dataset"
    raise ValueError(f"Unsupported queued target launcher: {choice}")


def _get_task_run_dir(task_obj: object) -> Path:
    run_dir = getattr(task_obj, "run_dir", None)
    if run_dir is None:
        run_dir = getattr(task_obj, "exp_dir", None)
    if run_dir is None or not isinstance(run_dir, Path):
        raise TypeError("Queued launcher requires task with `run_dir` support (use ArtifactsMixin-based tasks)")
    return run_dir.expanduser().resolve()


def _stage_task_environment(task_obj: object, staged_code_dir: Path) -> None:
    try:
        stage_environment(task_obj, staged_code_dir)
        return
    except Exception as stage_error:
        if (task_module := inspect.getmodule(task_obj.__class__)) is None:
            raise RuntimeError(f"Failed to stage environment for {task_obj.__class__}") from stage_error
        if (module_spec := task_module.__spec__) is None or module_spec.origin is None:
            raise RuntimeError(f"Failed to stage environment for {task_obj.__class__}") from stage_error

    # If the task module is imported as a top-level file (common in ad-hoc
    # scripts/tests), fall back to staging that module file directly.
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


class QueuedLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        if not is_observer_active():
            raise RuntimeError("Queued observer is not active. Start it with: xax queue start")

        args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--queued-launcher",
            choices=get_args(QueuedTargetLauncher),
            default="multi",
            help="Launcher used by the observer when running this queued job",
        )
        parser.add_argument(
            "--queue-position",
            type=int,
            default=None,
            help="Optional 1-based queue position. Omit to append to the back of the queue.",
        )
        queue_args, task_cli_args = parser.parse_known_intermixed_args(args=args)
        use_cli_next: bool | list[str] = False if not use_cli else task_cli_args

        target_launcher = cast(QueuedTargetLauncher, queue_args.queued_launcher)
        if target_launcher in ("dataset", "d"):
            from xax.task.mixins.data_loader import DataloadersMixin  # noqa: PLC0415

            if not issubclass(task, DataloadersMixin):
                raise ValueError("The task must be a subclass of DataloadersMixin to use `--queued-launcher dataset`")

        task_obj = task.get_task(*cfgs, use_cli=use_cli_next)
        exp_dir = _get_task_run_dir(task_obj)
        exp_dir.mkdir(parents=True, exist_ok=True)

        config_path = exp_dir / "queued_config.yaml"
        staged_code_dir = exp_dir / "code"
        observer_log_path = exp_dir / "queue_observer.log"

        resolved_cfg = OmegaConf.structured(task_obj.config)
        OmegaConf.save(resolved_cfg, config_path)
        _stage_task_environment(task_obj, staged_code_dir)

        if queue_args.queue_position is not None and queue_args.queue_position < 1:
            raise ValueError("--queue-position must be >= 1")
        queue_position_idx = None if queue_args.queue_position is None else queue_args.queue_position - 1

        canonical_launcher = _canonical_launcher(target_launcher)
        job_id = enqueue_job(
            task_key=task_obj.task_key,
            launcher=canonical_launcher,
            exp_dir=exp_dir,
            stage_dir=staged_code_dir,
            config_path=config_path,
            observer_log_path=observer_log_path,
            position_idx=queue_position_idx,
        )
        logger = configure_logging()
        logger.log(
            LOG_STATUS,
            "Queued %s (%s) at %s [launcher=%s]",
            job_id,
            task_obj.task_key,
            exp_dir,
            canonical_launcher,
        )
