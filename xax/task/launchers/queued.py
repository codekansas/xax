"""Defines a launcher that queues jobs for the queue observer process."""

import sys
from typing import TYPE_CHECKING

from xax.task.base import RawConfigType
from xax.task.interfaces.queued import QueuedArtifactsTask
from xax.task.launchers.base import BaseLauncher
from xax.utils.launcher.queue_state import enqueue_job, is_observer_active
from xax.utils.launcher.queued import (
    parse_queue_cli_args,
    resolve_queued_run_paths,
    stage_task_environment,
)
from xax.utils.logging import LOG_STATUS, configure_logging
from xax.utils.structured_config import save_yaml, to_primitive

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


class QueuedLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        if not is_observer_active():
            raise RuntimeError("Queued observer is not active. Start the user service with `xax queue start`.")

        args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
        queue_args, task_cli_args = parse_queue_cli_args(args)
        use_cli_next: bool | list[str] = False if not use_cli else task_cli_args

        task_obj = task.get_task(*cfgs, use_cli=use_cli_next)
        if not isinstance(task_obj, QueuedArtifactsTask):
            raise TypeError(
                "Queued launcher requires tasks compatible with ArtifactsMixin "
                "(config.run_dir/config.runs_dir, run_dir property, set_run_dir, get_queue_runs_dir)."
            )
        run_paths = resolve_queued_run_paths(task_obj)
        run_dir = run_paths.run_dir

        config_payload_raw = to_primitive(task_obj.config, preserve_missing=True)
        if not isinstance(config_payload_raw, dict):
            raise TypeError(f"Expected task config payload to be a mapping, got {type(config_payload_raw)!r}")
        config_payload = {str(key): value for key, value in config_payload_raw.items()}
        # Persist the resolved queue run directory so the observer-executed
        # task writes logs/metrics into the staged run directory instead of
        # allocating a new `run_*` directory.
        config_payload["run_dir"] = str(run_dir)

        save_yaml(run_paths.config_path, config_payload)
        stage_task_environment(task_obj, run_paths.stage_dir)

        if queue_args.queue_position is not None and queue_args.queue_position < 1:
            raise ValueError("--queue-position must be >= 1")
        queue_position_idx = None if queue_args.queue_position is None else queue_args.queue_position - 1

        # Queue jobs always run through MultiDeviceLauncher.
        canonical_launcher = "multi"
        job_id = enqueue_job(
            task_key=task_obj.task_key,
            launcher=canonical_launcher,
            python_executable=sys.executable,
            run_dir=run_dir,
            stage_dir=run_paths.stage_dir,
            config_path=run_paths.config_path,
            observer_log_path=run_paths.observer_log_path,
            position_idx=queue_position_idx,
        )
        logger = configure_logging()
        logger.log(
            LOG_STATUS,
            "Queued %s (%s) at %s [launcher=%s]",
            job_id,
            task_obj.task_key,
            run_dir,
            canonical_launcher,
        )
