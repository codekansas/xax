"""Defines a launcher that queues jobs for the queue observer process."""

import sys
from typing import TYPE_CHECKING, cast

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.launcher.queue_state import enqueue_job, is_observer_active
from xax.utils.launcher.queued import (
    get_task_run_dir,
    parse_queue_cli_args,
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
            raise RuntimeError(
                "Queued observer is not active. Start the user service with `xax queue start`."
            )

        args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
        queue_args, task_cli_args = parse_queue_cli_args(args)
        use_cli_next: bool | list[str] = False if not use_cli else task_cli_args

        task_obj = task.get_task(*cfgs, use_cli=use_cli_next)
        run_dir = get_task_run_dir(task_obj)
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.yaml"
        staged_code_dir = run_dir / "code"
        observer_log_path = run_dir / "queue_observer.log"

        config_payload_raw = to_primitive(task_obj.config, preserve_missing=True)
        if not isinstance(config_payload_raw, dict):
            raise TypeError(f"Expected task config payload to be a mapping, got {type(config_payload_raw)!r}")
        config_payload = cast(dict[str, object], config_payload_raw)
        # Persist the resolved queue run directory so the observer-executed
        # task writes logs/metrics into the staged run directory instead of
        # allocating a new `run_*` directory.
        config_payload["run_dir"] = str(run_dir)

        save_yaml(config_path, config_payload)
        stage_task_environment(task_obj, staged_code_dir)

        if queue_args.queue_position is not None and queue_args.queue_position < 1:
            raise ValueError("--queue-position must be >= 1")
        queue_position_idx = None if queue_args.queue_position is None else queue_args.queue_position - 1

        # Queue jobs always run through MultiDeviceLauncher.
        canonical_launcher = "multi"
        job_id = enqueue_job(
            task_key=task_obj.task_key,
            launcher=canonical_launcher,
            run_dir=run_dir,
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
            run_dir,
            canonical_launcher,
        )
