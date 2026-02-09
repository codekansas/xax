"""Defines a launcher to train a model locally, on a single device."""

from typing import TYPE_CHECKING

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.launcher.cli import help_requested
from xax.utils.launcher.gpu_visibility import apply_queue_gpu_visibility
from xax.utils.launcher.single_device import configure_single_device
from xax.utils.launcher.task_runner import run_runnable_task
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


class SingleDeviceLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        if help_requested(use_cli):
            run_runnable_task(task, *cfgs, use_cli=use_cli, logger=None)
            return

        logger = configure_logging()
        visible_gpu_indices = apply_queue_gpu_visibility(logger)
        if visible_gpu_indices == []:
            raise RuntimeError(
                "No GPUs remain for local single-device launch after hiding queue-reserved GPUs. "
                "Use a non-queue GPU via CUDA_VISIBLE_DEVICES or adjust queue GPU allocation "
                "(xax queue start --queue-gpus/--queue-num-gpus)."
            )
        configure_single_device(logger)
        run_runnable_task(task, *cfgs, use_cli=use_cli, logger=logger)
