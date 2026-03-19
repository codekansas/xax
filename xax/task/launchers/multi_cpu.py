"""Defines a launcher to train a model on multiple CPU devices for testing model parallelism."""

from typing import TYPE_CHECKING

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.launcher.multi_cpu import configure_cpu_devices
from xax.utils.launcher.task_runner import run_runnable_task
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


class MultiCpuLauncher(BaseLauncher):
    """Launcher for multi-CPU training, useful for testing model parallelism without GPUs."""

    def __init__(self, num_cpus: int | None = None) -> None:
        super().__init__()
        self.num_cpus = num_cpus

    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        logger = configure_logging()
        configure_cpu_devices(num_cpus=self.num_cpus, logger=logger)
        run_runnable_task(task, *cfgs, use_cli=use_cli, logger=logger)
