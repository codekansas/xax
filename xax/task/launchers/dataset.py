"""Defines a launcher to build datasets without running training."""

from typing import TYPE_CHECKING, Any, cast

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.launcher.task_runner import run_dataset_processing
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.data_loader import DataloadersMixin
    from xax.task.mixins.runnable import RunnableMixin


class DatasetLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Any]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        logger = configure_logging()
        run_dataset_processing(cast("type[DataloadersMixin[Any]]", task), *cfgs, use_cli=use_cli, logger=logger)
