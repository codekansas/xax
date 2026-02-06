"""Defines a launcher to train a model locally, on all available devices."""

import logging
from typing import TYPE_CHECKING, Any, cast

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.data_loader import Config, DataloadersMixin
    from xax.task.mixins.runnable import RunnableMixin


def run_dataset_processing(
    task: "type[DataloadersMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logger = configure_logging()
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.add_logger_handlers(logger)
    task_obj.build_all_datasets()


class DatasetLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Any]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        logger = configure_logging()
        run_dataset_processing(cast("type[DataloadersMixin[Any]]", task), *cfgs, use_cli=use_cli, logger=logger)
