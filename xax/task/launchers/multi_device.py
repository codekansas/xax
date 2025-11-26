"""Defines a launcher to train a model locally, on all available devices."""

import logging
from typing import TYPE_CHECKING

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


def run_training(
    task: "type[RunnableMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logger = configure_logging()
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.add_logger_handlers(logger)
    task_obj.run()


class MultiDeviceLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        logger = configure_logging()
        run_training(task, *cfgs, use_cli=use_cli, logger=logger)
