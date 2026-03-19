"""Helpers for executing task entrypoints from launchers."""

import logging
from typing import TYPE_CHECKING, Any

from xax.task.base import RawConfigType
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.data_loader import DataloadersMixin
    from xax.task.mixins.runnable import RunnableMixin


def run_runnable_task(
    task: "type[RunnableMixin[Any]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
    logger: logging.Logger | None = None,
) -> None:
    """Builds and executes a runnable task."""
    if logger is None:
        logger = configure_logging()
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.add_logger_handlers(logger)
    task_obj.run()


def run_dataset_processing(
    task: "type[DataloadersMixin[Any]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
    logger: logging.Logger | None = None,
) -> None:
    """Builds and executes dataset preprocessing for a task."""
    if logger is None:
        logger = configure_logging()
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.add_logger_handlers(logger)
    task_obj.build_all_datasets()
