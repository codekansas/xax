"""Defines a launcher to train a model locally, on all available devices."""

import logging
import shutil
from typing import TYPE_CHECKING

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.data_loader import Config, DataloadersMixin


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

    cache_path = task_obj.preprocessed_dataset_path
    if cache_path.exists() and not task_obj.config.overwrite_dataset:
        response: str = ""
        while response.lower() not in ["y", "n"]:
            response = input(f"Dataset already exists at {cache_path}. Overwrite? (y/n): ")
        if response.lower() == "n":
            logger.info("Dataset not overwritten. Exiting...")
            return

    ds = task_obj.preprocess_dataset()
    if cache_path.exists():
        logger.info("Removing existing dataset at %s", cache_path)
        shutil.rmtree(cache_path)

    logger.info("Saving dataset to %s", cache_path)
    ds.save_to_disk(cache_path)
    logger.info("Dataset saved to %s", cache_path)


class DatasetLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[DataloadersMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        logger = configure_logging()
        run_dataset_processing(task, *cfgs, use_cli=use_cli, logger=logger)
