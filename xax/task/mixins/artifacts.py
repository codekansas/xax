"""Defines a mixin for storing any task artifacts."""

import functools
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypeVar

import jax

from xax.core.conf import field, get_run_dir
from xax.nn.parallel import is_master
from xax.task.base import BaseConfig, BaseTask
from xax.utils.experiments import stage_environment
from xax.utils.logging import LOG_STATUS, RankFilter
from xax.utils.text import show_info

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class ArtifactsConfig(BaseConfig):
    exp_dir: str | None = field(None, help="The fixed experiment directory")
    log_to_file: bool = field(True, help="If set, add a file handler to the logger to write all logs to the exp dir")


Config = TypeVar("Config", bound=ArtifactsConfig)


class ArtifactsMixin(BaseTask[Config]):
    _exp_dir: Path | None
    _stage_dir: Path | None

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._exp_dir = None
        self._stage_dir = None

    def add_logger_handlers(self, logger: logging.Logger) -> None:
        super().add_logger_handlers(logger)
        if is_master() and self.config.log_to_file:
            file_handler = logging.FileHandler(self.exp_dir / "logs.txt")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            file_handler.addFilter(RankFilter(rank=0))
            logger.addHandler(file_handler)

    @functools.cached_property
    def run_dir(self) -> Path:
        run_dir = get_run_dir()
        if run_dir is None:
            try:
                task_file = inspect.getfile(self.__class__)
                run_dir = Path(task_file).resolve().parent
            except OSError:
                logger.warning(
                    "Could not resolve task path for %s, returning current working directory", self.__class__.__name__
                )
                run_dir = Path.cwd()
        return run_dir / self.task_name

    @property
    def exp_dir(self) -> Path:
        return self.get_exp_dir()

    def set_exp_dir(self, exp_dir: str | Path) -> Self:
        self._exp_dir = Path(exp_dir).expanduser().resolve()
        return self

    def get_exp_dir(self) -> Path:
        if self._exp_dir is not None:
            return self._exp_dir

        if self.config.exp_dir is not None:
            exp_dir = Path(self.config.exp_dir).expanduser().resolve()
            exp_dir.mkdir(parents=True, exist_ok=True)
            self._exp_dir = exp_dir
            logger.log(LOG_STATUS, self._exp_dir)
            return self._exp_dir

        def get_exp_dir(run_id: int) -> Path:
            return self.run_dir / f"run_{run_id:03d}"

        run_id = 0
        while (exp_dir := get_exp_dir(run_id)).is_dir():
            run_id += 1
        exp_dir.mkdir(exist_ok=True, parents=True)
        self._exp_dir = exp_dir.expanduser().resolve()
        logger.log(LOG_STATUS, self._exp_dir)
        return self._exp_dir

    def stage_environment(self) -> Path | None:
        if self._stage_dir is None:
            stage_dir = (self.exp_dir / "code").resolve()
            try:
                stage_environment(self, stage_dir)
            except Exception:
                logger.exception("Failed to stage environment!")
                return None
            self._stage_dir = stage_dir
        return self._stage_dir

    def on_training_end(self) -> None:
        super().on_training_end()

        if is_master():
            if self._exp_dir is None:
                show_info("Exiting training job", important=True)
            else:
                show_info(f"Exiting training job for {self.exp_dir}", important=True)
