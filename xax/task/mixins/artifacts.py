"""Defines a mixin for storing any task artifacts."""

import functools
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypeVar

import jax

from xax.core.conf import field, get_runs_dir
from xax.nn.parallel import is_master
from xax.task.base import BaseConfig, BaseTask
from xax.utils.experiments import stage_environment
from xax.utils.logging import LOG_STATUS, RankFilter
from xax.utils.text import show_info

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class ArtifactsConfig(BaseConfig):
    runs_dir: str | None = field(None, help="Directory containing all runs for this task")
    run_dir: str | None = field(None, help="The fixed run directory")
    exp_dir: str | None = field(None, help="Deprecated alias for `run_dir`")
    log_to_file: bool = field(True, help="If set, add a file handler to the logger to write all logs to the run dir")


Config = TypeVar("Config", bound=ArtifactsConfig)


class ArtifactsMixin(BaseTask[Config]):
    _run_dir: Path | None
    _exp_dir: Path | None
    _stage_dir: Path | None

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._run_dir = None
        self._exp_dir = None
        self._stage_dir = None

    def add_logger_handlers(self, logger: logging.Logger) -> None:
        super().add_logger_handlers(logger)
        if is_master() and self.config.log_to_file:
            file_handler = logging.FileHandler(self.run_dir / "logs.txt")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            file_handler.addFilter(RankFilter(rank=0))
            logger.addHandler(file_handler)

    @functools.cached_property
    def runs_dir(self) -> Path:
        runs_dir = self._get_default_runs_dir()
        if self.config.runs_dir is not None:
            runs_dir = Path(self.config.runs_dir).expanduser().resolve()
        return runs_dir / self.task_name

    def _get_default_runs_dir(self) -> Path:
        runs_dir = get_runs_dir()
        if runs_dir is None:
            try:
                task_file = inspect.getfile(self.__class__)
                runs_dir = Path(task_file).resolve().parent
            except OSError:
                logger.warning(
                    "Could not resolve task path for %s, returning current working directory", self.__class__.__name__
                )
                runs_dir = Path.cwd()
        return runs_dir

    @property
    def run_dir(self) -> Path:
        return self.get_run_dir()

    def set_run_dir(self, run_dir: str | Path) -> Self:
        self._run_dir = Path(run_dir).expanduser().resolve()
        self._exp_dir = self._run_dir
        return self

    def get_run_dir(self) -> Path:
        if self._run_dir is not None:
            return self._run_dir
        if self._exp_dir is not None:
            self._run_dir = self._exp_dir
            return self._run_dir

        fixed_run_dir = self.config.run_dir if self.config.run_dir is not None else self.config.exp_dir
        if fixed_run_dir is not None:
            run_dir = Path(fixed_run_dir).expanduser().resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            self._run_dir = run_dir
            self._exp_dir = run_dir
            logger.log(LOG_STATUS, self._run_dir)
            return self._run_dir

        def get_next_run_dir(run_id: int) -> Path:
            return self.runs_dir / f"run_{run_id:03d}"

        run_id = 0
        while (run_dir := get_next_run_dir(run_id)).is_dir():
            run_id += 1
        run_dir.mkdir(exist_ok=True, parents=True)
        self._run_dir = run_dir.expanduser().resolve()
        self._exp_dir = self._run_dir
        logger.log(LOG_STATUS, self._run_dir)
        return self._run_dir

    @property
    def exp_dir(self) -> Path:
        """Deprecated alias for `run_dir`."""
        return self.get_run_dir()

    def set_exp_dir(self, exp_dir: str | Path) -> Self:
        """Deprecated alias for `set_run_dir`."""
        return self.set_run_dir(exp_dir)

    def get_exp_dir(self) -> Path:
        """Deprecated alias for `get_run_dir`."""
        return self.get_run_dir()

    def stage_environment(self) -> Path | None:
        if self._stage_dir is None:
            stage_dir = (self.run_dir / "code").resolve()
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
            if self._run_dir is None:
                show_info("Exiting training job", important=True)
            else:
                show_info(f"Exiting training job for {self.run_dir}", important=True)
