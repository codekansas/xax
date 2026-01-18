"""Defines a mixin for incorporating some logging functionality."""

import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Generic, Self, TypeVar

import jax
import numpy as np

from xax.core.conf import field
from xax.core.state import State
from xax.task.logger import Logger, LoggerImpl
from xax.task.loggers.json import JsonLogger
from xax.task.loggers.state import StateLogger
from xax.task.loggers.stdout import StdoutLogger
from xax.task.loggers.tensorboard import TensorboardLogger
from xax.task.loggers.wandb import (
    WandbConfigMode,
    WandbConfigModeOption,
    WandbConfigReinitOption,
    WandbConfigResume,
    WandbLogger,
)
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin


class LoggerBackend(str, Enum):
    STDOUT = "stdout"
    JSON = "json"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


@jax.tree_util.register_dataclass
@dataclass
class LoggerConfig(ArtifactsConfig):
    log_interval_seconds: float = field(
        value=1.0,
        help="The interval between successive log lines.",
    )
    logger_backend: list[LoggerBackend] = field(
        value=[LoggerBackend.STDOUT, LoggerBackend.JSON, LoggerBackend.TENSORBOARD],
        help="The logger backend to use",
    )
    tensorboard_log_interval_seconds: float = field(
        value=10.0,
        help="The interval between successive Tensorboard log lines.",
    )
    wandb_project: str | None = field(
        value=None,
        help="The name of the W&B project to log to.",
    )
    wandb_entity: str | None = field(
        value=None,
        help="The W&B entity (team or user) to log to.",
    )
    wandb_name: str | None = field(
        value=None,
        help="The name of this run in W&B.",
    )
    wandb_tags: list[str] | None = field(
        value=None,
        help="List of tags for this W&B run.",
    )
    wandb_notes: str | None = field(
        value=None,
        help="Notes about this W&B run.",
    )
    wandb_log_interval_seconds: float = field(
        value=10.0,
        help="The interval between successive W&B log lines.",
    )
    wandb_mode: WandbConfigMode = field(
        value=WandbConfigModeOption.ONLINE,
        help="Mode for wandb (online, offline, or disabled).",
    )
    wandb_resume: WandbConfigResume = field(
        value=False,
        help="Whether to resume a previous run. Can be a run ID string.",
    )
    wandb_reinit: WandbConfigReinitOption = field(
        value=WandbConfigReinitOption.RETURN_PREVIOUS,
        help="Whether to allow multiple wandb.init() calls in the same process.",
    )


Config = TypeVar("Config", bound=LoggerConfig)


def get_env_var(name: str, default: bool) -> bool:
    if name not in os.environ:
        return default
    return os.environ[name].strip() == "1"


class LoggerMixin(ArtifactsMixin[Config], Generic[Config]):
    logger: Logger

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.logger = Logger()

        # Hook up the decode_tokens method to the logger.
        self.logger.decode_tokens = self.decode_tokens

    def log_directory(self) -> Path | None:
        return None

    def add_logger(self, *logger: LoggerImpl) -> None:
        self.logger.add_logger(*logger)

    def set_loggers(self) -> None:
        for backend in self.config.logger_backend:
            self.add_logger(self._create_logger_backend(backend))

        # If this is also an ArtifactsMixin, we should default add some
        # additional loggers which log data to the artifacts directory.
        if isinstance(self, ArtifactsMixin):
            self.add_logger(StateLogger(run_directory=self.exp_dir))

    def _create_logger_backend(self, backend: LoggerBackend) -> LoggerImpl:
        match backend:
            case LoggerBackend.STDOUT:
                return StdoutLogger(
                    log_interval_seconds=self.config.log_interval_seconds,
                )

            case LoggerBackend.JSON:
                return JsonLogger(
                    run_directory=self.exp_dir,
                    log_interval_seconds=self.config.log_interval_seconds,
                )

            case LoggerBackend.TENSORBOARD:
                return TensorboardLogger(
                    run_directory=self.exp_dir,
                    log_interval_seconds=self.config.tensorboard_log_interval_seconds,
                )

            case LoggerBackend.WANDB:
                return WandbLogger(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_name,
                    run_directory=self.exp_dir,
                    config=asdict(self.config),
                    tags=self.config.wandb_tags,
                    notes=self.config.wandb_notes,
                    log_interval_seconds=self.config.wandb_log_interval_seconds,
                    reinit=self.config.wandb_reinit,
                    resume=self.config.wandb_resume,
                    mode=self.config.wandb_mode,
                )

            case _:
                # This shouldn't happen, as validation should take care of this
                raise Exception(f"Invalid logger_backend '{self.config.logger_backend}'")

    def decode_tokens(self, tokens: np.ndarray, token_type: str) -> str:
        raise NotImplementedError(
            "When using a Tokens metric you must implement the `decode_tokens` method "
            "to convert to a string which can be logged. The `token_type` argument "
            "is passed from the `log_tokens` method or `LogTokens` class."
        )

    def write_logs(self, state: State, heavy: bool) -> None:
        self.logger.write(state, heavy)

    def __enter__(self) -> Self:
        self.logger.__enter__()
        return self

    def __exit__(self, t: type[BaseException] | None, e: BaseException | None, tr: TracebackType | None) -> None:
        self.logger.__exit__(t, e, tr)
        return super().__exit__(t, e, tr)
