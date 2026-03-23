"""Composes the base task with all the mixins into a single task interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import jax

from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins import (
    ArtifactsConfig,
    ArtifactsMixin,
    CheckpointingConfig,
    CheckpointingMixin,
    CompileConfig,
    CompileMixin,
    CPUStatsConfig,
    CPUStatsMixin,
    DataloadersConfig,
    DataloadersMixin,
    GPUStatsConfig,
    GPUStatsMixin,
    InitParams,
    LoggerConfig,
    LoggerMixin,
    ProcessConfig,
    ProcessMixin,
    RunnableConfig,
    RunnableMixin,
    StepContextConfig,
    StepContextMixin,
    SupervisedConfig as BaseSupervisedConfig,
    SupervisedMixin as BaseSupervisedMixin,
    TrainConfig,
    TrainMixin,
)


@jax.tree_util.register_dataclass
@dataclass
class Config(
    TrainConfig,
    CheckpointingConfig,
    CompileConfig,
    DataloadersConfig,
    CPUStatsConfig,
    GPUStatsConfig,
    ProcessConfig,
    LoggerConfig,
    StepContextConfig,
    ArtifactsConfig,
    RunnableConfig,
    BaseConfig,
):
    pass


ConfigT = TypeVar("ConfigT", bound=Config)
InitParamsT = TypeVar("InitParamsT", bound=InitParams)


class Task[ConfigT, InitParamsT](
    TrainMixin[ConfigT, InitParamsT],
    CheckpointingMixin[ConfigT],
    CompileMixin[ConfigT],
    DataloadersMixin[ConfigT],
    CPUStatsMixin[ConfigT],
    GPUStatsMixin[ConfigT],
    ProcessMixin[ConfigT],
    LoggerMixin[ConfigT],
    StepContextMixin[ConfigT],
    ArtifactsMixin[ConfigT],
    RunnableMixin[ConfigT],
    BaseTask[ConfigT],
):
    pass


@jax.tree_util.register_dataclass
@dataclass
class SupervisedConfig(
    BaseSupervisedConfig,
    Config,
):
    pass


SupervisedConfigT = TypeVar("SupervisedConfigT", bound=SupervisedConfig)


class SupervisedTask[SupervisedConfigT](
    BaseSupervisedMixin[SupervisedConfigT],
    Task[SupervisedConfigT, InitParams],
):
    pass
