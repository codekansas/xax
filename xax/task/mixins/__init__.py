"""Defines a single interface for all the mixins."""

from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from xax.task.mixins.checkpointing import CheckpointingConfig, CheckpointingMixin
from xax.task.mixins.compile import CompileConfig, CompileMixin
from xax.task.mixins.cpu_stats import CPUStatsConfig, CPUStatsMixin
from xax.task.mixins.data_loader import DataloadersConfig, DataloadersMixin
from xax.task.mixins.gpu_stats import GPUStatsConfig, GPUStatsMixin
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.parallel import ParallelConfig, ParallelMixin
from xax.task.mixins.process import ProcessConfig, ProcessMixin
from xax.task.mixins.runnable import RunnableConfig, RunnableMixin
from xax.task.mixins.step_wrapper import StepContextConfig, StepContextMixin
from xax.task.mixins.supervised import SupervisedConfig, SupervisedMixin
from xax.task.mixins.train import InitParams, TrainConfig, TrainMixin
