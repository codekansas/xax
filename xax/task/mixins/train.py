"""Defines a mixin for running the training loop."""

import enum
import functools
import itertools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    cast,
    get_args,
    overload,
)

import equinox as eqx
import jax
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from xax.core.conf import field
from xax.core.state import Batch, Output, Phase, State, StepKind
from xax.nn.functions import set_random_seed
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from xax.task.mixins.checkpointing import (
    CheckpointingConfig,
    CheckpointingMixin,
    CheckpointPart,
    load_ckpt,
)
from xax.task.mixins.data_loader import DataloadersConfig, DataloadersMixin
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.runnable import RunnableConfig, RunnableMixin
from xax.task.mixins.step_wrapper import StepContextConfig, StepContextMixin
from xax.utils.experiments import (
    StateTimer,
    diff_configs,
    get_diff_string,
    get_info_json,
    get_state_file_string,
    get_training_code,
)
from xax.utils.logging import LOG_PING, LOG_STATUS
from xax.utils.types.frozen_dict import FrozenDict

logger = logging.getLogger(__name__)

PRINT_FINISH_TIME_EVERY_N_SECONDS = 60 * 2


def cast_step_kind(s: str) -> StepKind:
    assert s in get_args(StepKind), f"`step_kind` must be one of {get_args(StepKind)}, not {s}"
    return cast(StepKind, s)


class Precision(enum.Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


@functools.lru_cache(maxsize=None)
def batch_chunks_schedule(schedule: list[int] | None) -> list[int] | None:
    if schedule is None:
        return None
    if any(s < 1 for s in schedule):
        raise ValueError("Batch chunk schedule must be positive")
    return list(itertools.accumulate([0] + schedule))


@functools.lru_cache(maxsize=None)
def batches_per_step_schedule(schedule: list[int] | None) -> list[int] | None:
    if schedule is None:
        return None
    if any(s < 1 for s in schedule):
        raise ValueError("Batch chunk schedule must be positive")
    return list(itertools.accumulate([0] + schedule))


class ValidStepTimer:
    def __init__(
        self,
        valid_every_n_steps: int | None = None,
        valid_first_n_steps: int = 0,
        valid_every_n_seconds: float | None = None,
        valid_first_n_seconds: float | None = None,
    ) -> None:
        super().__init__()

        self.valid_every_n_steps = valid_every_n_steps
        self.valid_first_n_steps = valid_first_n_steps
        self.valid_every_n_seconds = valid_every_n_seconds
        self.valid_first_n_seconds = valid_first_n_seconds
        self.first_valid_step_flag = True

        self.last_valid_time: float | None = None
        self.last_valid_step: int | None = None

    def _reset(self, state: State) -> None:
        self.last_valid_time = state.elapsed_time_s.item()
        self.last_valid_step = state.num_steps.item()

    def __call__(self, state: State) -> bool:
        if state.num_steps < self.valid_first_n_steps:
            return True

        if self.last_valid_time is None or self.last_valid_step is None:
            self._reset(state)
            return False

        # Step-based validation.
        valid_every_n_steps = self.valid_every_n_steps
        if valid_every_n_steps is not None and state.num_steps >= valid_every_n_steps + self.last_valid_step:
            self._reset(state)
            return True

        # Time-based validation.
        valid_every_n_seconds = self.valid_every_n_seconds
        if (
            valid_every_n_seconds is not None
            and state.elapsed_time_s.item() - self.last_valid_time >= valid_every_n_seconds
        ):
            self._reset(state)
            return True

        # Time-based validation for first validation step.
        if self.first_valid_step_flag:
            valid_first_n_seconds = self.valid_first_n_seconds
            if valid_first_n_seconds is not None and state.elapsed_time_s.item() >= valid_first_n_seconds:
                self._reset(state)
                self.first_valid_step_flag = False
                return True

        return False


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InitParams:
    key: PRNGKeyArray


# Subclasses should be able to override the init params.
InitParamsT = TypeVar("InitParamsT", bound=InitParams)


@jax.tree_util.register_dataclass
@dataclass
class TrainConfig(
    CheckpointingConfig,
    DataloadersConfig,
    LoggerConfig,
    StepContextConfig,
    ArtifactsConfig,
    RunnableConfig,
):
    valid_every_n_steps: int | None = field(None, help="Number of training steps to run per validation step")
    valid_first_n_steps: int = field(0, help="Treat the first N steps as validation steps")
    valid_every_n_seconds: float | None = field(60.0 * 10.0, help="Run validation every N seconds")
    valid_first_n_seconds: float | None = field(60.0, help="Run first validation after N seconds")
    max_steps: int | None = field(None, help="Maximum number of steps to run")
    step_kind: str = field("step", help=f"How to measure a step; one of [{', '.join(get_args(StepKind))}]")
    precision: Precision = field(Precision.BFLOAT16, help="Precision to use for the task")
    random_seed: int = field(1337, help="Random seed for the task")


Config = TypeVar("Config", bound=TrainConfig)


class TrainMixin(
    CheckpointingMixin[Config],
    DataloadersMixin[Config],
    LoggerMixin[Config],
    StepContextMixin[Config],
    ArtifactsMixin[Config],
    RunnableMixin[Config],
    Generic[Config, InitParamsT],
    ABC,
):
    valid_step_timer: ValidStepTimer
    state_timers: dict[Phase, StateTimer]

    _training_over_flag: bool
    _last_printed_remaining_time: float
    _step_kind: StepKind

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Sets the random seed whenever we instantiate a new train mixin.
        set_random_seed(self.config.random_seed)

        # Timer for validation steps.
        self.valid_step_timer = ValidStepTimer(
            valid_every_n_steps=config.valid_every_n_steps,
            valid_first_n_steps=config.valid_first_n_steps,
            valid_every_n_seconds=config.valid_every_n_seconds,
            valid_first_n_seconds=config.valid_first_n_seconds,
        )

        # Timers for iterations.
        self.state_timers = {phase: StateTimer() for phase in get_args(Phase)}

        # This flag can be toggled to end training from anywhere in the task.
        self._training_over_flag = False

        self._last_printed_remaining_time = 0.0

        # The kind of step that was specified in the config.
        self._step_kind = cast_step_kind(self.config.step_kind)

        # Update Jax configuration based on the precision.
        match self.config.precision:
            case Precision.FLOAT32:
                jax.config.update("jax_default_matmul_precision", "float32")
            case Precision.BFLOAT16:
                jax.config.update("jax_default_matmul_precision", "bfloat16")
            case _:
                raise ValueError(f"Invalid precision: {self.config.precision}")

    def prng_key(self) -> PRNGKeyArray:
        return jax.random.PRNGKey(self.config.random_seed)

    def log_train_step(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        metrics: FrozenDict[str, Array],
        state: State,
    ) -> None:
        """Override this function to do logging during the training phase.

        This function is called after the model forward pass and before the
        backward pass. It is called in the training phase.

        Args:
            model: The current model.
            batch: The batch from the dataloader.
            output: The model output.
            metrics: The metrics for the current batch.
            state: The current training state.
        """

    def log_valid_step(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        metrics: FrozenDict[str, Array],
        state: State,
    ) -> None:
        """Override this function to do logging during the validation phase.

        This function is called after the model forward pass. It is called in
        the validation phase.

        Args:
            model: The current model.
            batch: The batch from the dataloader.
            output: The model output.
            metrics: The metrics for the current batch.
            state: The current training state.
        """

    def log_state_timers(self, state: State) -> None:
        timer = self.state_timers[state.phase]
        timer.step(state)
        for k, v in timer.log_dict().items():
            if isinstance(v, tuple):
                v, secondary = v
            else:
                secondary = False
            self.logger.log_scalar(k, v, namespace="🕒 timers", secondary=secondary)

    def log_step(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        metrics: FrozenDict[str, Array],
        state: State,
    ) -> None:
        phase = state.phase

        for k, v in metrics.items():
            if v.size == 1:
                self.logger.log_scalar(k, v.item())
            else:
                self.logger.log_histogram(k, v)

        self.log_state_timers(state)

        # Delegate to the appropriate logging function based on the phase.
        match phase:
            case "train":
                self.log_train_step(model, batch, output, metrics, state)
            case "valid":
                self.log_valid_step(model, batch, output, metrics, state)
            case _:
                raise KeyError(f"Unknown phase: {phase}")

        self.write_logs(state)

    @abstractmethod
    def get_model(self, params: InitParamsT) -> PyTree | Sequence[PyTree]:
        """Returns the Equinox model to train.

        Args:
            params: The parameters for initializing the model.

        Returns:
            The model to train.
        """

    def _get_models(self, params: InitParamsT) -> list[PyTree]:
        models = self.get_model(params)
        if isinstance(models, Sequence):
            models = list(models)
        elif isinstance(models, eqx.Module):
            models = [models]
        else:
            logger.warning("Model is not a sequence or an eqx.Module, wrapping it in a list anyway")
            models = [models]
        return models

    @abstractmethod
    def get_optimizer(self) -> optax.GradientTransformation | Sequence[optax.GradientTransformation]:
        """Gets the optimizer for the model.

        Returns:
            The optimizer to use to train the model.
        """

    def _get_optimizers(self) -> list[optax.GradientTransformation]:
        optimizers = self.get_optimizer()
        if isinstance(optimizers, optax.GradientTransformation):
            optimizers = [optimizers]
        elif isinstance(optimizers, Sequence):
            optimizers = list(optimizers)
        return optimizers

    def get_initial_opt_state(
        self,
        models: list[PyTree],
        optimizers: list[optax.GradientTransformation],
    ) -> list[optax.OptState]:
        return [opt.init(eqx.filter(model, eqx.is_array)) for model, opt in zip(models, optimizers, strict=True)]

    @overload
    def load_initial_state(
        self,
        params: InitParamsT,
        load_optimizer: Literal[False] = False,
    ) -> tuple[PyTree, State]: ...

    @overload
    def load_initial_state(
        self,
        params: InitParamsT,
        load_optimizer: Literal[True],
    ) -> tuple[list[PyTree], list[optax.GradientTransformation], list[optax.OptState], State]: ...

    def load_initial_state(
        self,
        params: InitParamsT,
        load_optimizer: bool = False,
    ) -> (
        tuple[list[PyTree], State]
        | tuple[list[PyTree], list[optax.GradientTransformation], list[optax.OptState], State]
    ):
        init_ckpt_path = self.get_init_ckpt_path()

        if init_ckpt_path is not None:
            logger.info("Loading checkpoint from %s", init_ckpt_path)
            model, state, config = self.load_ckpt(init_ckpt_path, params, part="model_state_config")
            config_diff = get_diff_string(diff_configs(asdict(config), asdict(self.config)))
            if config_diff:
                logger.warning("Loaded config differs from current config:\n%s", config_diff)

            if not load_optimizer:
                return model, state

            optimizer = self.load_ckpt(init_ckpt_path, params, part="opt")
            opt_state = self.load_ckpt(init_ckpt_path, params, part="opt_state", model=model, optimizer=optimizer)
            return model, optimizer, opt_state, state

        logger.info("Starting a new training run")
        models = self._get_models(params)
        state = State.init_state()

        if not load_optimizer:
            return models, state

        # Gets the optimizer(s) for the model.
        optimizers = self._get_optimizers()
        opt_states = self.get_initial_opt_state(models, optimizers)

        return models, optimizers, opt_states, state

    @overload
    def load_ckpt(
        self,
        path: Path,
        init_params: InitParamsT,
        *,
        part: Literal["all"],
    ) -> tuple[list[PyTree], list[optax.GradientTransformation], list[optax.OptState], State, Config]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        init_params: InitParamsT,
        *,
        part: Literal["model_state_config"],
    ) -> tuple[list[PyTree], State, Config]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        init_params: InitParamsT,
        *,
        part: Literal["model"],
    ) -> list[PyTree]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        init_params: InitParamsT,
        *,
        part: Literal["opt"],
    ) -> list[optax.GradientTransformation]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        init_params: InitParamsT,
        *,
        part: Literal["opt_state"],
        model: PyTree | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ) -> list[optax.OptState]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        init_params: InitParamsT,
        *,
        part: Literal["state"],
    ) -> list[State]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        init_params: InitParamsT,
        *,
        part: Literal["config"],
    ) -> list[Config]: ...

    def load_ckpt(
        self,
        path: str | Path,
        init_params: InitParamsT,
        *,
        part: CheckpointPart = "all",
        model: PyTree | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ) -> (
        tuple[list[PyTree], list[optax.GradientTransformation], list[optax.OptState], State, Config]
        | tuple[list[PyTree], State, Config]
        | list[PyTree]
        | list[optax.GradientTransformation]
        | list[optax.OptState]
        | State
        | Config
    ):
        path = Path(path)

        match part:
            case "model_state_config":
                model_specs = eqx.filter_eval_shape(self._get_models, init_params)
                model, state, config = load_ckpt(path, part="model_state_config", model_templates=model_specs)
                config = self.get_config(config, use_cli=False)
                return model, state, config

            case "model":
                model_specs = eqx.filter_eval_shape(self._get_models, init_params)
                return load_ckpt(path, part="model", model_templates=model_specs)

            case "opt":
                optimizer_specs = eqx.filter_eval_shape(self._get_optimizers)
                return load_ckpt(path, part="opt", optimizer_templates=optimizer_specs)

            case "opt_state":
                if model is None:
                    model_specs = eqx.filter_eval_shape(self._get_models, init_params)
                    model = load_ckpt(path, part="model", model_templates=model_specs)
                if optimizer is None:
                    optimizer_specs = eqx.filter_eval_shape(self._get_optimizers)
                    optimizer = load_ckpt(path, part="opt", optimizer_templates=optimizer_specs)
                opt_state_specs = eqx.filter_eval_shape(self.get_initial_opt_state, model, optimizer)
                return load_ckpt(path, part="opt_state", opt_state_templates=opt_state_specs)

            case "state":
                return load_ckpt(path, part="state")

            case "config":
                return self.get_config(load_ckpt(path, part="config"), use_cli=False)

            case "all":
                model_specs = eqx.filter_eval_shape(self._get_models, init_params)
                model = load_ckpt(path, part="model", model_templates=model_specs)
                optimizer_specs = eqx.filter_eval_shape(self._get_optimizers)
                optimizer = load_ckpt(path, part="opt", optimizer_templates=optimizer_specs)
                opt_state_specs = eqx.filter_eval_shape(self.get_initial_opt_state, model, optimizer)
                opt_state = load_ckpt(path, part="opt_state", opt_state_templates=opt_state_specs)
                state = load_ckpt(path, part="state")
                config = self.get_config(load_ckpt(path, part="config"), use_cli=False)
                return model, optimizer, opt_state, state, config

            case _:
                raise ValueError(f"Unknown checkpoint part: {part}")

    def get_size_of_batch(self, batch: Batch) -> int | None:
        """Gets the batch size for the current batch.

        Args:
            batch: The current minibatch of samples.

        Returns:
            The parsed batch size, or None if the batch size could not be
            determined.
        """
        if isinstance(batch, (np.ndarray, Array)):
            return batch.shape[0]
        if is_dataclass(batch):
            for v in batch.__dict__.values():
                if bsz := self.get_size_of_batch(v):
                    return bsz
        if isinstance(batch, Mapping):
            for v in batch.values():
                if bsz := self.get_size_of_batch(v):
                    return bsz
        if isinstance(batch, Sequence):
            for i in batch:
                if bsz := self.get_size_of_batch(i):
                    return bsz
        return None

    def set_training_over(self) -> None:
        self._training_over_flag = True

    def maybe_log_termination_time(self, remaining_percent: float, state: State) -> None:
        if self._last_printed_remaining_time + PRINT_FINISH_TIME_EVERY_N_SECONDS > state.elapsed_time_s:
            return
        self._last_printed_remaining_time = state.elapsed_time_s.item()
        remaining_seconds = remaining_percent * state.elapsed_time_s.item() / (1 - remaining_percent)
        termination_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
        logger.log(LOG_PING, "Estimated finish time: %s", termination_time)

    def get_remaining_percent(self, state: State) -> float | None:
        if self.config.max_steps is None:
            return None
        return (self.config.max_steps - self.get_step(state)) / self.config.max_steps

    def is_training_over(self, state: State) -> bool:
        if self._training_over_flag:
            return True
        remaining_percent = self.get_remaining_percent(state)
        if remaining_percent is None:
            return False
        self.maybe_log_termination_time(remaining_percent, state)
        return remaining_percent <= 0.0

    def get_step(self, state: State) -> int:
        match self._step_kind:
            case "step":
                return int(state.num_steps.item())
            case "sample":
                return int(state.num_samples.item())
            case "second":
                return int(state.elapsed_time_s.item())
            case _:
                raise ValueError(f"Invalid step kind {self._step_kind}")

    def log_state(self) -> None:
        logger.log(LOG_STATUS, self.task_path)
        logger.log(LOG_STATUS, self.exp_dir)
        logger.log(LOG_STATUS, "JAX devices: %s", jax.devices())
        self.logger.log_file("state.txt", get_state_file_string(self))
        self.logger.log_file("training_code.py", get_training_code(self))
        self.logger.log_file("config.yaml", self.config_str(self.config, use_cli=False))
        self.logger.log_file("info.json", get_info_json())

    def model_partition_fn(self, item: Any) -> bool:  # noqa: ANN401
        return eqx.is_inexact_array(item)
