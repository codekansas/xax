"""Defines a mixin for running the training loop."""

import functools
import itertools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    get_args,
    overload,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from xax.core.conf import field
from xax.core.state import Batch, State, StepKind, cast_step_kind
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
from xax.task.mixins.parallel import ParallelConfig, ParallelMixin
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
from xax.utils.jax import jit as xax_jit
from xax.utils.logging import LOG_PING, LOG_STATUS
from xax.utils.types.training import Optimizer, PrecisionConfig, as_shape_dtype

logger = logging.getLogger(__name__)

PRINT_FINISH_TIME_EVERY_N_SECONDS = 60 * 2


@functools.lru_cache(maxsize=None)
def batch_chunks_schedule(schedule: list[int] | None) -> list[int] | None:
    if schedule is None:
        return None
    if any(s < 1 for s in schedule):
        raise ValueError("Batch chunk schedule must be positive")
    return list(itertools.accumulate([0] + schedule))


@functools.lru_cache(maxsize=None)
def gradient_accumulation_schedule(schedule: list[int] | None) -> list[int] | None:
    if schedule is None:
        return None
    if any(s < 1 for s in schedule):
        raise ValueError("Gradient accumulation schedule must be positive")
    return list(itertools.accumulate([0] + schedule))


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
    ParallelConfig,
):
    log_heavy_first_n_steps: int = field(1, help="Log heavy metrics for the first N steps")
    log_heavy_every_n_seconds: int = field(60 * 5, help="Log heavy metrics every N seconds")
    max_steps: int | None = field(None, help="Maximum number of steps to run")
    step_kind: str = field("step", help=f"How to measure a step; one of [{', '.join(get_args(StepKind))}]")
    precision: PrecisionConfig = field(PrecisionConfig, help="Config specifying floating point precisions")
    random_seed: int = field(1337, help="Random seed for the task")


Config = TypeVar("Config", bound=TrainConfig)


class TrainMixin(
    ParallelMixin[Config],
    CheckpointingMixin[Config],
    DataloadersMixin[Config],
    LoggerMixin[Config],
    StepContextMixin[Config],
    ArtifactsMixin[Config],
    RunnableMixin[Config],
    Generic[Config, InitParamsT],
    ABC,
):
    state_timer: StateTimer

    _training_over_flag: bool
    _last_printed_remaining_time: float
    _step_kind: StepKind

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Sets the random seed whenever we instantiate a new train mixin.
        set_random_seed(self.config.random_seed)

        # Timers for iterations.
        self.state_timer = StateTimer()

        # This flag can be toggled to end training from anywhere in the task.
        self._training_over_flag = False

        self._last_printed_remaining_time = 0.0

        # The kind of step that was specified in the config.
        self._step_kind = cast_step_kind(self.config.step_kind)

        jax.config.update("jax_default_matmul_precision", self.config.precision.compute_dtype.value)

    @property
    def precision_config(self) -> PrecisionConfig:
        """Returns the precision configuration for this task."""
        return self.config.precision

    def cast_data_dtype(self, x: Any) -> Array:  # noqa: ANN401
        """Cast data to the configured data dtype."""
        if isinstance(x, Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(self.config.precision.data_jax_dtype)
        return x

    def cast_param_dtype(self, x: Any) -> Array:  # noqa: ANN401
        """Cast parameters to the configured param dtype."""
        if isinstance(x, Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(self.config.precision.param_jax_dtype)
        return x

    def cast_compute_dtype(self, x: Any) -> Array:  # noqa: ANN401
        """Cast values to the configured compute dtype."""
        if isinstance(x, Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(self.config.precision.compute_jax_dtype)
        return x

    def prng_key(self) -> PRNGKeyArray:
        return jax.random.key(self.config.random_seed)

    def log_state_timers(self, state: State) -> None:
        timer = self.state_timer
        timer.step(state)
        for k, v in timer.log_dict().items():
            if isinstance(v, tuple):
                v, secondary = v
            else:
                secondary = False
            self.logger.log_scalar(k, v, namespace="🕒 timers", secondary=secondary)

    @xax_jit(static_argnames=["self"], donate_argnames=["state"], jit_level=3)
    def get_log_mode(self, state: State) -> tuple[State, Array]:
        log_heavy = jnp.where(
            state.num_steps <= self.config.log_heavy_first_n_steps,
            True,
            state.elapsed_time_s - state.last_log_time_s >= self.config.log_heavy_every_n_seconds,
        )
        last_log_time_s = jnp.where(log_heavy, state.elapsed_time_s, state.last_log_time_s)
        return state.replace(last_log_time_s=last_log_time_s), log_heavy

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
    def get_optimizer(self) -> Optimizer | Sequence[Optimizer]:
        """Gets the optimizer for the model.

        Returns:
            The optimizer to use to train the model.
        """

    def _get_optimizers(self) -> list[Optimizer]:
        optimizers = self.get_optimizer()
        if isinstance(optimizers, Optimizer):
            optimizers = [optimizers]
        elif isinstance(optimizers, Sequence):
            optimizers = list(optimizers)
        for opt in optimizers:
            if not isinstance(opt, Optimizer):
                raise ValueError(f"Optimizer {opt} is not a valid optimizer")
        return optimizers

    def get_initial_opt_state(
        self,
        models: list[PyTree],
        optimizers: list[Optimizer],
    ) -> list[optax.OptState]:
        # Cast model params to grad_dtype for optimizer initialization
        # This ensures gradient accumulation happens in grad_dtype precision
        grad_dtype = self.config.precision.grad_jax_dtype

        def cast_to_grad_dtype(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, Array) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(grad_dtype)
            return x

        opt_states = []
        for model, opt in zip(models, optimizers, strict=True):
            # Use filter spec to only create optimizer state for trainable params
            filter_spec = self.get_model_filter_spec(model)
            trainable, _ = eqx.partition(model, filter_spec)
            trainable_casted = jax.tree.map(cast_to_grad_dtype, trainable)
            opt_states.append(opt.init(trainable_casted))
        return opt_states

    @overload
    def load_initial_state(
        self,
        params: InitParamsT,
        load_optimizer: Literal[False] = False,
        model_sharding: jax.sharding.NamedSharding | None = None,
    ) -> tuple[PyTree, State]: ...

    @overload
    def load_initial_state(
        self,
        params: InitParamsT,
        load_optimizer: Literal[True],
        model_sharding: jax.sharding.NamedSharding | None = None,
    ) -> tuple[list[PyTree], list[Optimizer], list[optax.OptState], State]: ...

    def load_initial_state(
        self,
        params: InitParamsT,
        load_optimizer: bool = False,
        model_sharding: jax.sharding.NamedSharding | None = None,
    ) -> tuple[list[PyTree], State] | tuple[list[PyTree], list[Optimizer], list[optax.OptState], State]:
        init_ckpt_path = self.get_init_ckpt_path()

        def _shard(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, Array):
                return jax.device_put(x, src=model_sharding)
            return x

        if init_ckpt_path is not None:
            logger.info("Loading checkpoint from %s", init_ckpt_path)
            models, state, config = self.load_ckpt(init_ckpt_path, params, part="model_state_config")
            if model_sharding is not None:
                models = jax.tree.map(_shard, models)
            config_diff = get_diff_string(diff_configs(asdict(config), asdict(self.config)))
            if config_diff:
                logger.warning("Loaded config differs from current config:\n%s", config_diff)

            if not load_optimizer:
                return models, state

            optimizers = self._get_optimizers()
            opt_states = self.load_ckpt(init_ckpt_path, params, part="opt_state", models=models, optimizers=optimizers)
            return models, optimizers, opt_states, state

        logger.info("Starting a new training run")
        models = self._get_models(params)
        state = State.init_state()

        # Casts the model to the desired dtype.
        models = jax.tree.map(self.cast_param_dtype, models)

        if model_sharding is not None:
            models = jax.tree.map(_shard, models)

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
    ) -> tuple[list[PyTree], list[optax.OptState], State, Config]: ...

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
        part: Literal["opt_state"],
        models: list[PyTree] | None = None,
        optimizers: list[Optimizer] | None = None,
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
        models: list[PyTree] | None = None,
        optimizers: list[Optimizer] | None = None,
    ) -> (
        tuple[list[PyTree], list[optax.OptState], State, Config]
        | tuple[list[PyTree], State, Config]
        | list[PyTree]
        | list[optax.OptState]
        | State
        | Config
    ):
        path = Path(path)

        match part:
            case "model_state_config":
                if models is None:
                    models_shape = eqx.filter_eval_shape(self._get_models, init_params)
                    models_specs = jax.tree_util.tree_map(as_shape_dtype, models_shape)
                else:
                    models_specs = jax.tree_util.tree_map(as_shape_dtype, models)
                models, state, config = load_ckpt(path, part="model_state_config", model_templates=models_specs)
                config = self.get_config(config, use_cli=False)
                return models, state, config

            case "model":
                model_shape = eqx.filter_eval_shape(self._get_models, init_params)
                model_specs = jax.tree_util.tree_map(as_shape_dtype, model_shape)
                return load_ckpt(path, part="model", model_templates=model_specs)

            case "opt_state":
                if models is None:
                    model_shape = eqx.filter_eval_shape(self._get_models, init_params)
                    model_specs = jax.tree_util.tree_map(as_shape_dtype, model_shape)
                    models = load_ckpt(path, part="model", model_templates=model_specs)
                if optimizers is None:
                    optimizers = self._get_optimizers()
                opt_state_specs = eqx.filter_eval_shape(self.get_initial_opt_state, models, optimizers)
                return load_ckpt(path, part="opt_state", opt_state_templates=opt_state_specs)

            case "state":
                return load_ckpt(path, part="state")

            case "config":
                return self.get_config(load_ckpt(path, part="config"), use_cli=False)

            case "all":
                if models is None:
                    models_shape = eqx.filter_eval_shape(self._get_models, init_params)
                    models_specs = jax.tree_util.tree_map(as_shape_dtype, models_shape)
                    models = load_ckpt(path, part="model", model_templates=models_specs)
                if optimizers is None:
                    optimizers = self._get_optimizers()
                opt_state_specs = eqx.filter_eval_shape(self.get_initial_opt_state, models, optimizers)
                opt_states = load_ckpt(path, part="opt_state", opt_state_templates=opt_state_specs)
                state = load_ckpt(path, part="state")
                config = self.get_config(load_ckpt(path, part="config"), use_cli=False)
                return models, opt_states, state, config

            case _:
                raise ValueError(f"Unknown checkpoint part: {part}")

    def get_size_of_batch(self, batch: Batch, index: int = 0) -> int | None:
        """Gets the batch size for the current batch.

        Args:
            batch: The current minibatch of samples.
            index: The index of the batch to get the size of.

        Returns:
            The parsed batch size, or None if the batch size could not be
            determined.
        """
        if isinstance(batch, (np.ndarray, Array)) and len(batch.shape) > index:
            return batch.shape[index]
        if is_dataclass(batch):
            for v in batch.__dict__.values():
                if bsz := self.get_size_of_batch(v, index):
                    return bsz
        if isinstance(batch, Mapping):
            for v in batch.values():
                if bsz := self.get_size_of_batch(v, index):
                    return bsz
        if isinstance(batch, Sequence):
            for i in batch:
                if bsz := self.get_size_of_batch(i, index):
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

    def get_model_filter_spec(self, model: PyTree) -> PyTree | Callable[[Any], bool]:
        """Returns a filter spec for partitioning the model into trainable/frozen parts.

        Override this method to customize which parameters are trainable.
        For LoRA fine-tuning, return a pytree filter spec marking only LoRA params.

        Args:
            model: The model to partition.

        Returns:
            Either a callable (applied to each leaf) or a pytree of booleans.
        """
        return self.model_partition_fn
