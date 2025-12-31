"""Defines a mixin for running the supervised training loop."""

import itertools
import logging
import signal
from abc import ABC
from dataclasses import dataclass
from threading import Thread
from typing import (
    Generic,
    Iterator,
    Sequence,
    TypeVar,
    cast,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray, PyTree

from xax.core.conf import field
from xax.core.state import Batch, Output, State
from xax.nn.parallel import is_master
from xax.task.logger import Metric, Scalar
from xax.task.mixins.data_loader import iter_samples
from xax.task.mixins.train import InitParams, Optimizer, TrainConfig, TrainMixin
from xax.utils.experiments import ContextTimer
from xax.utils.jax import jit as xax_jit, scan as xax_scan
from xax.utils.logging import LOG_PING
from xax.utils.pytree import get_pytree_param_count
from xax.utils.text import show_info
from xax.utils.types.frozen_dict import FrozenDict
from xax.utils.types.training import TrainingState

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class SupervisedConfig(TrainConfig):
    batches_per_step: int = field(1, help="Number of batches to process per step")
    updates_per_step: int = field(1, help="Number of updates to perform per step")
    max_grad_norm: float | None = field(None, help="Clip gradient norm to this value.")


Config = TypeVar("Config", bound=SupervisedConfig)


class SupervisedMixin(
    TrainMixin[Config, InitParams],
    Generic[Config],
    ABC,
):
    def get_output(self, model: PyTree, batch: Batch, state: State, key: PRNGKeyArray) -> Output:
        """Gets the output from the model.

        By default, we assume the model is a function that takes the batch as
        input and returns the loss. This function can be patched to do more
        complex operations instead.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            state: The current training state.
            key: The current PRNG key.

        Returns:
            The output from the model.
        """
        raise NotImplementedError("`get_output` must be implemented by the subclass")

    def compute_loss(self, model: PyTree, batch: Batch, output: Output, state: State, key: PRNGKeyArray) -> Array:
        """Gets the loss for the current batch.

        By default, we assume the model is a function that takes the batch as
        input and returns the loss. This function can be patched to do more
        complex operations instead.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            output: The output from the model.
            state: The current training state.
            key: The current PRNG key.

        Returns:
            The computed loss, as a tensor.
        """
        if not isinstance(output, Array):
            raise ValueError(f"When model output is not the loss, you must override `compute_loss`. Got {type(output)}")
        return output

    def compute_metrics(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        state: State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> dict[str, Metric]:
        """Computes the metrics for the current batch.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            output: The output from the model.
            state: The current training state.
            heavy: If we should be logging heavy metrics.
            key: The current PRNG key.

        Returns:
            A dictionary of metrics.
        """
        return {}

    @xax_jit(static_argnames=["self", "model_static"], jit_level=3)
    def get_output_and_loss(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        batch: Batch,
        state: State,
        key: PRNGKeyArray,
    ) -> tuple[Array, Output]:
        output_key, loss_key = jax.random.split(key)
        model = eqx.combine(model_arr, model_static)
        output = self.get_output(model, batch, state, output_key)
        loss = self.compute_loss(model, batch, output, state, loss_key)
        return loss, (loss, output)

    @xax_jit(
        static_argnames=["self", "model_static", "optimizer"],
        donate_argnames=["model_arr", "opt_state", "batch", "state", "key"],
        jit_level=3,
    )
    def update(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: Batch,
        state: State,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, Output, Array, Array]:
        grad_fn = jax.grad(self.get_output_and_loss, argnums=0, has_aux=True)
        grad_fn = xax_jit(static_argnums=[1], jit_level=3)(grad_fn)
        grads, aux = grad_fn(model_arr, model_static, batch, state, key)
        loss, output = cast(tuple[Array, Output], aux)
        grad_norm = optax.global_norm(grads)
        if self.config.max_grad_norm is not None:
            clip_fn = optax.clip_by_global_norm(self.config.max_grad_norm)
            clip_state = clip_fn.init(grads)  # Fine since this function is init_empty_state
            grads, _ = clip_fn.update(grads, clip_state)

        updates, opt_state = optimizer.update(grads, opt_state, model_arr)
        model_arr = eqx.apply_updates(model_arr, updates)

        return model_arr, opt_state, output, loss, grad_norm  # type: ignore[return-value]

    def decode_tokens(self, tokens: Array) -> str:
        raise NotImplementedError(
            "When using a Tokens metric you must implement the `decode_tokens` method "
            "to convert to a string which can be logged."
        )

    def log_step(self, metrics: FrozenDict[str, Metric], state: State, heavy: bool) -> None:
        for k, v in metrics.items():
            try:
                self.logger.log_metric(k, v, decode_tokens=self.decode_tokens)
            except Exception as e:
                raise ValueError(f"Error logging metric {k}") from e
        self.log_state_timers(state)
        self.write_logs(state, heavy)

    @xax_jit(
        static_argnames=["self", "model_static", "optimizer", "heavy"],
        donate_argnames=["model_arr", "opt_state", "state", "key"],
        jit_level=3,
    )
    def train_step(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batches: tuple[Batch, ...],
        state: State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Metric], State]:
        @xax_jit(donate_argnames=["carry"])
        def update_fn(
            carry: tuple[PyTree, optax.OptState, State, PRNGKeyArray],
            batch: Batch,
        ) -> tuple[tuple[PyTree, optax.OptState, State, PRNGKeyArray], tuple[Output, Array, Array]]:
            model_arr, opt_state, state, key = carry
            model_arr, opt_state, output, loss, grad_norm = self.update(
                model_arr,
                model_static,
                optimizer,
                opt_state,
                batch,
                state,
                key,
            )
            state = state.replace(
                num_steps=state.num_steps + 1,
                num_samples=state.num_samples + (self.get_size_of_batch(batch) or 0),
            )
            return (model_arr, opt_state, state, key), (output, loss, grad_norm)

        batches_stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *batches)

        @xax_jit(donate_argnames=["carry"])
        def update_batch_fn(
            carry: tuple[PyTree, optax.OptState, State, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, optax.OptState, State, PRNGKeyArray], tuple[Output, Array, Array]]:
            model_arr, opt_state, state, key = carry

            (model_arr, opt_state, state, key), (output, loss, grad_norm) = xax_scan(
                update_fn,
                (model_arr, opt_state, state, key),
                batches_stacked,
                jit_level=3,
            )

            return (model_arr, opt_state, state, key), (output, loss, grad_norm)

        update_key, key = jax.random.split(key)
        (model_arr, opt_state, state, _), (output, loss, grad_norm) = xax_scan(
            update_batch_fn,
            (model_arr, opt_state, state, update_key),
            length=self.config.updates_per_step,
            jit_level=3,
        )

        # Compute metrics for the final batch and output.
        batch = batches[-1]
        output = jax.tree.map(lambda x: x[-1, -1], output)

        model = eqx.combine(model_arr, model_static)
        metrics = self.compute_metrics(model, batch, output, state, heavy, key)

        # Adds loss and gradient norm to metrics.
        metrics["loss"] = Scalar(loss.mean())
        metrics["grad_norm"] = Scalar(grad_norm.mean())

        return model_arr, opt_state, FrozenDict(metrics), state

    def train_loop(
        self,
        training_state: TrainingState,
        optimizers: Sequence[Optimizer],
        ds: Iterator[Batch],
        key: PRNGKeyArray,
    ) -> None:
        models = training_state.models
        opt_states = training_state.opt_states
        state = training_state.state

        if len(models) != 1 or len(optimizers) != 1 or len(opt_states) != 1:
            raise ValueError(
                "Vanilla training expects a single model, optimizer and optimizer state. "
                f"Found {len(models)} models, {len(optimizers)} optimizers and {len(opt_states)} optimizer states."
            )

        model_arr, model_static = eqx.partition(models[0], self.model_partition_fn)
        optimizer = optimizers[0]
        opt_state = opt_states[0]

        while not self.is_training_over(state):
            with ContextTimer() as timer:
                self.on_step_start()
                batches = tuple(itertools.islice(ds, self.config.batches_per_step))
                state, heavy_arr = self.get_log_mode(state)
                heavy = heavy_arr.item()
                step_key, key = jax.random.split(key)
                model_arr, opt_state, metrics, state = self.train_step(
                    model_arr=model_arr,
                    model_static=model_static,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    batches=batches,
                    state=state,
                    heavy=heavy,
                    key=step_key,
                )
                self.log_step(metrics, state, heavy)
                self.on_step_end()

            state = state.replace(elapsed_time_s=state.elapsed_time_s + timer.elapsed_time)

            # Update the state dataclass so that the calling function can
            # access it in the event of an exception.
            model = eqx.combine(model_arr, model_static)
            training_state.models[0] = model
            training_state.opt_states[0] = opt_state
            training_state.state = state

            if state.num_steps <= 3:
                logger.log(LOG_PING, "Step %d took %.2f second", state.num_steps, timer.elapsed_time)

            if self.should_checkpoint(state):
                self.save_checkpoint(
                    models=training_state.models,
                    optimizers=optimizers,
                    opt_states=training_state.opt_states,
                    state=training_state.state,
                )

        # After finishing training, save the final checkpoint.
        self.save_checkpoint(
            models=training_state.models,
            optimizers=optimizers,
            opt_states=training_state.opt_states,
            state=training_state.state,
        )

    def run(self) -> None:
        self.run_training()

    def get_sharding(self) -> NamedSharding:
        ndevices = jax.local_device_count()
        mesh = jax.make_mesh((ndevices,), axis_names=("batch",))
        return jax.sharding.NamedSharding(mesh, P("batch"))

    def run_training(self) -> None:
        """Runs the training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler

        Raises:
            ValueError: If the task is not a supervised learning task
        """
        with self:
            key = self.prng_key()

            self.set_loggers()

            if is_master():
                Thread(target=self.log_state, daemon=True).start()

            # Gets the sharding strategy.
            mesh = self.get_mesh()
            data_sharding = self.get_data_sharding(mesh)
            model_sharding = self.get_model_sharding(mesh)

            model_key, key = jax.random.split(key)
            init_params = InitParams(key=model_key)
            models, optimizers, opt_states, state = self.load_initial_state(
                init_params,
                load_optimizer=True,
                model_sharding=model_sharding,
            )

            logger.info("Model size: %s", f"{get_pytree_param_count(models):,}")
            logger.info("Optimizer size: %s", f"{get_pytree_param_count(opt_states):,}")

            self.on_training_start()

            latest_state = TrainingState(models, opt_states, state)

            def on_exit() -> None:
                self.save_checkpoint(
                    models=latest_state.models,
                    optimizers=optimizers,
                    opt_states=latest_state.opt_states,
                    state=latest_state.state,
                )

            # Handle user-defined interrupts during the training loop, like
            # when the Slurm job gets preempted.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            ds = self.get_tf_dataset()
            ds = iter_samples(ds, data_sharding)

            self.train_loop(
                training_state=latest_state,
                optimizers=optimizers,
                ds=ds,
                key=key,
            )

            if is_master():
                num_steps, num_samples = int(latest_state.state.num_steps), int(latest_state.state.num_samples)
                show_info(f"Finished training after {num_steps} steps, {num_samples} samples", important=True)

            self.save_checkpoint(
                models=latest_state.models,
                optimizers=optimizers,
                opt_states=latest_state.opt_states,
                state=latest_state.state,
            )

            self.on_training_end()
