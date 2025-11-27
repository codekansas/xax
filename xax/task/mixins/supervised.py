"""Defines a mixin for running the supervised training loop."""

import bdb
import itertools
import logging
import signal
import sys
import textwrap
import traceback
from abc import ABC
from dataclasses import dataclass
from threading import Thread
from typing import (
    Generic,
    Iterator,
    Sequence,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray, PyTree

from xax.core.conf import field
from xax.core.state import State
from xax.nn.parallel import is_master
from xax.task.mixins.data_loader import iter_samples
from xax.task.mixins.train import Batch, InitParams, Output, TrainConfig, TrainMixin
from xax.utils.experiments import (
    ContextTimer,
    TrainingFinishedError,
)
from xax.utils.jax import jit as xax_jit, scan as xax_scan
from xax.utils.logging import LOG_PING
from xax.utils.pytree import get_pytree_param_count
from xax.utils.text import highlight_exception_message, show_info
from xax.utils.types.frozen_dict import FrozenDict

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class SupervisedConfig(TrainConfig):
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
        loss: Array,
        state: State,
    ) -> dict[str, Array]:
        """Computes the metrics for the current batch.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            output: The output from the model.
            loss: The loss for the current batch.
            state: The current training state.

        Returns:
            A dictionary of metrics.
        """
        return {
            "loss": loss,
        }

    @xax_jit(static_argnames=["self", "model_static"], jit_level=3)
    def get_output_and_loss(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        batch: Batch,
        state: State,
        key: PRNGKeyArray,
    ) -> tuple[Array, tuple[Output, dict[str, Array]]]:
        output_key, loss_key = jax.random.split(key)
        model = eqx.combine(model_arr, model_static)
        output = self.get_output(model, batch, state, output_key)
        loss = self.compute_loss(model, batch, output, state, loss_key)
        metrics = self.compute_metrics(model, batch, output, loss, state)
        return loss, (output, metrics)

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
    ) -> tuple[PyTree, optax.OptState, Output, dict[str, Array]]:
        grad_fn = jax.grad(self.get_output_and_loss, argnums=0, has_aux=True)
        grad_fn = xax_jit(static_argnums=[1], jit_level=3)(grad_fn)
        grads, (output, metrics) = grad_fn(model_arr, model_static, batch, state, key)
        grad_norm = optax.global_norm(grads)
        if self.config.max_grad_norm is not None:
            clip_fn = optax.clip_by_global_norm(self.config.max_grad_norm)
            grads, _ = clip_fn.update(grads, None)

        updates, opt_state = optimizer.update(grads, opt_state, model_arr)
        model_arr = eqx.apply_updates(model_arr, updates)

        # Add gradient norm to metrics
        metrics = dict(metrics)
        metrics["grad_norm"] = grad_norm

        return model_arr, opt_state, output, metrics

    @xax_jit(
        static_argnames=["self", "model_static", "optimizer"],
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
        key: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, Output, FrozenDict[str, Array], State]:
        def update_fn(
            carry: tuple[PyTree, optax.OptState, State],
            batch: Batch,
        ) -> tuple[tuple[PyTree, optax.OptState, State], tuple[Output, FrozenDict[str, Array]]]:
            model_arr, opt_state, state = carry
            model_arr, opt_state, output, metrics = self.update(
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
            return (model_arr, opt_state, state), (output, FrozenDict(metrics))

        (model_arr, opt_state, state), (output, metrics) = xax_scan(
            update_fn,
            (model_arr, opt_state, state),
            jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *batches),
            jit_level=3,
        )

        # Only get the final output and metrics.
        output = jax.tree.map(lambda x: x[-1], output)
        metrics = jax.tree.map(lambda x: x[-1], metrics)

        return model_arr, opt_state, output, metrics, state

    def train_loop(
        self,
        models: Sequence[PyTree],
        optimizers: Sequence[optax.GradientTransformation],
        opt_states: Sequence[optax.OptState],
        ds: Iterator[Batch],
        state: State,
        key: PRNGKeyArray,
    ) -> None:
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
                batches = tuple(itertools.islice(ds, self.config.updates_per_step))
                step_key, log_key, key = jax.random.split(key, 3)
                model_arr, opt_state, output, metrics, state = self.train_step(
                    model_arr=model_arr,
                    model_static=model_static,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    batches=batches,
                    state=state,
                    key=step_key,
                )
                state = self.log_step(
                    eqx.combine(model_arr, model_static),
                    batches[-1],
                    output,
                    metrics,
                    state,
                    log_key,
                )
                self.on_step_end()

            state = state.replace(elapsed_time_s=state.elapsed_time_s + timer.elapsed_time)

            if state.num_steps <= 3:
                logger.log(LOG_PING, "Step %d took %.2f second", state.num_steps, timer.elapsed_time)

            if self.should_checkpoint(state):
                model = eqx.combine(model_arr, model_static)
                self.save_checkpoint(models=[model], optimizers=[optimizer], opt_states=[opt_state], state=state)

        # After finishing training, save the final checkpoint.
        model = eqx.combine(model_arr, model_static)
        self.save_checkpoint(models=[model], optimizers=[optimizer], opt_states=[opt_state], state=state)

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

            def on_exit() -> None:
                self.save_checkpoint(models=models, optimizers=optimizers, opt_states=opt_states, state=state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            ds = self.get_data_iterator()
            ds = iter_samples(ds, data_sharding)

            try:
                self.train_loop(
                    models=models,
                    optimizers=optimizers,
                    opt_states=opt_states,
                    ds=ds,
                    state=state,
                    key=key,
                )

            except TrainingFinishedError:
                if is_master():
                    num_steps, num_samples = int(state.num_steps), int(state.num_samples)
                    show_info(f"Finished training after {num_steps} steps, {num_samples} samples", important=True)
                self.save_checkpoint(models=models, optimizers=optimizers, opt_states=opt_states, state=state)

            except (KeyboardInterrupt, bdb.BdbQuit):
                if is_master():
                    show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(models=models, optimizers=optimizers, opt_states=opt_states, state=state)
                raise

            finally:
                self.on_training_end()
