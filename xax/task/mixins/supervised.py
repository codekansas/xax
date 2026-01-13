"""Defines a mixin for running the supervised training loop."""

import functools
import itertools
import logging
import os
import signal
import sys
import traceback
from abc import ABC
from dataclasses import dataclass
from threading import Thread
from typing import (
    Callable,
    Generic,
    Iterator,
    Sequence,
    TypeVar,
    cast,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, PRNGKeyArray, PyTree

from xax.core.conf import field
from xax.core.state import Batch, Output, State
from xax.nn.parallel import is_master
from xax.task.logger import Metric, Scalar
from xax.task.mixins.data_loader import InMemoryBatchIterator
from xax.task.mixins.train import InitParams, Optimizer, TrainConfig, TrainMixin
from xax.utils.experiments import ContextTimer
from xax.utils.jax import fix_unspecified_sharding, jit
from xax.utils.logging import LOG_PING, LOG_STATUS
from xax.utils.pytree import get_pytree_param_count
from xax.utils.text import show_info
from xax.utils.types.frozen_dict import FrozenDict

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class SupervisedConfig(TrainConfig):
    gradient_accumulation_steps: int = field(1, help="Micro-batches to accumulate before sync")
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
            state: The current training state, or None during pmap execution.
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
            state: The current training state, or None during pmap execution.
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

    def decode_tokens(self, tokens: Array | np.ndarray) -> str:
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

    def create_train_step_fn(
        self,
        model_static: PyTree,
        optimizer: Optimizer,
    ) -> Callable[
        [PyTree, optax.OptState, Batch, State, PRNGKeyArray, bool],
        tuple[PyTree, optax.OptState, State, dict[str, Metric]],
    ]:
        """Create a JIT-compiled training step function.

        Args:
            model_static: The static (non-trainable) parts of the model.
            optimizer: The optimizer.

        Returns:
            A JIT-compiled function that performs one training step.
            The function takes a `heavy` bool argument (static) to control
            whether heavy metrics are computed.
        """
        num_accum_steps = self.config.gradient_accumulation_steps
        grad_dtype = self.config.precision.grad_jax_dtype
        param_dtype = self.config.precision.param_jax_dtype
        max_grad_norm = self.config.max_grad_norm

        # Capture methods for use in inner function
        get_output = self.get_output
        compute_loss = self.compute_loss
        compute_metrics = self.compute_metrics
        cast_compute_dtype = self.cast_compute_dtype

        def train_step(
            model_arr: PyTree,
            opt_state: optax.OptState,
            batches: Batch,
            state: State,
            key: PRNGKeyArray,
            heavy: bool,
        ) -> tuple[PyTree, optax.OptState, State, dict[str, Metric]]:
            def compute_grads_and_loss(
                model_arr: PyTree, batch: Batch, step_key: PRNGKeyArray
            ) -> tuple[PyTree, Array, Output]:
                def loss_fn(model_arr: PyTree) -> tuple[Array, Output]:
                    model = eqx.combine(model_arr, model_static)
                    batch_casted = jax.tree.map(cast_compute_dtype, batch)
                    output_key, loss_key = jax.random.split(step_key)
                    output = get_output(model, batch_casted, state, output_key)
                    loss = compute_loss(model, batch_casted, output, state, loss_key)
                    return loss, output

                (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_arr)
                grads = jax.tree.map(lambda g: g.astype(grad_dtype) if eqx.is_inexact_array(g) else g, grads)
                return grads, loss, output

            # Get first leaf for batch size calculation later
            first_leaf = jax.tree.leaves(batches)[0]

            def accum_fn(
                carry: tuple[PyTree, Array, Output, PRNGKeyArray],
                batch: Batch,
            ) -> tuple[tuple[PyTree, Array, Output, PRNGKeyArray], None]:
                acc_grads, acc_loss, _, key = carry
                key, step_key = jax.random.split(key)
                grads, loss, output = compute_grads_and_loss(model_arr, batch, step_key)
                acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
                acc_loss = acc_loss + loss
                # Keep only the last output in carry; return None to avoid storing all outputs
                return (acc_grads, acc_loss, output, key), None

            # Run first micro-batch to get initial output structure
            first_batch = jax.tree.map(lambda x: x[0], batches)
            key, first_key = jax.random.split(key)
            first_grads, first_loss, init_output = compute_grads_and_loss(model_arr, first_batch, first_key)

            # Accumulate remaining gradients over micro-batches using scan.
            # Only the last output is kept in the carry; we return None to avoid storing all outputs.
            remaining_batches = jax.tree.map(lambda x: x[1:], batches)
            (acc_grads, acc_loss, output, key), _ = jax.lax.scan(
                accum_fn,
                (first_grads, first_loss, init_output, key),
                remaining_batches,
            )

            # Average gradients and loss over micro-batches.
            acc_grads = jax.tree.map(lambda g: g / num_accum_steps, acc_grads)
            acc_loss = acc_loss / num_accum_steps

            # Clip gradients if configured
            grad_norm = cast(Array, optax.global_norm(acc_grads))
            if max_grad_norm is not None:
                clip_fn = optax.clip_by_global_norm(max_grad_norm)
                clip_state = clip_fn.init(acc_grads)
                acc_grads, _ = clip_fn.update(acc_grads, clip_state)

            # Apply optimizer updates
            updates, new_opt_state = optimizer.update(acc_grads, opt_state, model_arr)
            updates = jax.tree.map(lambda u: u.astype(param_dtype) if eqx.is_inexact_array(u) else u, updates)
            new_model_arr = eqx.apply_updates(model_arr, updates)

            # Calculate batch size and update state
            batch_size = first_leaf.shape[1]
            total_samples = batch_size * num_accum_steps
            new_state = state.replace(
                num_steps=state.num_steps + 1,
                num_samples=state.num_samples + total_samples,
            )
            # Ensure state arrays have replicated sharding for multi-GPU
            mesh = jax.sharding.get_abstract_mesh()
            if mesh is not None and mesh.shape_tuple:
                replicated = NamedSharding(mesh, PartitionSpec())
                # Apply sharding constraint to underlying state arrays
                new_int32_arr = jax.lax.with_sharding_constraint(new_state._int32_arr, replicated)
                new_float32_arr = jax.lax.with_sharding_constraint(new_state._float32_arr, replicated)
                new_state = State(_int32_arr=new_int32_arr, _float32_arr=new_float32_arr)

            # Extract batch for metrics computation
            batch = jax.tree.map(lambda x: x[-1], batches)
            batch = jax.tree.map(lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, batch)
            model = eqx.combine(new_model_arr, model_static)

            # Compute metrics
            metrics = compute_metrics(model, batch, output, new_state, heavy, key)
            metrics["loss"] = Scalar(acc_loss)
            metrics["grad_norm"] = Scalar(grad_norm)

            return new_model_arr, new_opt_state, new_state, metrics

        # Use static_argnames for `heavy` so both versions are compiled and cached
        return jit(static_argnames=["heavy"])(train_step)

    def train_loop(
        self,
        models: list[PyTree],
        opt_states: list[optax.OptState],
        state: State,
        optimizers: Sequence[Optimizer],
        ds: Iterator[Batch],
        key: PRNGKeyArray,
    ) -> State:
        if len(models) != 1 or len(optimizers) != 1 or len(opt_states) != 1:
            raise ValueError(
                "Vanilla training expects a single model, optimizer and optimizer state. "
                f"Found {len(models)} models, {len(optimizers)} optimizers and {len(opt_states)} optimizer states."
            )

        filter_spec = self.get_model_filter_spec(models[0])
        model_arr, model_static = eqx.partition(models[0], filter_spec)
        optimizer = optimizers[0]
        opt_state = opt_states[0]

        # Handle arrays with unspecified sharding from multi-GPU JIT
        # by extracting data from addressable shards and replicating across mesh
        mesh = jax.sharding.get_mesh()
        if mesh is not None and mesh.devices.size > 1:
            replicated_sharding = NamedSharding(mesh, PartitionSpec())
            unspecified_sharding_fn = functools.partial(fix_unspecified_sharding, sharding=replicated_sharding)

            # Make sure that all arrays have specified sharding.
            model_arr = jax.tree.map(unspecified_sharding_fn, model_arr)
            opt_state = jax.tree.map(unspecified_sharding_fn, opt_state)
            state = jax.tree.map(unspecified_sharding_fn, state)

        # Create JIT-compiled train step (heavy is a static arg, so both versions get cached)
        train_step_fn = self.create_train_step_fn(model_static, optimizer)

        def save_checkpoint() -> None:
            model = eqx.combine(model_arr, model_static)
            self.save_checkpoint(
                models=[model],
                optimizers=optimizers,
                opt_states=[opt_state],
                state=state,
            )

        self.add_signal_handler(save_checkpoint, signal.SIGUSR1, signal.SIGTERM)

        while True:
            is_done = self.is_training_over(state)

            with ContextTimer() as timer:
                self.on_step_start()

                # Get stacked batches for gradient accumulation
                if isinstance(ds, InMemoryBatchIterator):
                    batches_stacked = ds.get_stacked_batches(self.config.gradient_accumulation_steps)
                else:
                    batches = tuple(itertools.islice(ds, self.config.gradient_accumulation_steps))
                    batches_stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *batches)

                # Ensure batches have consistent sharding to prevent recompilation
                if mesh is not None and mesh.devices.size > 1:
                    batches_stacked = jax.tree.map(unspecified_sharding_fn, batches_stacked)

                step_key, key = jax.random.split(key)

                # Get log mode (heavy flag) and update state. Always do heavy
                # logging on the last step.
                state, heavy_arr = self.get_log_mode(state)
                heavy = heavy_arr.item() or is_done

                # Execute training step
                model_arr, opt_state, state, metrics = train_step_fn(
                    model_arr,
                    opt_state,
                    batches_stacked,
                    state,
                    step_key,
                    heavy,
                )

                self.log_step(FrozenDict(metrics), state, heavy)
                self.on_step_end()

            state = state.replace(elapsed_time_s=state.elapsed_time_s + timer.elapsed_time)

            if state.num_steps <= 3:
                logger.log(LOG_PING, "Step %d took %.2f second", state.num_steps, timer.elapsed_time)

            if self.should_checkpoint(state):
                save_checkpoint()

            if is_done:
                break

        save_checkpoint()
        return state

    def run(self) -> None:
        self.run_training()

    def run_training(self) -> None:
        """Runs the training loop."""
        training_error: BaseException | None = None

        with self:
            key = self.prng_key()

            self.set_loggers()

            if is_master():
                Thread(target=self.log_state, daemon=True).start()

            mesh = self.get_mesh()
            jax.set_mesh(mesh)

            data_sharding = self.get_data_sharding(mesh)
            model_sharding = self.get_model_sharding(mesh)

            model_key, key = jax.random.split(key)
            init_params = InitParams(key=model_key)
            models, optimizers, opt_states, state = self.load_initial_state(
                init_params,
                load_optimizer=True,
                model_sharding=model_sharding,
            )

            # Log the model and optimizer sizes.
            m_params = get_pytree_param_count(models)
            o_params = get_pytree_param_count(opt_states)
            logger.log(LOG_STATUS, "Model: %s params, Optimizer: %s params", f"{m_params:,}", f"{o_params:,}")

            self.on_training_start()

            try:
                # Choose data loading strategy based on config
                data_dtype = self.config.precision.data_jax_dtype
                if self.config.load_in_memory:
                    ds: Iterator[Batch] = self.get_in_memory_iterator(data_sharding, key, data_dtype)
                else:
                    ds = self.get_streaming_iterator(data_sharding, data_dtype)

                state = self.train_loop(
                    models=models,
                    opt_states=opt_states,
                    state=state,
                    optimizers=optimizers,
                    ds=ds,
                    key=key,
                )

                if is_master():
                    num_steps, num_samples = int(state.num_steps), int(state.num_samples)
                    show_info(f"Finished training after {num_steps} steps, {num_samples} samples", important=True)

            except BaseException as e:
                training_error = e

            finally:
                self.on_training_end()

        # If an error occurred during training, print the traceback and force
        # exit. This is necessary because JAX/NCCL can leave background threads
        # in a bad state after an error, which prevents Python from exiting
        # cleanly.
        if training_error is not None:
            traceback.print_exception(type(training_error), training_error, training_error.__traceback__)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
