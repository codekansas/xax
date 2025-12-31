"""Defines a mixin for running the supervised training loop."""

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
from xax.task.mixins.data_loader import InMemoryBatchIterator
from xax.task.mixins.train import InitParams, Optimizer, TrainConfig, TrainMixin
from xax.utils.experiments import ContextTimer
from xax.utils.jax import jit as xax_jit
from xax.utils.logging import LOG_PING
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
        # Cast batch from data_dtype to compute_dtype for forward pass
        batch = jax.tree.map(self.cast_compute_dtype, batch)
        output = self.get_output(model, batch, state, output_key)
        loss = self.compute_loss(model, batch, output, state, loss_key)
        return loss, (loss, output)

    @xax_jit(
        static_argnames=["self", "model_static"],
        donate_argnames=["batch", "key"],
        jit_level=3,
    )
    def compute_gradients(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        batch: Batch,
        state: State,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, Array, Output]:
        """Compute gradients for a single batch.

        This function computes gradients without applying them, enabling
        gradient accumulation across multiple batches before syncing.

        Args:
            model_arr: The model array parameters.
            model_static: The model static parameters.
            batch: The current minibatch of samples.
            state: The current training state.
            key: The current PRNG key.

        Returns:
            A tuple of (gradients, loss, output).
        """
        grad_fn = jax.grad(self.get_output_and_loss, argnums=0, has_aux=True)
        grad_fn = xax_jit(static_argnums=[1], jit_level=3)(grad_fn)
        grads, aux = grad_fn(model_arr, model_static, batch, state, key)
        loss, output = cast(tuple[Array, Output], aux)

        # Cast gradients to grad_dtype for accumulation
        grad_dtype = self.config.precision.grad_jax_dtype
        grads = jax.tree.map(lambda g: g.astype(grad_dtype) if eqx.is_inexact_array(g) else g, grads)

        return grads, loss, output

    @xax_jit(
        static_argnames=["self", "optimizer"],
        donate_argnames=["model_arr", "opt_state", "grads"],
        jit_level=3,
    )
    def apply_gradients(
        self,
        model_arr: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        grads: PyTree,
        loss: Array,
    ) -> tuple[PyTree, optax.OptState, Array, Array]:
        """Apply gradients to update the model.

        This function handles gradient clipping and optimizer updates.
        Gradient synchronization across devices should be done before calling this.

        Args:
            model_arr: The model array parameters.
            optimizer: The optimizer.
            opt_state: The optimizer state.
            grads: The gradients (already synchronized if multi-device).
            loss: The loss (already synchronized if multi-device).

        Returns:
            A tuple of (updated_model_arr, updated_opt_state, loss, grad_norm).
        """
        grad_norm = optax.global_norm(grads)
        if self.config.max_grad_norm is not None:
            clip_fn = optax.clip_by_global_norm(self.config.max_grad_norm)
            clip_state = clip_fn.init(grads)  # Fine since this function is init_empty_state
            grads, _ = clip_fn.update(grads, clip_state)

        updates, opt_state = optimizer.update(grads, opt_state, model_arr)

        # Cast updates to param_dtype before applying (needed when grad_dtype != param_dtype)
        param_dtype = self.config.precision.param_jax_dtype
        updates = jax.tree.map(lambda u: u.astype(param_dtype) if eqx.is_inexact_array(u) else u, updates)

        model_arr = eqx.apply_updates(model_arr, updates)

        return model_arr, opt_state, loss, grad_norm  # type: ignore[return-value]

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

    def train_step_pmap(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: Optimizer,
        opt_state: optax.OptState,
        batches_pmap: Batch,
        state: State,
        key: PRNGKeyArray,
        num_devices: int,
    ) -> tuple[PyTree, optax.OptState, Array, Array, Output]:
        """Run a single pmap training step.

        This is the inner training step that runs on replicated model/opt_state.
        Model and opt_state should already be in replicated form (num_devices, ...).
        Returns replicated outputs.

        Args:
            model_arr: Replicated model array (num_devices, ...).
            model_static: The model static parameters.
            optimizer: The optimizer.
            opt_state: Replicated optimizer state (num_devices, ...).
            batches_pmap: Batches reshaped for pmap (num_devices, num_accum, local_batch, ...).
            state: The training state.
            key: The PRNG key.
            num_devices: Number of devices.

        Returns:
            Replicated (model_arr, opt_state, loss, grad_norm, output).
        """
        num_accum_steps = self.config.gradient_accumulation_steps
        grad_dtype = self.config.precision.grad_jax_dtype
        param_dtype = self.config.precision.param_jax_dtype
        max_grad_norm = self.config.max_grad_norm

        def compute_grads_and_loss(model_arr: PyTree, batch: Batch, key: PRNGKeyArray) -> tuple[PyTree, Array, Output]:  # noqa: ANN001
            """Compute gradients and loss for a local batch shard."""

            def loss_fn(model_arr: PyTree) -> tuple[Array, Output]:  # noqa: ANN001
                model = eqx.combine(model_arr, model_static)
                batch_casted = jax.tree.map(self.cast_compute_dtype, batch)
                output_key, loss_key = jax.random.split(key)
                output = self.get_output(model, batch_casted, state, output_key)
                loss = self.compute_loss(model, batch_casted, output, state, loss_key)
                return loss, output

            (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_arr)
            grads = jax.tree.map(lambda g: g.astype(grad_dtype) if eqx.is_inexact_array(g) else g, grads)
            return grads, loss, output

        def pmap_train_step(  # noqa: ANN001, ANN202
            model_arr: PyTree,
            opt_state: optax.OptState,
            batches: Batch,
            keys: PRNGKeyArray,
        ) -> tuple[PyTree, optax.OptState, Array, Array, Output]:
            """Run training step on each device shard, aggregate gradients."""

            def accumulate_and_update(  # noqa: ANN001, ANN202
                model_arr: PyTree, opt_state: optax.OptState, batches: Batch, key: PRNGKeyArray
            ) -> tuple[PyTree, optax.OptState, Array, Array, Output]:
                init_grads = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=grad_dtype), model_arr)
                init_loss = jnp.array(0.0)

                def accum_fn(
                    carry: tuple[PyTree, Array],
                    batch_and_key: tuple[Batch, PRNGKeyArray],
                ) -> tuple[tuple[PyTree, Array], Output]:
                    acc_grads, acc_loss = carry
                    batch, key = batch_and_key
                    grads, loss, output = compute_grads_and_loss(model_arr, batch, key)
                    acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
                    acc_loss = acc_loss + loss
                    return (acc_grads, acc_loss), output

                accum_keys = jax.random.split(key, num_accum_steps)
                (acc_grads, acc_loss), outputs = jax.lax.scan(accum_fn, (init_grads, init_loss), (batches, accum_keys))

                if num_accum_steps > 1:
                    acc_grads = jax.tree.map(lambda g: g / num_accum_steps, acc_grads)
                    acc_loss = acc_loss / num_accum_steps

                acc_grads = jax.tree.map(lambda g: jax.lax.pmean(g, axis_name="batch"), acc_grads)
                acc_loss = jax.lax.pmean(acc_loss, axis_name="batch")

                grad_norm = optax.global_norm(acc_grads)
                if max_grad_norm is not None:
                    clip_fn = optax.clip_by_global_norm(max_grad_norm)
                    clip_state = clip_fn.init(acc_grads)
                    acc_grads, _ = clip_fn.update(acc_grads, clip_state)

                updates, opt_state = optimizer.update(acc_grads, opt_state, model_arr)
                updates = jax.tree.map(lambda u: u.astype(param_dtype) if eqx.is_inexact_array(u) else u, updates)
                model_arr = eqx.apply_updates(model_arr, updates)

                output = jax.tree.map(lambda x: x[-1], outputs)
                return model_arr, opt_state, acc_loss, grad_norm, output  # type: ignore[return-value]

            return accumulate_and_update(model_arr, opt_state, batches, keys)

        pmap_fn = jax.pmap(pmap_train_step, axis_name="batch")
        keys = jax.random.split(key, num_devices)

        return pmap_fn(model_arr, opt_state, batches_pmap, keys)

    def reshape_batch_for_pmap(self, batches_stacked: Batch, num_devices: int) -> Batch:
        """Reshape stacked batches for pmap consumption.

        Args:
            batches_stacked: Batches with shape (num_accum, batch_size, ...).
            num_devices: Number of devices.

        Returns:
            Batches reshaped to (num_devices, num_accum, local_batch_size, ...).
        """

        def reshape_for_pmap(x: Array) -> Array:  # noqa: ANN001
            shape = x.shape
            num_accum = shape[0]
            batch_size = shape[1]
            local_batch_size = batch_size // num_devices
            rest = shape[2:]
            reshaped = x.reshape(num_accum, num_devices, local_batch_size, *rest)
            return jnp.transpose(reshaped, (1, 0, 2, *range(3, len(reshaped.shape))))

        return jax.tree.map(reshape_for_pmap, batches_stacked)

    def replicate_for_pmap(self, pytree: PyTree, num_devices: int) -> PyTree:
        """Replicate a pytree across devices for pmap."""
        return jax.tree.map(lambda x: jnp.stack([x] * num_devices), pytree)

    def unreplicate_from_pmap(self, pytree: PyTree) -> PyTree:
        """Extract device 0's copy from a replicated pytree."""
        return jax.tree.map(lambda x: x[0], pytree)

    def train_loop(
        self,
        models: list[PyTree],
        opt_states: list[optax.OptState],
        state: State,
        optimizers: Sequence[Optimizer],
        ds: Iterator[Batch],
        key: PRNGKeyArray,
        num_devices: int = 1,
    ) -> State:
        if len(models) != 1 or len(optimizers) != 1 or len(opt_states) != 1:
            raise ValueError(
                "Vanilla training expects a single model, optimizer and optimizer state. "
                f"Found {len(models)} models, {len(optimizers)} optimizers and {len(opt_states)} optimizer states."
            )

        model_arr, model_static = eqx.partition(models[0], self.model_partition_fn)
        optimizer = optimizers[0]
        opt_state = opt_states[0]

        # Replicate model and optimizer state once for pmap.
        # For single device, this creates shape (1, ...) which pmap handles correctly.
        model_arr_rep = self.replicate_for_pmap(model_arr, num_devices)
        opt_state_rep = self.replicate_for_pmap(opt_state, num_devices)

        def save_checkpoint() -> None:
            model_arr_local = self.unreplicate_from_pmap(model_arr_rep)
            opt_state_local = self.unreplicate_from_pmap(opt_state_rep)
            model = eqx.combine(model_arr_local, model_static)
            self.save_checkpoint(
                models=[model],
                optimizers=optimizers,
                opt_states=[opt_state_local],
                state=state,
            )

        self.add_signal_handler(save_checkpoint, signal.SIGUSR1, signal.SIGTERM)

        while not self.is_training_over(state):
            with ContextTimer() as timer:
                self.on_step_start()

                # Get stacked batches
                if isinstance(ds, InMemoryBatchIterator):
                    batches_stacked = ds.get_stacked_batches(self.config.gradient_accumulation_steps)
                else:
                    batches = tuple(itertools.islice(ds, self.config.gradient_accumulation_steps))
                    batches_stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *batches)

                state, heavy_arr = self.get_log_mode(state)
                heavy = heavy_arr.item()
                step_key, key = jax.random.split(key)

                # Reshape batch for pmap: (num_accum, batch, ...) -> (num_devices, num_accum, local_batch, ...)
                batches_pmap = self.reshape_batch_for_pmap(batches_stacked, num_devices)

                # Run pmap training step
                model_arr_rep, opt_state_rep, loss_rep, grad_norm_rep, output_rep = self.train_step_pmap(
                    model_arr=model_arr_rep,
                    model_static=model_static,
                    optimizer=optimizer,
                    opt_state=opt_state_rep,
                    batches_pmap=batches_pmap,
                    state=state,
                    key=step_key,
                    num_devices=num_devices,
                )

                # Extract scalars for logging (cheap - just index into replicated array)
                loss = loss_rep[0]
                grad_norm = grad_norm_rep[0]

                # Compute metrics - only extract full model/output for heavy logging
                batch_size = jax.tree.leaves(batches_stacked)[0].shape[1]
                if heavy:
                    model_arr_local = self.unreplicate_from_pmap(model_arr_rep)
                    output = self.unreplicate_from_pmap(output_rep)
                    batch = jax.tree.map(lambda x: x[0][-1], batches_pmap)  # device 0, last accum step
                    batch = jax.tree.map(
                        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, batch
                    )
                    model = eqx.combine(model_arr_local, model_static)
                    metrics = self.compute_metrics(model, batch, output, state, heavy, step_key)
                else:
                    metrics = {}

                # Update state
                total_samples = batch_size * self.config.gradient_accumulation_steps
                state = state.replace(
                    num_steps=state.num_steps + 1,
                    num_samples=state.num_samples + total_samples,
                )

                metrics["loss"] = Scalar(loss)
                metrics["grad_norm"] = Scalar(grad_norm)

                self.log_step(FrozenDict(metrics), state, heavy)
                self.on_step_end()

            state = state.replace(elapsed_time_s=state.elapsed_time_s + timer.elapsed_time)

            if state.num_steps <= 3:
                logger.log(LOG_PING, "Step %d took %.2f second", state.num_steps, timer.elapsed_time)

            if self.should_checkpoint(state):
                save_checkpoint()

        save_checkpoint()
        return state

    def run(self) -> None:
        self.run_training()

    def run_training(self) -> None:
        """Runs the training loop.

        Uses pmap for data-parallel training across multiple devices.
        Data is loaded without device sharding, then split by pmap.
        """
        training_error: BaseException | None = None

        with self:
            key = self.prng_key()

            self.set_loggers()

            if is_master():
                Thread(target=self.log_state, daemon=True).start()

            # Get number of devices for data parallelism
            num_devices = jax.local_device_count()

            # For pmap, we don't use sharding - data is loaded to host then split by pmap
            # Create a simple sharding for data loading (no device sharding)
            mesh = self.get_mesh()
            # Use replicated sharding for data (pmap will split it)
            data_sharding = NamedSharding(mesh, P())

            model_key, key = jax.random.split(key)
            init_params = InitParams(key=model_key)
            models, optimizers, opt_states, state = self.load_initial_state(
                init_params,
                load_optimizer=True,
                model_sharding=None,  # Don't shard model, pmap handles replication
            )

            logger.info("Model size: %s", f"{get_pytree_param_count(models):,}")
            logger.info("Optimizer size: %s", f"{get_pytree_param_count(opt_states):,}")

            self.on_training_start()

            try:
                # Choose data loading strategy based on config
                # For pmap, we load data without device sharding
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
                    num_devices=num_devices,
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
