"""Trains a speech generation model on the LJSpeech dataset."""

from dataclasses import dataclass
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray, PyTree

import xax


class Batch(TypedDict):
    audio: Array
    text: Array


@dataclass
class Config(xax.SupervisedConfig):
    learning_rate: float = xax.field(1e-3, help="The learning rate")
    hidden_dim: int = xax.field(512, help="Hidden layer dimension")
    num_layers: int = xax.field(4, help="Number of layers")
    fast_context_length: int = xax.field(10, help="Context length of the fast transformer")
    slow_context_length: int = xax.field(512, help="Context length of the slow transformer")


class SpeechModel(eqx.Module):
    slow_transformer: xax.TransformerStack
    fast_transformer: xax.TransformerStack

    def __init__(self, config: Config, key: PRNGKeyArray) -> None:
        # Creates the slow transformer, which processes at 5 tokens / second.
        slow_key, key = jax.random.split(key)
        self.slow_transformer = xax.TransformerStack(
            embed_dim=config.hidden_dim,
            num_heads=config.hidden_dim // 64,
            ff_dim=config.hidden_dim * 4,
            num_layers=config.num_layers,
            causal=True,
            cross_attention=False,
            context_length=config.slow_context_length,
            use_rotary_embeddings=True,
            key=slow_key,
        )

        # Creates the fast transformer, which processes at 50 tokens / second.
        fast_key, key = jax.random.split(key)
        self.fast_transformer = xax.TransformerStack(
            embed_dim=config.hidden_dim,
            num_heads=config.hidden_dim // 64,
            ff_dim=config.hidden_dim * 4,
            num_layers=config.num_layers,
            causal=True,
            cross_attention=False,
            context_length=config.fast_context_length,
            use_rotary_embeddings=True,
            key=fast_key,
        )


class SpeechGeneration(xax.SupervisedTask[Config]):
    def get_model(self, params: xax.InitParams) -> SpeechModel:
        return SpeechModel(self.config, key=params.key)

    def get_optimizer(self) -> xax.Optimizer:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=self.config.learning_rate,
            transition_steps=self.config.warmup_steps,
        )

        if self.config.max_steps is not None:
            cosine_schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=max(self.config.max_steps - self.config.warmup_steps, 1),
                alpha=0.1,  # Decay to 10% of initial learning rate
            )

            learning_rate_schedule = optax.join_schedules(
                schedules=[warmup_schedule, cosine_schedule],
                boundaries=[self.config.warmup_steps],
            )
        else:
            learning_rate_schedule = warmup_schedule

        opt = optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=1e-4,
            b1=0.9,
            b2=0.999,
        )

        return optax.MultiSteps(opt, every_k_schedule=8)

    def get_output(
        self,
        model: SpeechModel,
        batch: Batch,
        state: xax.State,
        key: PRNGKeyArray,
    ) -> Array:
        images_bhw = (batch["image"].astype(np.float32) / 255.0) - 0.5
        labels_b = batch["label"]

        def model_fn(x_bhw: Array, t_b: Array) -> Array:
            return xax.vmap(model)(x_bhw, t_b, labels_b)

        loss = self.diffusion.loss(key, model_fn, images_bhw)
        return loss.mean()

    def compute_loss(
        self,
        model: SpeechModel,
        batch: Batch,
        output: Array,
        state: xax.State,
        key: PRNGKeyArray,
    ) -> Array:
        """The output is already the loss."""
        return output

    def compute_metrics(
        self,
        model: PyTree,
        batch: Batch,
        output: Array,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> dict[str, xax.Metric]:
        """Generate and log sample images."""
        metrics: dict[str, xax.Metric] = {}
        if not heavy:
            return metrics

        max_images = 9
        num_sample = self.config.sampling_timesteps or 50

        images = (batch["image"].astype(np.float32) / 255.0) - 0.5
        class_id = batch["label"]
        metrics["real"] = xax.LabeledImages(
            images,
            class_id,
            max_images=max_images,
            target_resolution=(64, 64),
        )

        def vanilla_func(x_bt: Array, t_b: Array) -> Array:
            return jax.vmap(model)(x_bt, t_b, class_id)

        # Generate samples.
        gen = self.diffusion.sample(
            key,
            vanilla_func,
            shape=(len(class_id), 32, 32),
            sampling_timesteps=num_sample,
        )

        # Get the final denoised samples (first element is the final sample).
        generated_images = gen[0]
        metrics["generated"] = xax.LabeledImages(
            generated_images,
            class_id,
            max_images=max_images,
            target_resolution=(64, 64),
        )

        # Log single image sequence.
        indices = jnp.linspace(0, gen.shape[0] - 1, max_images).astype(jnp.int32).clip(0, gen.shape[0] - 1)
        one_gen = gen[indices, 0]
        metrics["generated_single"] = xax.LabeledImages(
            one_gen,
            indices,
            max_images=max_images,
            target_resolution=(64, 64),
        )

        return metrics

    def get_dataset(self) -> Dataset:
        ds = load_dataset("keithito/lj_speech")
        breakpoint()
        ds.set_format(type="numpy", columns=["audio", "label"])
        return ds


if __name__ == "__main__":
    # python -m examples.mnist_diffusion
    SpeechGeneration.launch(
        Config(
            batch_size=32,
            log_heavy_every_n_seconds=60 * 5,
            max_grad_norm=1.0,
        ),
    )
