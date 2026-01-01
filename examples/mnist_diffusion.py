#!/usr/bin/env -S uv run --no-project --script
"""Trains a Gaussian diffusion model to conditionally generate MNIST digits.

Run this example with `python -m examples.mnist_diffusion`.
"""

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
    image: Array
    label: Array


@dataclass
class Config(xax.SupervisedConfig):
    learning_rate: float = xax.field(1e-3, help="The learning rate")
    hidden_dim: int = xax.field(128, help="Hidden layer dimension")
    dim_scales: list[int] = xax.field([1, 2, 4, 8], help="List of dimension scales for UNet")
    num_timesteps: int = xax.field(500, help="Number of diffusion timesteps")
    pred_mode: str = xax.field("pred_v", help="Prediction mode: pred_x_0, pred_eps, or pred_v")
    beta_schedule: str = xax.field("linear", help="Beta schedule type")
    warmup_steps: int = xax.field(100, help="Number of warmup steps")
    sampling_timesteps: int | None = xax.field(50, help="Number of timesteps for sampling")


class UNet(eqx.Module):
    class_embs: eqx.nn.Embedding
    fourier_emb: xax.FourierEmbeddings
    time_proj_1: eqx.nn.Linear
    time_proj_2: eqx.nn.Linear
    unet: xax.UNet

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        dim_scales: list[int],
        num_classes: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        # Class embeddings.
        emb_key, key = jax.random.split(key)
        self.class_embs = eqx.nn.Embedding(
            num_embeddings=num_classes,
            embedding_size=embed_dim,
            key=emb_key,
        )

        # Time embeddings.
        key1, key2, key = jax.random.split(key, 3)
        self.fourier_emb = xax.FourierEmbeddings(embed_dim)
        self.time_proj_1 = eqx.nn.Linear(embed_dim, embed_dim, key=key1)
        self.time_proj_2 = eqx.nn.Linear(embed_dim, embed_dim, key=key2)

        self.unet = xax.UNet(
            in_dim=in_dim,
            embed_dim=embed_dim,
            dim_scales=dim_scales,
            input_embedding_dim=embed_dim,
            key=key,
        )

    def __call__(self, x_hw: Array, t: Array, c: Array) -> Array:
        dtype = self.class_embs.weight.dtype

        # Embed class.
        c_n = self.class_embs(c)

        # Embed time.
        t_n = self.fourier_emb(t).astype(dtype)
        t_n = self.time_proj_1(t_n)
        t_n = xax.get_activation("silu")(t_n)
        t_n = self.time_proj_2(t_n)

        # Combine embeddings.
        e_n = c_n + t_n

        o_hw = self.unet(x_hw[None].astype(dtype), e_n)[0]

        return o_hw


class MnistDiffusion(xax.SupervisedTask[Config]):
    diffusion: xax.GaussianDiffusion

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.diffusion = xax.GaussianDiffusion(
            beta_schedule=xax.cast_beta_schedule(config.beta_schedule),
            num_beta_steps=config.num_timesteps,
            pred_mode=xax.cast_diffusion_pred_mode(config.pred_mode),
        )

    def get_model(self, params: xax.InitParams) -> UNet:
        return UNet(
            in_dim=1,
            embed_dim=self.config.hidden_dim,
            dim_scales=self.config.dim_scales,
            num_classes=10,
            key=params.key,
        )

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

        return optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=1e-4,
            b1=0.9,
            b2=0.999,
        )

    def get_output(self, model: UNet, batch: Batch, state: xax.State | None, key: PRNGKeyArray) -> Array:
        """Computes the diffusion loss.

        The model is called with (x_t, t, class_label) internally, but we need
        to wrap it to match the diffusion API which expects (x_t, t) -> output.
        """
        images_bhw = (batch["image"].astype(np.float32) / 255.0) - 0.5
        labels_b = batch["label"]

        def model_fn(x_bhw: Array, t_b: Array) -> Array:
            return xax.vmap(model)(x_bhw, t_b, labels_b)

        loss = self.diffusion.loss(key, model_fn, images_bhw)
        return loss.mean()

    def compute_loss(
        self, model: UNet, batch: Batch, output: Array, state: xax.State | None, key: PRNGKeyArray
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
        ds = load_dataset("ylecun/mnist", split="train")
        ds.set_format(type="numpy", columns=["image", "label"])
        return ds


if __name__ == "__main__":
    MnistDiffusion.launch(
        Config(
            batch_size=256,
            log_heavy_every_n_seconds=60 * 5,
            max_grad_norm=1.0,
            # MNIST dataset is very small and this greatly improves throughput.
            load_in_memory=True,
            gradient_accumulation_steps=64,
        ),
    )
