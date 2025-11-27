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
from PIL.Image import Image as PILImage

import xax


class Batch(TypedDict):
    image: Array
    label: Array


@dataclass
class Config(xax.SupervisedConfig):
    batch_size: int = xax.field(128, help="The size of a minibatch")
    learning_rate: float = xax.field(1e-4, help="The learning rate")
    hidden_dim: int = xax.field(256, help="Hidden layer dimension")
    num_timesteps: int = xax.field(1000, help="Number of diffusion timesteps")
    pred_mode: str = xax.field("pred_x_0", help="Prediction mode: pred_x_0, pred_eps, or pred_v")
    beta_schedule: str = xax.field("jsd", help="Beta schedule type")
    warmup_steps: int = xax.field(250, help="Number of warmup steps")
    sampling_timesteps: int = xax.field(100, help="Number of timesteps for sampling")


class ResBlock(eqx.Module):
    """Residual block with time and class conditioning."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    time_proj: eqx.nn.Linear
    class_proj: eqx.nn.Linear | None
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(
        self,
        channels: int,
        time_dim: int,
        class_dim: int | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2, key_time, key_class = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=key2)
        self.time_proj = eqx.nn.Linear(time_dim, channels, key=key_time)
        self.class_proj = eqx.nn.Linear(class_dim, channels, key=key_class) if class_dim is not None else None
        self.norm1 = eqx.nn.GroupNorm(min(8, channels), channels)
        self.norm2 = eqx.nn.GroupNorm(min(8, channels), channels)

    def __call__(self, x: Array, t_emb: Array, class_emb: Array | None = None) -> Array:
        """Forward pass of residual block.

        Args:
            x: Input features, shape (C, H, W)
            t_emb: Time embedding, shape (time_dim,)
            class_emb: Class embedding, shape (class_dim,) or None

        Returns:
            Output features, shape (out_channels, H, W)
        """
        h = self.norm1(x)
        h = jax.nn.swish(h)
        h = self.conv1(h)

        # Add time conditioning
        t_proj = self.time_proj(t_emb)  # (out_channels,)
        h = h + t_proj[:, None, None]  # Broadcast to spatial dimensions

        # Add class conditioning if provided
        if class_emb is not None and self.class_proj is not None:
            c_proj = self.class_proj(class_emb)  # (out_channels,)
            h = h + c_proj[:, None, None]  # Broadcast to spatial dimensions

        h = self.norm2(h)
        h = jax.nn.swish(h)
        h = self.conv2(h)

        return h + x


class DiffusionModel(eqx.Module):
    """A UNet-style diffusion model with class conditioning."""

    time_embed: xax.FourierEmbeddings
    class_embed: eqx.nn.Embedding
    input_conv: eqx.nn.Conv2d
    down_blocks: tuple[tuple[ResBlock, ...], ...]
    down_samples: tuple[eqx.nn.Conv2d, ...]
    bottleneck: ResBlock
    up_samples: tuple[eqx.nn.ConvTranspose2d, ...]
    up_blocks: tuple[tuple[ResBlock, ...], ...]
    output_norm: eqx.nn.GroupNorm
    output_conv: eqx.nn.Conv2d
    hidden_dim: int
    num_classes: int

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_timesteps: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Time embedding using Fourier features
        time_dim = hidden_dim
        self.time_embed = xax.FourierEmbeddings(dim=time_dim, max_period=num_timesteps)

        # Class embedding
        class_dim = hidden_dim
        embed_key, input_key = jax.random.split(key)
        self.class_embed = eqx.nn.Embedding(num_classes, class_dim, key=embed_key)

        # Input convolution: 1 channel (grayscale) -> hidden_dim
        self.input_conv = eqx.nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, key=input_key)

        # UNet architecture: 3 down/up levels for 28x28 images
        # Level 0: 28x28, Level 1: 14x14, Level 2: 7x7
        channels = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        num_levels = len(channels)

        # Downsampling path
        down_blocks_list: list[tuple[ResBlock, ...]] = []
        down_samples_list: list[eqx.nn.Conv2d] = []

        for i in range(num_levels):
            level_blocks: list[ResBlock] = []
            num_blocks = 2 if i == 0 else 1  # More blocks at first level
            for _ in range(num_blocks):
                block_key, key = jax.random.split(key)
                level_blocks.append(ResBlock(channels[i], time_dim, class_dim, key=block_key))
            down_blocks_list.append(tuple(level_blocks))

            # Downsample (except for last level)
            if i < num_levels - 1:
                downsample_key, key = jax.random.split(key)
                downsample = eqx.nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=downsample_key,
                )
                down_samples_list.append(downsample)

        self.down_blocks = tuple(down_blocks_list)
        self.down_samples = tuple(down_samples_list)

        # Bottleneck
        bottleneck_key, key = jax.random.split(key)
        self.bottleneck = ResBlock(channels[-1], time_dim, class_dim, key=bottleneck_key)

        # Upsampling path
        up_blocks_list: list[tuple[ResBlock, ...]] = []
        up_samples_list: list[eqx.nn.ConvTranspose2d] = []

        for i in range(num_levels - 1, -1, -1):
            # Upsample (except for first level)
            if i < num_levels - 1:
                upsample_key, key = jax.random.split(key)
                upsample = eqx.nn.ConvTranspose2d(
                    channels[i + 1],
                    channels[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    key=upsample_key,
                )
                up_samples_list.append(upsample)

            up_level_blocks: list[ResBlock] = []
            num_blocks = 2 if i == 0 else 1
            for _ in range(num_blocks):
                block_key, key = jax.random.split(key)
                up_level_blocks.append(ResBlock(channels[i], time_dim, class_dim, key=block_key))
            up_blocks_list.append(tuple(up_level_blocks))

        self.up_blocks = tuple(up_blocks_list)
        self.up_samples = tuple(up_samples_list)

        # Output layers
        self.output_norm = eqx.nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.output_conv = eqx.nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, key=key)

    def __call__(self, x_hw: Array, t: Array, class_label: Array | None = None) -> Array:
        """Forward pass of the diffusion model.

        Args:
            x_hw: Noisy image at timestep t, shape (H, W) for MNIST (28, 28)
            t: Timestep, shape ()
            class_label: Class label for conditioning, shape (). If None, uses zeros.

        Returns:
            Predicted target, shape (H, W) same as input
        """
        dtype = self.input_conv.weight.dtype
        x_1hw = x_hw[None].astype(dtype)

        # Time embedding
        t_emb = self.time_embed(t[None])[0].astype(dtype)

        # Class embedding
        if class_label is None:
            class_label = jnp.zeros((), dtype=jnp.int32)
        class_emb = self.class_embed(class_label)  # Shape: (class_dim,)

        # Input projection
        x = self.input_conv(x_1hw)  # (H, W, hidden_dim)

        # Downsampling path with skip connections
        skip_connections: list[Array] = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                x = block(x, t_emb, class_emb)
            skip_connections.append(x)
            if i < len(self.down_samples):
                x = self.down_samples[i](x)

        # Bottleneck
        x = self.bottleneck(x, t_emb, class_emb)

        # Upsampling path with skip connections
        for i, blocks in enumerate(self.up_blocks):
            if i > 0:
                x = self.up_samples[i - 1](x)
                skip = skip_connections[-(i + 1)]
                x = x + skip
            for block in blocks:
                x = block(x, t_emb, class_emb)

        # Output projection
        x = self.output_norm(x)
        x = jax.nn.swish(x)
        x = self.output_conv(x)  # (1, H, W)

        # Remove channel dimension: (1, H, W) -> (H, W)
        return x[0].astype(x_hw.dtype)


class MnistDiffusion(xax.SupervisedTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.diffusion = xax.GaussianDiffusion(
            beta_schedule=xax.cast_beta_schedule(config.beta_schedule),
            num_beta_steps=config.num_timesteps,
            pred_mode=xax.cast_diffusion_pred_mode(config.pred_mode),
        )
        self.diffusion = self.diffusion.reset_parameters()

    def get_model(self, params: xax.InitParams) -> DiffusionModel:
        return DiffusionModel(
            hidden_dim=self.config.hidden_dim,
            num_classes=10,
            num_timesteps=self.config.num_timesteps,
            key=params.key,
        )

    def get_optimizer(self) -> optax.GradientTransformation:
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

            opt = optax.adamw(
                learning_rate=optax.join_schedules(
                    schedules=[warmup_schedule, cosine_schedule],
                    boundaries=[self.config.warmup_steps],
                ),
                weight_decay=0.01,
            )

        else:
            opt = optax.adamw(
                learning_rate=warmup_schedule,
                weight_decay=0.01,
            )

        # Gradient accumulation.
        # return optax.MultiSteps(opt, every_k_schedule=8)

        return opt

    def get_output(self, model: DiffusionModel, batch: Batch, state: xax.State) -> Array:
        """Computes the diffusion loss.

        The model is called with (x_t, t, class_label) internally, but we need
        to wrap it to match the diffusion API which expects (x_t, t) -> output.
        """
        images_bhw = batch["image"]
        labels_b = batch["label"]

        def model_fn(x_bt: Array, t_b: Array) -> Array:
            return xax.vmap(model)(x_bt, t_b, labels_b)

        key = jax.random.PRNGKey(state.num_steps)
        loss = self.diffusion.loss(key, model_fn, images_bhw)
        return loss.mean()

    def compute_loss(self, model: DiffusionModel, batch: Batch, output: Array, state: xax.State) -> Array:
        """The output is already the loss."""
        return output

    def compute_metrics(
        self,
        model: PyTree,
        batch: Batch,
        output: Array,
        loss: Array,
        state: xax.State,
    ) -> dict[str, Array]:
        return {
            "loss": loss,
        }

    def log_heavy(
        self,
        model: DiffusionModel,
        batch: Batch,
        output: Array,
        metrics: xax.FrozenDict[str, Array],
        state: xax.State,
    ) -> None:
        """Generate and log sample images."""
        max_images = 16

        # Generate samples for each class.
        num_classes = 10
        samples_per_class = 2
        total_samples = num_classes * samples_per_class

        # Create class labels for generation.
        class_labels = jnp.repeat(jnp.arange(num_classes), samples_per_class)

        def model_fn(x_bt: Array, t_b: Array) -> Array:
            return jax.vmap(model)(x_bt, t_b, class_labels)

        key = jax.random.PRNGKey(state.num_steps)
        samples = self.diffusion.sample(
            key,
            model_fn,
            shape=(total_samples, 28, 28),
            sampling_timesteps=self.config.sampling_timesteps,
        )

        # Get the final denoised samples (first element is the final sample).
        generated_images = samples[0]

        # Create labels for the generated images.
        labels = [f"class: {c}" for c in class_labels]

        # Log the generated images.
        self.logger.log_labeled_images("generated_samples", (generated_images, labels), max_images=total_samples)

        # Also log some real images from the batch for comparison.
        real_images = batch["image"][:max_images]
        real_labels = [f"class: {int(label)}" for label in batch["label"][:max_images]]
        self.logger.log_labeled_images("real_samples", (real_images, real_labels), max_images=max_images)

    def get_dataset(self) -> Dataset:
        ds = load_dataset("ylecun/mnist", split="train")

        def process_fn(example: dict) -> dict:
            image: PILImage = example["image"]
            label: int = example["label"]
            return {
                "image": (np.array(image).astype(np.float32) / 255.0) * 2.0 - 1.0,
                "label": label,
            }

        ds = ds.map(process_fn)
        ds.set_format(type="numpy", columns=["image", "label"])
        return ds


if __name__ == "__main__":
    # python -m examples.mnist_diffusion
    MnistDiffusion.launch(
        Config(
            log_heavy_every_n_seconds=60,
            num_timesteps=1000,
            pred_mode="pred_eps",
            beta_schedule="cosine",
            sampling_timesteps=50,
        ),
    )
