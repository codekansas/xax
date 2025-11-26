"""Trains a Gaussian diffusion model to conditionally generate MNIST digits.

Run this example with `python -m examples.mnist_diffusion`.
"""

from dataclasses import dataclass
from typing import Callable, TypedDict

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
    num_hidden_layers: int = xax.field(3, help="Number of hidden layers")
    num_timesteps: int = xax.field(1000, help="Number of diffusion timesteps")
    pred_mode: str = xax.field("pred_eps", help="Prediction mode: pred_x_0, pred_eps, or pred_v")
    beta_schedule: str = xax.field("cosine", help="Beta schedule type")
    sampling_timesteps: int = xax.field(50, help="Number of timesteps for sampling")


class DiffusionModel(eqx.Module):
    """A simple MLP-based diffusion model with class conditioning."""

    time_embed: xax.FourierEmbeddings
    class_embed: eqx.nn.Embedding
    input_proj: eqx.nn.Linear
    layers: tuple[Callable[[Array], Array], ...]
    output_proj: eqx.nn.Linear
    hidden_dim: int
    num_classes: int

    def __init__(
        self,
        image_size: int,
        hidden_dim: int,
        num_hidden_layers: int,
        num_classes: int,
        num_timesteps: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Time embedding using Fourier features
        self.time_embed = xax.FourierEmbeddings(dim=hidden_dim // 4, max_period=num_timesteps)

        # Class embedding
        embed_key, input_key, output_key = jax.random.split(key, 3)
        self.class_embed = eqx.nn.Embedding(num_classes, hidden_dim // 4, key=embed_key)

        # Input projection: image + time_emb + class_emb -> hidden_dim
        input_dim = image_size + hidden_dim // 4 + hidden_dim // 4
        self.input_proj = eqx.nn.Linear(input_dim, hidden_dim, key=input_key)

        # Hidden layers
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers: list[Callable[[Array], Array]] = []
        for i in range(num_hidden_layers):
            layers.extend([eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i]), jax.nn.swish])

        self.layers = tuple(layers)

        # Output projection: hidden_dim -> image_size
        self.output_proj = eqx.nn.Linear(hidden_dim, image_size, key=output_key)

    def __call__(self, x_t: Array, t: Array, class_label: Array | None = None) -> Array:
        """Forward pass of the diffusion model.

        Args:
            x_t: Noisy image at timestep t, shape (batch_size, image_size)
            t: Timestep, shape (batch_size,)
            class_label: Class label for conditioning, shape (batch_size,). If None, uses zeros.

        Returns:
            Predicted target (noise, x_0, or v depending on pred_mode), shape (batch_size, image_size)
        """
        batch_size = x_t.shape[0]

        # Time embedding
        t_emb = self.time_embed(t.astype(jnp.float32))  # Shape: (batch_size, hidden_dim // 4)

        # Class embedding
        if class_label is None:
            class_label = jnp.zeros((batch_size,), dtype=jnp.int32)
        class_emb = self.class_embed(class_label)  # Shape: (batch_size, hidden_dim // 4)

        # Concatenate inputs
        x_flat = x_t.reshape(batch_size, -1)  # Flatten image
        h = jnp.concatenate([x_flat, t_emb, class_emb], axis=-1)  # Shape: (batch_size, input_dim)

        # Project to hidden dimension
        h = self.input_proj(h)

        # Apply hidden layers
        for layer in self.layers:
            h = layer(h)

        # Project to output
        output = self.output_proj(h)

        return output.reshape(x_t.shape)


class MnistDiffusion(xax.SupervisedTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Initialize diffusion model
        beta_schedule: xax.DiffusionBetaSchedule = xax.cast_beta_schedule(config.beta_schedule)
        pred_mode: xax.DiffusionPredMode = config.pred_mode  # type: ignore[assignment]
        self.diffusion = xax.GaussianDiffusion(
            beta_schedule=beta_schedule,
            num_beta_steps=config.num_timesteps,
            pred_mode=pred_mode,
        )
        # Reset parameters to initialize bar_alpha
        self.diffusion = self.diffusion.reset_parameters()

    def get_model(self, params: xax.InitParams) -> DiffusionModel:
        return DiffusionModel(
            image_size=28 * 28,
            hidden_dim=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
            num_classes=10,
            num_timesteps=self.config.num_timesteps,
            key=params.key,
        )

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: DiffusionModel, batch: Batch, state: xax.State) -> Array:
        """Computes the diffusion loss.

        The model is called with (x_t, t, class_label) internally, but we need
        to wrap it to match the diffusion API which expects (x_t, t) -> output.
        """
        # Generate a random key based on the step number
        images = batch["image"]
        labels = batch["label"]

        # Create a wrapped model that includes class conditioning
        def model_fn(x_t: Array, t: Array) -> Array:
            return model(x_t, t, labels)

        # Compute loss using the diffusion model.
        key = jax.random.PRNGKey(state.num_steps[0])
        loss = self.diffusion.loss(key, model_fn, images)
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
        # Generate a random key based on the step number
        key = jax.random.PRNGKey(int(state.num_steps.item()))

        # Generate samples for each class
        num_classes = 10
        samples_per_class = 2
        total_samples = num_classes * samples_per_class

        # Create class labels for generation
        class_labels = jnp.repeat(jnp.arange(num_classes), samples_per_class)

        # Generate samples
        # Create a model function that includes class conditioning
        # The diffusion API expects (x_t, t) -> output, but we need class labels
        # We'll create a closure that captures the class labels
        def model_fn(x_t: Array, t: Array) -> Array:
            # x_t shape: (total_samples, 28, 28)
            # t shape: (total_samples,)
            # class_labels shape: (total_samples,)
            # Use vmap to apply the model to each sample with its corresponding class label
            return jax.vmap(model, in_axes=(0, 0, 0))(x_t, t, class_labels)

        sample_key, _ = jax.random.split(key)
        samples = self.diffusion.sample(
            sample_key,
            model_fn,
            shape=(total_samples, 28, 28),
            sampling_timesteps=self.config.sampling_timesteps,
        )

        # Get the final denoised samples (first element is the final sample)
        generated_images = samples[0]  # Shape: (total_samples, 28, 28)

        # Convert to numpy for logging and normalize from [-1, 1] to [0, 1]
        generated_images = jax.device_get(generated_images)
        generated_images = (generated_images + 1.0) / 2.0
        generated_images = np.clip(generated_images, 0, 1)

        # Create labels for the generated images
        labels = [f"class: {c}" for c in class_labels]

        # Log the generated images
        self.logger.log_labeled_images("generated_samples", (generated_images, labels), max_images=total_samples)

        # Also log some real images from the batch for comparison
        # Convert from [-1, 1] to [0, 1] for display
        real_images = jax.device_get(batch["image"][:max_images])
        real_images = (real_images + 1.0) / 2.0
        real_images = np.clip(real_images, 0, 1)
        real_labels = [f"class: {int(label)}" for label in batch["label"][:max_images]]
        self.logger.log_labeled_images("real_samples", (real_images, real_labels), max_images=max_images)

    def get_dataset(self) -> Dataset:
        ds = load_dataset("ylecun/mnist", split="train")

        def process_fn(example: dict) -> dict:
            image: PILImage = example["image"]
            label: int = example["label"]
            # Normalize to [-1, 1] for diffusion models
            image_array = np.array(image).astype(np.float32) / 255.0
            image_array = image_array * 2.0 - 1.0
            return {
                "image": image_array,
                "label": label,
            }

        ds = ds.map(process_fn)
        ds.set_format(type="numpy", columns=["image", "label"])
        return ds


if __name__ == "__main__":
    # python -m examples.mnist_diffusion
    MnistDiffusion.launch(
        Config(
            log_heavy_every_n_seconds=300,  # Log samples every 5 minutes
            num_timesteps=1000,
            pred_mode="pred_eps",
            beta_schedule="cosine",
            sampling_timesteps=50,
        ),
    )
