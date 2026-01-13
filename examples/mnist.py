#!/usr/bin/env -S uv run --no-project --script
"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.mnist`.
"""

from dataclasses import dataclass
from typing import Callable, TypedDict, override

import equinox as eqx
import jax
import numpy as np
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from PIL.Image import Image as PILImage

import xax


class Batch(TypedDict):
    image: Array
    label: Array


@dataclass
class Config(xax.SupervisedConfig):
    learning_rate: float = xax.field(1e-3, help="The learning rate")
    hidden_dim: int = xax.field(512, help="Hidden layer dimension")
    num_hidden_layers: int = xax.field(2, help="Number of hidden layers")


class Model(eqx.Module):
    num_hidden_layers: int
    hidden_dim: int
    layers: tuple[Callable[[Array], Array], ...]

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        # Input and output dimensions
        input_dim = 28 * 28
        output_dim = 10

        # Split the PRNG key for all layers
        keys = jax.random.split(key, num_hidden_layers + 1)

        # Build layers list
        layers: list[Callable[[Array], Array]] = []
        current_dim = input_dim

        # Add hidden layers
        for i in range(num_hidden_layers):
            layers.extend([eqx.nn.Linear(current_dim, hidden_dim, key=keys[i]), jax.nn.relu])
            current_dim = hidden_dim

        # Add output layer
        layers.extend([eqx.nn.Linear(current_dim, output_dim, key=keys[-1]), jax.nn.log_softmax])

        self.layers = tuple(layers)

    def __call__(self, x: Array) -> Array:
        x = x.reshape(28 * 28)
        for layer in self.layers:
            x = layer(x)
        return x


class MnistClassification(xax.SupervisedTask[Config]):
    @override
    def get_model(self, params: xax.InitParams) -> Model:
        return Model(
            self.config.num_hidden_layers,
            self.config.hidden_dim,
            key=params.key,
        )

    @override
    def get_optimizer(self) -> xax.Optimizer:
        return optax.adam(self.config.learning_rate)

    @override
    def compute_loss(
        self,
        model: Model,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        output = jax.vmap(model)(batch["image"])
        y = batch["label"]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y)

        # Compute metrics
        yhat = output.argmax(axis=1)
        metrics: dict[str, xax.Metric] = {}
        metrics["acc"] = xax.Scalar((yhat == y).astype(float).mean())

        if heavy:
            max_images = 16
            x = batch["image"][:max_images]
            metrics["predictions"] = xax.LabeledImages(
                x,
                {"pred": yhat[:max_images], "true": y[:max_images]},
                max_images=max_images,
                target_resolution=(64, 64),
            )

        return loss.mean(), metrics

    @override
    def get_dataset(self) -> Dataset:
        ds = load_dataset("ylecun/mnist", split="train")

        def process_fn(example: dict) -> dict:
            image: PILImage = example["image"]
            label: int = example["label"]
            return {
                "image": np.array(image).astype(np.float32) / 255.0,
                "label": label,
            }

        ds = ds.map(process_fn)
        ds.set_format(type="numpy", columns=["image", "label"])
        return ds


if __name__ == "__main__":
    MnistClassification.launch(
        Config(
            batch_size=256,
            log_heavy_every_n_seconds=120,
            # MNIST dataset is very small and this greatly improves throughput.
            load_in_memory=True,
            gradient_accumulation_steps=64,
        ),
    )
