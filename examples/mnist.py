"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.mnist`.
"""

from dataclasses import dataclass
from typing import Callable, Iterator

import equinox as eqx
import jax
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

import xax


@dataclass
class Config(xax.SupervisedConfig):
    batch_size: int = xax.field(128, help="The size of a minibatch")
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
    def get_model(self, params: xax.InitParams) -> Model:
        return Model(
            self.config.num_hidden_layers,
            self.config.hidden_dim,
            key=params.key,
        )

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: Model, batch: tuple[Array, Array], state: xax.State) -> Array:
        x, _ = batch
        return jax.vmap(model)(x)

    def compute_loss(self, model: Model, batch: tuple[Array, Array], output: Array, state: xax.State) -> Array:
        (_, y), yhat = batch, output
        return xax.cross_entropy(y, yhat, axis=1)

    def compute_metrics(
        self,
        model: PyTree,
        batch: tuple[Array, Array],
        output: Array,
        loss: Array,
        state: xax.State,
    ) -> dict[str, Array]:
        _, y = batch
        yhat = output.argmax(axis=1)
        return {
            "loss": loss,
            "acc": (yhat == y).astype(float).mean(),
        }

    def log_valid_step(
        self,
        model: Model,
        output: Array,
        metrics: xax.FrozenDict[str, Array],
        state: xax.State,
    ) -> None:
        max_images = 16
        batch = jax.tree.map(lambda x: jax.device_get(x[:max_images]), batch)
        (x, y), yhat = batch, output.argmax(axis=1)
        labels = [f"pred: {p}\ntrue: {t}" for p, t in zip(yhat[:max_images], y[:max_images], strict=True)]
        self.logger.log_labeled_images("predictions", (x, labels), max_images=max_images)

    def get_data_iterator(self, phase: xax.Phase, key: PRNGKeyArray) -> Iterator:
        ds = MNIST(train=phase == "train", root_dir=xax.get_data_dir() / "mnist", dtype="float32")
        images, labels = jax.device_put((ds.images, ds.labels))

        while True:
            key, ind_key = jax.random.split(key)
            indices = jax.random.randint(ind_key, (self.config.batch_size,), 0, images.shape[0])
            yield images[indices], labels[indices]


if __name__ == "__main__":
    # python -m examples.mnist
    MnistClassification.launch(Config())
