# mypy: disable-error-code="import-not-found"
"""This example demonstrates the logging features of Xax."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import equinox as eqx
import jax
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray

import xax

try:
    import trimesh

except ModuleNotFoundError as err:
    raise ImportError("trimesh is required to run this example. Please install it with `pip install trimesh`.") from err


@dataclass
class Config(xax.SupervisedConfig):
    batch_size: int = xax.field(128, help="The size of a minibatch")
    learning_rate: float = xax.field(1e-3, help="The learning rate")
    dims: int = xax.field(16, help="The dimension of the model")


class Model(eqx.Module):
    layer: eqx.nn.Linear

    def __init__(self, dims: int, *, key: PRNGKeyArray) -> None:
        super().__init__()

        self.layer = eqx.nn.Linear(dims, dims, key=key)

    def __call__(self, x: Array) -> Array:
        return self.layer(x)


class LoggingExample(xax.SupervisedTask[Config]):
    def get_model(self, params: xax.InitParams) -> Model:
        return Model(self.config.dims, key=params.key)

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: Model, batch: tuple[Array, Array], state: xax.State) -> Array:
        x, _ = batch
        return jax.vmap(model)(x)

    def compute_loss(self, model: Model, batch: tuple[Array, Array], output: Array, state: xax.State) -> Array:
        (_, y), yhat = batch, output
        return xax.get_norm(y - yhat, "l2").sum()

    def log_valid_step(
        self,
        model: Model,
        batch: tuple[Array, Array],
        output: Array,
        metrics: xax.FrozenDict[str, Array],
        state: xax.State,
    ) -> None:
        # Tests logging a 3D mesh.
        mesh = trimesh.load_mesh(Path(__file__).parent / "assets" / "teapot.stl")
        assert isinstance(mesh, trimesh.Trimesh)
        self.logger.log_mesh("test_mesh", vertices=np.array(mesh.vertices), faces=np.array(mesh.faces))

    def get_data_iterator(self, key: PRNGKeyArray) -> Iterator[tuple[Array, Array]]:
        xkey, ykey = jax.random.split(key)
        xs = jax.random.normal(xkey, (self.config.batch_size, self.config.dims))
        ys = jax.random.normal(ykey, (self.config.batch_size, self.config.dims))

        while True:
            yield xs, ys


if __name__ == "__main__":
    # python -m examples.logging
    LoggingExample.launch(Config())
