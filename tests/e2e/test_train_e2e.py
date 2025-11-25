"""End-to-end tests for training functionality."""

import tempfile
from dataclasses import dataclass
from typing import Any, Iterator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray

import xax


@dataclass
class SimpleConfig(xax.SupervisedConfig):
    """Test configuration."""

    batch_size: int = xax.field(32, help="Batch size for training")
    hidden_dim: int = xax.field(64, help="Hidden dimension")
    learning_rate: float = xax.field(1e-3, help="Learning rate")
    max_steps: int = xax.field(10, help="Maximum number of training steps")


class SimpleModel(eqx.Module):
    """Simple test model."""

    layers: list[Any]

    def __init__(self, config: SimpleConfig, *, key: PRNGKeyArray) -> None:
        super().__init__()

        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(4, config.hidden_dim, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(config.hidden_dim, 2, key=keys[1]),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleTask(xax.SupervisedTask[SimpleConfig]):
    """Test task for end-to-end training verification."""

    def get_model(self, params: xax.InitParams) -> SimpleModel:
        return SimpleModel(self.config, key=params.key)

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: SimpleModel, batch: Tuple[Array, Array], state: xax.State) -> Array:
        x, _ = batch
        return jax.vmap(model)(x)

    def compute_loss(self, model: SimpleModel, batch: Tuple[Array, Array], output: Array, state: xax.State) -> Array:
        _, y = batch
        return -jnp.mean(jnp.sum(output * y, axis=-1))

    def get_data_iterator(self) -> Iterator[Tuple[Array, Array]]:
        key = jax.random.PRNGKey(42)
        while True:
            key, batch_key = jax.random.split(key)
            x = jax.random.normal(batch_key, (self.config.batch_size, 4))
            y = jax.nn.one_hot(jax.random.randint(batch_key, (self.config.batch_size,), 0, 2), 2)
            yield x, y


def test_save_load_model() -> None:
    """Test that models can be saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SimpleTask.launch(
            SimpleConfig(
                max_steps=5,
                exp_dir=tmpdir,
            ),
            use_cli=False,
        )

        SimpleTask.launch(
            SimpleConfig(
                max_steps=10,
                exp_dir=tmpdir,
            ),
            use_cli=False,
        )


if __name__ == "__main__":
    # python -m tests.e2e.test_train_e2e
    test_save_load_model()
