"""End-to-end tests for training functionality."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset
from jaxtyping import Array, PRNGKeyArray

import xax


class Batch(TypedDict):
    x: Array
    y: Array


@dataclass
class SimpleConfig(xax.SupervisedConfig):
    """Test configuration."""

    batch_size: int = xax.field(32, help="Batch size for training")
    hidden_dim: int = xax.field(64, help="Hidden dimension")
    learning_rate: float = xax.field(1e-3, help="Learning rate")
    max_steps: int = xax.field(10, help="Maximum number of training steps")
    save_every_n_steps: int | None = xax.field(3, help="Save checkpoint every N steps")


class SimpleModel(eqx.Module):
    """Simple test model."""

    layers: tuple[Any, ...]

    def __init__(self, config: SimpleConfig, *, key: PRNGKeyArray) -> None:
        super().__init__()

        keys = jax.random.split(key, 3)
        self.layers = (
            eqx.nn.Linear(4, config.hidden_dim, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(config.hidden_dim, 2, key=keys[1]),
            jax.nn.log_softmax,
        )

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleTask(xax.SupervisedTask[SimpleConfig]):
    """Test task for end-to-end training verification."""

    @override
    def get_model(self, params: xax.InitParams) -> SimpleModel:
        return SimpleModel(self.config, key=params.key)

    @override
    def get_optimizer(self) -> xax.Optimizer:
        return optax.adam(self.config.learning_rate)

    @override
    def compute_loss(
        self,
        model: SimpleModel,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        output = jax.vmap(model)(batch["x"])
        y_one_hot = jax.nn.one_hot(batch["y"], 2)
        loss = -jnp.mean(jnp.sum(output * y_one_hot, axis=-1))
        return loss, {}

    @override
    def get_dataset(self) -> Dataset:
        """Create a dummy dataset for testing."""
        key = jax.random.key(42)
        num_samples = 1000

        # Generate dummy data
        data = []
        for _ in range(num_samples):
            key, sample_key = jax.random.split(key)
            x = jax.random.normal(sample_key, (4,))
            y = jax.random.randint(sample_key, (), 0, 2)
            data.append(
                {
                    "x": np.array(x, dtype=np.float32),
                    "y": np.array(y, dtype=np.int32),
                }
            )

        ds = Dataset.from_list(data)
        ds.set_format(type="numpy", columns=["x", "y"])
        return ds


def test_save_load_model(tmpdir: Path) -> None:
    """Test that models can be saved and loaded correctly."""
    os.environ["DISABLE_TENSORBOARD"] = "1"

    launcher = xax.MultiCpuLauncher(num_cpus=8)

    # Train for 5 steps with checkpointing enabled
    SimpleTask.launch(
        SimpleConfig(
            max_steps=5,
            exp_dir=str(tmpdir),
            save_every_n_steps=3,
        ),
        use_cli=False,
        launcher=launcher,
    )

    # Verify checkpoint was created
    checkpoints_dir = Path(tmpdir) / "checkpoints"
    assert checkpoints_dir.exists(), "Checkpoints directory should exist"

    # Find the highest step checkpoint
    step_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda x: int(x.name.split("_")[1]) if x.name.startswith("step_") else -1,
        reverse=True,
    )
    latest_ckpt = checkpoints_dir / "latest"
    assert step_dirs or latest_ckpt.exists(), "Checkpoint should be saved"

    # Use the highest step checkpoint, or latest if no step checkpoints
    ckpt_path = step_dirs[0] if step_dirs else latest_ckpt

    # Verify checkpoint contents
    assert (ckpt_path / "model_0").exists(), "Model checkpoint should exist"
    assert (ckpt_path / "state.json").exists(), "State checkpoint should exist"
    assert (ckpt_path / "config.yaml").exists(), "Config checkpoint should exist"

    # Test loading checkpoint
    task = SimpleTask.get_task(SimpleConfig(exp_dir=str(tmpdir)), use_cli=False)
    init_params = xax.InitParams(key=jax.random.key(42))

    # Load checkpoint components
    models, opt_states, state, _ = task.load_ckpt(ckpt_path, init_params=init_params, part="all")

    assert len(models) == 1, "Should load one model"
    assert len(opt_states) == 1, "Should load one optimizer state"
    assert state.num_steps >= 5, "State should have valid num_steps"

    # Resume training from checkpoint
    SimpleTask.launch(
        SimpleConfig(
            max_steps=10,
            exp_dir=str(tmpdir),
            save_every_n_steps=3,
        ),
        use_cli=False,
        launcher=launcher,
    )

    # Verify final checkpoint exists (should be at step 9 or later)
    final_step_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda x: int(x.name.split("_")[1]) if x.name.startswith("step_") else -1,
        reverse=True,
    )
    assert final_step_dirs or latest_ckpt.exists(), "Final checkpoint should exist"


if __name__ == "__main__":
    # python -m tests.e2e.test_train_e2e
    with tempfile.TemporaryDirectory() as tmpdir:
        test_save_load_model(Path(tmpdir))
