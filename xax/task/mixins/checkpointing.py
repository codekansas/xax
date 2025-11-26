"""Defines a mixin for handling model checkpointing."""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, Self, Sequence, TypeVar, cast, overload

import equinox as eqx
import jax
import optax
import orbax.checkpoint as ocp
from etils import epath
from jaxtyping import PyTree
from omegaconf import DictConfig, OmegaConf

from xax.core.conf import field
from xax.core.state import State
from xax.nn.parallel import is_master
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin

logger = logging.getLogger(__name__)

CheckpointPart = Literal["model", "opt_state", "state", "config", "model_state_config", "all"]


def get_ckpt_path(exp_dir: Path, state: State | None = None) -> Path:
    """Defines the path to the checkpoint for a given state.

    Args:
        exp_dir: The experiment directory
        state: The current trainer state

    Returns:
        The path to the checkpoint directory.
    """
    if state is None:
        return exp_dir / "checkpoints" / "latest"
    return exp_dir / "checkpoints" / f"step_{int(state.num_steps.item())}"


@jax.tree_util.register_dataclass
@dataclass
class CheckpointingConfig(ArtifactsConfig):
    save_every_n_steps: int | None = field(None, help="Save a checkpoint every N steps")
    save_every_n_seconds: float | None = field(60.0 * 60.0, help="Save a checkpoint every N seconds")
    only_save_most_recent: bool = field(True, help="Only keep the most recent checkpoint")
    load_from_ckpt_path: str | None = field(None, help="If set, load initial model weights from this path")


Config = TypeVar("Config", bound=CheckpointingConfig)


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["all"],
    model_templates: Sequence[PyTree],
    opt_state_templates: Sequence[optax.OptState],
) -> tuple[list[PyTree], list[optax.OptState], State, DictConfig]: ...


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["model_state_config"],
    model_templates: Sequence[PyTree],
) -> tuple[list[PyTree], State, DictConfig]: ...


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["model"],
    model_templates: Sequence[PyTree],
) -> list[PyTree]: ...


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["opt_state"],
    opt_state_templates: Sequence[optax.OptState],
) -> list[optax.OptState]: ...


@overload
def load_ckpt(path: Path, *, part: Literal["state"]) -> State: ...


@overload
def load_ckpt(path: Path, *, part: Literal["config"]) -> DictConfig: ...


def load_ckpt(
    path: str | Path,
    *,
    part: CheckpointPart = "model",
    model_templates: Sequence[PyTree] | None = None,
    opt_state_templates: Sequence[optax.OptState] | None = None,
) -> (
    tuple[list[PyTree], list[optax.GradientTransformation], list[optax.OptState], State, DictConfig]
    | tuple[list[PyTree], State, DictConfig]
    | list[PyTree]
    | list[optax.OptState]
    | State
    | DictConfig
):
    ckpt_path = epath.Path(path)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    def get_model() -> list[PyTree]:
        if model_templates is None:
            raise ValueError("model_template must be provided to load model weights")
        models: list[PyTree] = []
        for i, model_template in enumerate(model_templates):
            model_path = ckpt_path / f"model_{i}"
            if not model_path.exists():
                raise ValueError(f"Checkpoint does not contain a model file: {model_path}")
            restored_arrays = checkpointer.restore(model_path, item=model_template)
            models.append(eqx.combine(restored_arrays, model_template))
        return models

    def get_opt_state() -> list[optax.OptState]:
        if opt_state_templates is None:
            raise ValueError("opt_state_template must be provided to load optimizer state")
        opt_states: list[optax.OptState] = []
        for i, opt_state_template in enumerate(opt_state_templates):
            opt_state_path = ckpt_path / f"opt_state_{i}"
            if not opt_state_path.exists():
                raise ValueError(f"Checkpoint does not contain an optimizer state file: {opt_state_path}")
            restored_opt_state = checkpointer.restore(opt_state_path, item=opt_state_template)
            opt_states.append(restored_opt_state)
        return opt_states

    def get_state() -> State:
        state_path = ckpt_path / "state.json"
        if not state_path.exists():
            raise ValueError(f"Checkpoint does not contain a state file: {state_path}")
        with state_path.open() as f:
            return State.from_dict(**json.load(f))

    def get_config() -> DictConfig:
        config_path = ckpt_path / "config.yaml"
        if not config_path.exists():
            raise ValueError(f"Checkpoint does not contain a config file: {config_path}")
        # Convert epath.Path to string for OmegaConf.load
        return cast(DictConfig, OmegaConf.load(str(config_path)))

    match part:
        case "model":
            return get_model()
        case "opt_state":
            return get_opt_state()
        case "state":
            return get_state()
        case "config":
            return get_config()
        case "model_state_config":
            return get_model(), get_state(), get_config()
        case "all":
            return get_model(), get_opt_state(), get_state(), get_config()
        case _:
            raise ValueError(f"Invalid checkpoint part: {part}")


class CheckpointingMixin(ArtifactsMixin[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__last_ckpt_time = 0.0

    def get_ckpt_path(self, state: State | None = None) -> Path:
        return get_ckpt_path(self.exp_dir, state)

    def get_init_ckpt_path(self) -> Path | None:
        if self._exp_dir is not None:
            ckpt_path = self.get_ckpt_path()
            # Check if latest checkpoint exists
            if ckpt_path.exists() and ckpt_path.is_dir():
                return ckpt_path
            # Also check for step-based checkpoints
            checkpoints_dir = self.exp_dir / "checkpoints"
            if checkpoints_dir.exists():
                step_dirs = sorted(
                    [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
                    reverse=True,
                )
                if step_dirs:
                    return step_dirs[0]
        if self.config.load_from_ckpt_path is not None:
            ckpt_path = Path(self.config.load_from_ckpt_path)
            assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
            return ckpt_path
        return None

    def should_checkpoint(self, state: State) -> bool:
        if self.config.save_every_n_steps is not None:
            if state.num_steps % self.config.save_every_n_steps == 0:
                return True
        if self.config.save_every_n_seconds is not None:
            last_time, cur_time = self.__last_ckpt_time, state.elapsed_time_s.item()
            if cur_time - last_time >= self.config.save_every_n_seconds:
                self.__last_ckpt_time = cur_time
                return True
        return False

    def save_checkpoint(
        self,
        models: Sequence[PyTree] | None = None,
        optimizers: Sequence[optax.GradientTransformation] | None = None,
        opt_states: Sequence[optax.OptState] | None = None,
        aux_data: PyTree | None = None,
        state: State | None = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            models: The models to save
            optimizers: The optimizers to save
            opt_states: The optimizer states to save
            aux_data: Additional data to save
            state: The current training state

        Returns:
            Path to the saved checkpoint
        """
        ckpt_path = self.get_ckpt_path(state)

        if not is_master():
            return ckpt_path

        # Gets the path to the last checkpoint
        logger.info("Saving checkpoint to %s", ckpt_path)
        last_ckpt_path = self.get_ckpt_path()
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)

        # Potentially removes the last checkpoint
        if last_ckpt_path.exists() and self.config.only_save_most_recent:
            if last_ckpt_path.is_symlink():
                last_ckpt_path.unlink()
            elif last_ckpt_path.is_dir():
                shutil.rmtree(last_ckpt_path)
            elif last_ckpt_path.is_file():
                last_ckpt_path.unlink()

        # Use orbax checkpointer
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        ckpt_epath = epath.Path(ckpt_path)
        ckpt_epath.mkdir(parents=True, exist_ok=True)

        if models is not None:
            for i, model in enumerate(models):
                model_path = ckpt_epath / f"model_{i}"
                model_arrays = eqx.filter(model, eqx.is_array)
                if model_path.exists():
                    shutil.rmtree(model_path)
                checkpointer.save(model_path, model_arrays)

        if opt_states is not None:
            for i, opt_state in enumerate(opt_states):
                opt_state_path = ckpt_epath / f"opt_state_{i}"
                if opt_state_path.exists():
                    shutil.rmtree(opt_state_path)
                checkpointer.save(opt_state_path, opt_state)

        if aux_data is not None:
            aux_path = ckpt_epath / "aux_data"
            checkpointer.save(aux_path, aux_data)

        if state is not None:
            state_path = ckpt_epath / "state.json"
            with state_path.open("w") as f:
                json.dump(state.to_dict(), f, indent=2)

        config_path = ckpt_epath / "config.yaml"
        with config_path.open("w") as f:
            f.write(OmegaConf.to_yaml(self.config, sort_keys=True))

        # Update the symlink to the new checkpoint
        if last_ckpt_path != ckpt_path:
            last_ckpt_path.unlink(missing_ok=True)
            try:
                last_ckpt_path.symlink_to(ckpt_path.relative_to(last_ckpt_path.parent))
            except FileExistsError:
                logger.exception("Exception while trying to update %s", ckpt_path)

        # Calls the base callback
        self.on_after_checkpoint_save(ckpt_path, state)

        return ckpt_path

    @classmethod
    def load_config(cls, ckpt_path: str | Path) -> Config:
        return cls.get_config(load_ckpt(Path(ckpt_path), part="config"), use_cli=False)

    @classmethod
    def load_task(cls, ckpt_path: str | Path) -> Self:
        return cls(cls.load_config(ckpt_path))
