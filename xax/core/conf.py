"""Defines base configuration functions and utilities."""

import functools
import os
from dataclasses import dataclass, field as field_base
from pathlib import Path
from typing import Any, cast

from omegaconf import II, MISSING, Container as OmegaConfContainer, OmegaConf

from xax.utils.text import show_error

FieldType = Any


def field(
    value: FieldType,
    *,
    help: str | None = None,
    static: bool = True,
    **metadata: Any,  # noqa: ANN401
) -> FieldType:
    """Short-hand function for getting a config field.

    Args:
        value: The current field's default value.
        help: An optional help string for the field.
        static: Whether or not this field is static (i.e., does not change
            during training).
        metadata: Additional metadata fields to supply.

    Returns:
        The dataclass field.
    """
    metadata["static"] = static
    if help is not None:
        metadata["help"] = help

    if callable(value):
        return field_base(default_factory=value, metadata=metadata)
    if value.__class__.__hash__ is None:
        return field_base(default_factory=lambda: value, metadata=metadata)
    return field_base(default=value, metadata=metadata)


def is_missing(cfg: Any, key: str) -> bool:  # noqa: ANN401
    """Utility function for checking if a config key is missing.

    This is for cases when you are using a raw dataclass rather than an
    OmegaConf container but want to treat them the same way.

    Args:
        cfg: The config to check
        key: The key to check

    Returns:
        Whether or not the key is missing a value in the config
    """
    if isinstance(cfg, OmegaConfContainer):
        if OmegaConf.is_missing(cfg, key):
            return True
        if OmegaConf.is_interpolation(cfg, key):
            try:
                getattr(cfg, key)
                return False
            except Exception:
                return True
    if getattr(cfg, key) is MISSING:
        return True
    return False


@dataclass(kw_only=True)
class Logging:
    hide_third_party_logs: bool = field(True, help="If set, hide third-party logs")
    log_level: str = field(II("oc.env:XAX_LOG_LEVEL,INFO"), help="The logging level to use")


@dataclass(kw_only=True)
class Triton:
    use_triton_if_available: bool = field(True, help="Use Triton if available")


@dataclass(kw_only=True)
class Experiment:
    default_random_seed: int = field(1337, help="The default random seed to use")
    max_workers: int = field(32, help="Maximum number of workers to use")


@dataclass(kw_only=True)
class Directories:
    run: str = field(II("oc.env:RUN_DIR"), help="The run directory")
    data: str = field(II("oc.env:DATA_DIR"), help="The data directory")
    pretrained_models: str = field(II("oc.env:MODEL_DIR"), help="The models directory")


@dataclass(kw_only=True)
class SlurmPartition:
    partition: str = field(MISSING, help="The partition name")
    num_nodes: int = field(1, help="The number of nodes to use")


@dataclass(kw_only=True)
class Slurm:
    launch: dict[str, SlurmPartition] = field({}, help="The available launch configurations")


@dataclass(kw_only=True)
class UserConfig:
    logging: Logging = field(Logging)
    triton: Triton = field(Triton)
    experiment: Experiment = field(Experiment)
    directories: Directories = field(Directories)
    slurm: Slurm = field(Slurm)


def user_config_path() -> Path:
    xaxrc_path_raw = os.environ.get("XAXRC_PATH", "~/.xax.yml")
    xaxrc_path = Path(xaxrc_path_raw).expanduser()
    return xaxrc_path


@functools.lru_cache(maxsize=None)
def _load_user_config_cached() -> UserConfig:
    xaxrc_path = user_config_path()
    base_cfg = OmegaConf.structured(UserConfig)

    # Writes the config file.
    if xaxrc_path.exists():
        cfg = OmegaConf.merge(base_cfg, OmegaConf.load(xaxrc_path))
    else:
        show_error(f"No config file was found in {xaxrc_path}; writing one...", important=True)
        OmegaConf.save(base_cfg, xaxrc_path)
        cfg = base_cfg

    # Looks in the current directory for a config file.
    local_cfg_path = Path("xax.yml")
    if local_cfg_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(local_cfg_path))

    return cast(UserConfig, cfg)


def load_user_config() -> UserConfig:
    """Loads the ``~/.xax.yml`` configuration file.

    Returns:
        The loaded configuration.
    """
    return _load_user_config_cached()


def get_run_dir() -> Path | None:
    config = load_user_config().directories
    if is_missing(config, "run"):
        return None
    (run_dir := Path(config.run)).mkdir(parents=True, exist_ok=True)
    return run_dir


def get_data_dir() -> Path:
    config = load_user_config().directories
    if is_missing(config, "data"):
        raise RuntimeError(
            "The data directory has not been set! You should set it in your config file "
            f"in {user_config_path()} or set the DATA_DIR environment variable."
        )
    return Path(config.data)


def get_pretrained_models_dir() -> Path:
    config = load_user_config().directories
    if is_missing(config, "pretrained_models"):
        raise RuntimeError(
            "The data directory has not been set! You should set it in your config file "
            f"in {user_config_path()} or set the MODEL_DIR environment variable."
        )
    return Path(config.pretrained_models)
