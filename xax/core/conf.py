"""Defines base configuration functions and utilities."""

import copy
import functools
import os
from dataclasses import dataclass, field as field_base
from pathlib import Path
from typing import Any

from xax.utils.structured_config import MISSING, deep_merge, load_yaml, merge_config_sources, save_yaml, to_primitive
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

    metadata.setdefault("_xax_help_default_value", copy.deepcopy(value) if value.__class__.__hash__ is None else value)

    if value.__class__.__hash__ is None:
        return field_base(default_factory=lambda value=value: copy.deepcopy(value), metadata=metadata)
    return field_base(default=value, metadata=metadata)


def is_missing(cfg: Any, key: str) -> bool:  # noqa: ANN401
    """Utility function for checking if a config key is missing.

    This supports both dataclass-style configs and plain dictionaries.

    Args:
        cfg: The config to check
        key: The key to check

    Returns:
        Whether or not the key is missing a value in the config
    """
    value = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key)
    if value is MISSING or value is None:
        return True
    return False


@dataclass(kw_only=True)
class Logging:
    hide_third_party_logs: bool = field(True, help="If set, hide third-party logs")
    log_level: str = field("INFO", help="The logging level to use")


@dataclass(kw_only=True)
class Triton:
    use_triton_if_available: bool = field(True, help="Use Triton if available")


@dataclass(kw_only=True)
class Experiment:
    default_random_seed: int = field(1337, help="The default random seed to use")
    max_workers: int = field(32, help="Maximum number of workers to use")


@dataclass(kw_only=True)
class Directories:
    runs: str | None = field(None, help="Directory containing all training runs")
    experiments: str | None = field(None, help="Directory containing experiment-monitor sessions")
    data: str | None = field(None, help="The data directory")
    pretrained_models: str | None = field(None, help="The models directory")


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
    if (xaxrc_path_raw := os.environ.get("XAXRC_PATH")) is not None:
        xaxrc_path = Path(xaxrc_path_raw).expanduser()
        if xaxrc_path.suffix in (".yml", ".yaml"):
            return xaxrc_path
        return xaxrc_path / "config.yml"
    return get_user_global_dir() / "config.yml"


def legacy_user_config_path() -> Path:
    return Path("~/.xax.yml").expanduser()


def get_user_global_dir() -> Path:
    if (xax_home_raw := os.environ.get("XAX_HOME")) is not None:
        return Path(xax_home_raw).expanduser()
    if (xaxrc_path_raw := os.environ.get("XAXRC_PATH")) is not None:
        xaxrc_path = Path(xaxrc_path_raw).expanduser()
        if xaxrc_path.suffix in (".yml", ".yaml"):
            return xaxrc_path.parent
        return xaxrc_path
    return Path("~/.xax").expanduser()


@functools.lru_cache(maxsize=None)
def _load_user_config_cached() -> UserConfig:
    xaxrc_path = user_config_path()
    xaxrc_path.parent.mkdir(parents=True, exist_ok=True)
    base_cfg = UserConfig()
    base_payload = to_primitive(base_cfg)
    if not isinstance(base_payload, dict):
        raise TypeError(f"Invalid user config payload type: {type(base_payload)!r}")
    merged_payload: dict[str, object] = {str(key): value for key, value in base_payload.items()}

    def merge_payloads(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
        merged = deep_merge(base, override)
        if not isinstance(merged, dict):
            raise TypeError(f"Expected merged payload to be a mapping, got {type(merged)!r}")
        return {str(key): value for key, value in merged.items()}

    # Writes the config file.
    if xaxrc_path.exists():
        merged_payload = merge_payloads(merged_payload, load_yaml(xaxrc_path))
    elif "XAXRC_PATH" not in os.environ and (legacy_path := legacy_user_config_path()).exists():
        show_error(f"Migrating legacy config from {legacy_path} to {xaxrc_path}", important=True)
        legacy_path.replace(xaxrc_path)
        merged_payload = merge_payloads(merged_payload, load_yaml(xaxrc_path))
    else:
        show_error(f"No config file was found in {xaxrc_path}; writing one...", important=True)
        save_yaml(xaxrc_path, base_cfg)

    # Looks in the current directory for a config file.
    local_cfg_path = Path("xax.yml")
    if local_cfg_path.exists():
        merged_payload = merge_payloads(merged_payload, load_yaml(local_cfg_path))

    return merge_config_sources(UserConfig, [merged_payload])


def load_user_config() -> UserConfig:
    """Loads the user configuration file (default: ``~/.xax/config.yml``).

    Returns:
        The loaded configuration.
    """
    return _load_user_config_cached()


def get_runs_dir() -> Path | None:
    config = load_user_config().directories
    if is_missing(config, "runs"):
        return None
    runs_value = config.runs
    if runs_value is None:
        raise RuntimeError("Expected non-empty `directories.runs` value")
    (runs_dir := Path(runs_value)).mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_experiments_dir() -> Path | None:
    config = load_user_config().directories
    if is_missing(config, "experiments"):
        return None
    experiments_value = config.experiments
    if experiments_value is None:
        raise RuntimeError("Expected non-empty `directories.experiments` value")
    (experiments_dir := Path(experiments_value)).mkdir(parents=True, exist_ok=True)
    return experiments_dir


def get_data_dir() -> Path:
    config = load_user_config().directories
    if is_missing(config, "data"):
        raise RuntimeError(
            "The data directory has not been set! You should set it in your config file "
            f"in {user_config_path()}."
        )
    data_value = config.data
    if data_value is None:
        raise RuntimeError("Expected non-empty `directories.data` value")
    return Path(data_value)


def get_pretrained_models_dir() -> Path:
    config = load_user_config().directories
    if is_missing(config, "pretrained_models"):
        raise RuntimeError(
            "The models directory has not been set! You should set it in your config file "
            f"in {user_config_path()}."
        )
    models_value = config.pretrained_models
    if models_value is None:
        raise RuntimeError("Expected non-empty `directories.pretrained_models` value")
    return Path(models_value)
