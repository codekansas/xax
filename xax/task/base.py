"""Defines the base task interface.

This interface is built upon by a large number of other interfaces which
compose various functionality into a single cohesive unit. The base task
just stores the configuration and provides hooks which are overridden by
upstream classes.
"""

import dataclasses
import functools
import inspect
import logging
import sys
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, Self, Sequence, TypeVar, cast

import jax

from xax.core.state import State
from xax.utils.structured_config import (
    apply_dotted_override,
    dataclass_from_mapping,
    deep_merge,
    is_missing_value,
    load_yaml,
    parse_override_token,
    render_dataclass_help,
    to_primitive,
    to_yaml_text,
)
from xax.utils.text import camelcase_to_snakecase

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class BaseConfig:
    pass


Config = TypeVar("Config", bound=BaseConfig)

RawConfigType = BaseConfig | dict[str, Any] | str | Path


def _load_as_dict(path: str | Path) -> dict[str, object]:
    return load_yaml(path)


def get_config(cfg: RawConfigType, task_path: Path) -> dict[str, object]:
    if isinstance(cfg, (str, Path)):
        cfg = Path(cfg)
        if cfg.exists():
            cfg = _load_as_dict(cfg)
        elif task_path is not None and len(cfg.parts) == 1 and (other_cfg_path := task_path.parent / cfg).exists():
            cfg = _load_as_dict(other_cfg_path)
        else:
            raise FileNotFoundError(f"Could not find config file at {cfg}!")
    elif isinstance(cfg, dict):
        cfg = {str(key): value for key, value in cfg.items()}
    elif is_dataclass(cfg):
        cfg_payload = to_primitive(cfg, preserve_missing=True)
        if not isinstance(cfg_payload, dict):
            raise TypeError(f"Expected dataclass config payload to be a mapping, got {type(cfg_payload)!r}")
        cfg = {str(key): value for key, value in cfg_payload.items()}
    return cast(dict[str, object], cfg)


def _default_config_payload(config_class: type[Config]) -> dict[str, object]:
    default_config = config_class()
    payload = to_primitive(default_config, preserve_missing=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected default config payload to be a mapping, got {type(payload)!r}")
    return {str(key): value for key, value in payload.items()}


def _missing_field_paths(value: object, prefix: str = "") -> list[str]:
    if is_missing_value(value):
        return [prefix]
    if is_dataclass(value):
        missing: list[str] = []
        for field in dataclasses.fields(value):
            field_prefix = field.name if not prefix else f"{prefix}.{field.name}"
            missing.extend(_missing_field_paths(getattr(value, field.name), field_prefix))
        return missing
    if isinstance(value, dict):
        missing = []
        for key, item in value.items():
            key_str = str(key)
            field_prefix = key_str if not prefix else f"{prefix}.{key_str}"
            missing.extend(_missing_field_paths(item, field_prefix))
        return missing
    if isinstance(value, list):
        missing = []
        for idx, item in enumerate(value):
            field_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            missing.extend(_missing_field_paths(item, field_prefix))
        return missing
    return []


class BaseTask(Generic[Config]):
    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

    def on_step_start(self) -> None:
        pass

    def on_step_end(self) -> None:
        pass

    def on_training_start(self) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_after_checkpoint_save(self, ckpt_path: Path, state: State | None) -> State | None:
        return state

    def add_logger_handlers(self, logger: logging.Logger) -> None:
        pass

    @functools.cached_property
    def task_class_name(self) -> str:
        return self.__class__.__name__

    @functools.cached_property
    def task_name(self) -> str:
        return camelcase_to_snakecase(self.task_class_name)

    @functools.cached_property
    def task_path(self) -> Path:
        try:
            return Path(inspect.getfile(self.__class__))
        except OSError:
            logger.warning("Could not resolve task path for %s, returning current working directory")
            return Path.cwd()

    @functools.cached_property
    def task_module(self) -> str:
        if (mod := inspect.getmodule(self.__class__)) is None:
            raise RuntimeError(f"Could not find module for task {self.__class__}!")
        if (spec := mod.__spec__) is None:
            raise RuntimeError(f"Could not find spec for module {mod}!")
        return spec.name

    @property
    def task_key(self) -> str:
        return f"{self.task_module}.{self.task_class_name}"

    @classmethod
    def from_task_key(cls, task_key: str) -> type[Self]:
        task_module, task_class_name = task_key.rsplit(".", 1)
        try:
            mod = __import__(task_module, fromlist=[task_class_name])
        except ImportError as e:
            raise ImportError(f"Could not import module {task_module} for task {task_key}") from e
        if not hasattr(mod, task_class_name):
            raise RuntimeError(f"Could not find class {task_class_name} in module {task_module}")
        task_class = getattr(mod, task_class_name)
        if not issubclass(task_class, cls):
            raise RuntimeError(f"Class {task_class_name} in module {task_module} is not a subclass of {cls}")
        return task_class

    def debug(self) -> bool:
        return False

    @property
    def debugging(self) -> bool:
        return self.debug()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, t: type[BaseException] | None, e: BaseException | None, tr: TracebackType | None) -> None:
        pass

    @classmethod
    def get_config_class(cls) -> type[Config]:
        """Recursively retrieves the config class from the generic type.

        Returns:
            The parsed config class.

        Raises:
            ValueError: If the config class cannot be found, usually meaning
            that the generic class has not been used correctly.
        """
        if hasattr(cls, "__orig_bases__"):
            for base in cast(Sequence[type], cls.__orig_bases__):
                if hasattr(base, "__args__"):
                    for arg in cast(Sequence[type], base.__args__):
                        if isinstance(arg, TypeVar) and arg.__bound__ is not None:
                            arg = arg.__bound__
                        if issubclass(arg, BaseConfig):
                            return cast(type[Config], arg)

        raise ValueError(
            "The config class could not be parsed from the generic type, which usually means that the task is not "
            "being instantiated correctly. Your class should be defined as follows:\n\n"
            "  class ExampleTask(xax.SupervisedTask[Config]):\n      ...\n\nThis lets the both the task and the type "
            "checker know what config the task is using."
        )

    @classmethod
    def get_config(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> Config:
        """Builds the structured config from the provided config classes.

        Args:
            cfgs: The config classes to merge. If a string or Path is provided,
                it will be loaded as a YAML file.
            use_cli: Whether to allow additional overrides from the CLI.

        Returns:
            The merged configs.
        """
        try:
            task_path = Path(inspect.getfile(cls))
        except OSError:
            logger.warning("Could not resolve task path for %s, returning current working directory", cls.__name__)
            task_path = Path.cwd()
        config_class = cls.get_config_class()
        cfg_payload = _default_config_payload(config_class)
        for other_cfg in cfgs:
            cfg_payload = cast(dict[str, object], deep_merge(cfg_payload, get_config(other_cfg, task_path)))
        if use_cli:
            args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
            if "-h" in args or "--help" in args:
                sys.stdout.write(render_dataclass_help(config_class, prog=cls.__name__))
                sys.stdout.write("\n")
                sys.stdout.flush()
                sys.exit(0)

            # Attempts to load any paths as configs.
            is_path = [Path(arg).is_file() or (task_path / arg).is_file() for arg in args]
            paths = [arg for arg, is_path in zip(args, is_path, strict=True) if is_path]
            non_paths = [arg for arg, is_path in zip(args, is_path, strict=True) if not is_path]
            if paths:
                for path in paths:
                    cfg_payload = cast(dict[str, object], deep_merge(cfg_payload, get_config(path, task_path)))
            for override_token in non_paths:
                key, value = parse_override_token(override_token)
                apply_dotted_override(cfg_payload, key, value)

        config = dataclass_from_mapping(config_class, cfg_payload)
        if missing_paths := _missing_field_paths(config):
            missing_list = ", ".join(missing_paths)
            raise ValueError(f"Config has missing required value(s): {missing_list}")
        return config

    @classmethod
    def config_str(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> str:
        return to_yaml_text(cls.get_config(*cfgs, use_cli=use_cli), sort_keys=True)

    @classmethod
    def get_task(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> Self:
        """Builds the task from the provided config classes.

        Args:
            cfgs: The config classes to merge. If a string or Path is provided,
                it will be loaded as a YAML file.
            use_cli: Whether to allow additional overrides from the CLI.

        Returns:
            The task.
        """
        cfg = cls.get_config(*cfgs, use_cli=use_cli)
        return cls(cfg)
