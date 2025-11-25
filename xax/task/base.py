"""Defines the base task interface.

This interface is built upon by a large number of other interfaces which
compose various functionality into a single cohesive unit. The base task
just stores the configuration and provides hooks which are overridden by
upstream classes.
"""

import functools
import inspect
import logging
import sys
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from types import TracebackType
from typing import Generic, Self, TypeVar, cast

import jax
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import SCMode

from xax.core.state import State
from xax.utils.text import camelcase_to_snakecase

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class BaseConfig:
    pass


Config = TypeVar("Config", bound=BaseConfig)

RawConfigType = BaseConfig | dict | DictConfig | str | Path


def _load_as_dict(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config file at {path} must be a dictionary, not {type(cfg)}!")
    return cfg


def get_config(cfg: RawConfigType, task_path: Path) -> DictConfig:
    if isinstance(cfg, (str, Path)):
        cfg = Path(cfg)
        if cfg.exists():
            cfg = _load_as_dict(cfg)
        elif task_path is not None and len(cfg.parts) == 1 and (other_cfg_path := task_path.parent / cfg).exists():
            cfg = _load_as_dict(other_cfg_path)
        else:
            raise FileNotFoundError(f"Could not find config file at {cfg}!")
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)
    return cast(DictConfig, cfg)


class BaseTask(Generic[Config]):
    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

    def on_step_start(self, state: State) -> State:
        return state

    def on_step_end(self, state: State) -> State:
        return state

    def on_training_start(self, state: State) -> State:
        return state

    def on_training_end(self, state: State) -> State:
        return state

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

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
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
            for base in cls.__orig_bases__:
                if hasattr(base, "__args__"):
                    for arg in base.__args__:
                        if isinstance(arg, TypeVar) and arg.__bound__ is not None:
                            arg = arg.__bound__
                        if issubclass(arg, BaseConfig):
                            return arg

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
        cfg = OmegaConf.structured(cls.get_config_class())
        cfg = OmegaConf.merge(cfg, *(get_config(other_cfg, task_path) for other_cfg in cfgs))
        if use_cli:
            args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
            if "-h" in args or "--help" in args:
                sys.stdout.write(OmegaConf.to_yaml(cfg, sort_keys=True))
                sys.stdout.flush()
                sys.exit(0)

            # Attempts to load any paths as configs.
            is_path = [Path(arg).is_file() or (task_path / arg).is_file() for arg in args]
            paths = [arg for arg, is_path in zip(args, is_path, strict=True) if is_path]
            non_paths = [arg for arg, is_path in zip(args, is_path, strict=True) if not is_path]
            if paths:
                cfg = OmegaConf.merge(cfg, *(get_config(path, task_path) for path in paths))
            cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(non_paths))

        return cast(
            Config,
            OmegaConf.to_container(
                cfg,
                resolve=True,
                throw_on_missing=True,
                structured_config_mode=SCMode.INSTANTIATE,
            ),
        )

    @classmethod
    def config_str(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> str:
        return OmegaConf.to_yaml(cls.get_config(*cfgs, use_cli=use_cli), sort_keys=True)

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
