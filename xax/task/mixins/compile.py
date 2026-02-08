"""Defines a mixin for handling JAX compilation behavior.

This mixin allows control over JAX compilation settings like jit, pmap, and vmap
behavior during initialization and training.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import jax

from xax.task.base import BaseConfig, BaseTask
from xax.utils.structured_config import field

logger = logging.getLogger(__name__)


def get_cache_dir() -> str | None:
    # By default, only cache on MacOS, since Jax caching on Linux is very
    # prone to NaNs.
    match sys.platform:
        case "darwin" | "linux":
            return str((Path.home() / ".cache" / "jax" / "jaxcache").resolve())
        case _:
            return None


@jax.tree_util.register_dataclass
@dataclass
class CompileOptions:
    # JAX compilation options
    debug_nans: bool = field(
        value=False,
        help="If True, breaks on NaNs",
    )
    disable_jit: bool = field(
        value=False,
        help="If True, disables JIT compilation",
    )
    enable_x64: bool = field(
        value=False,
        help="If True, enables 64-bit precision",
    )
    default_device: str | None = field(
        value=None,
        help="Default device to use (e.g. 'cpu', 'gpu')",
    )

    # JAX logging options
    logging_level: str = field(
        value="WARNING",
        help="JAX logging verbosity level",
    )

    # JAX cache options
    cache_dir: str | None = field(
        # Only cache by default on MacOS systems.
        value=get_cache_dir,
        help="Directory for JAX compilation cache. If None, caching is disabled",
    )
    cache_min_size_bytes: int = field(
        value=-1,
        help="Minimum size in bytes for cache entries. -1 means no minimum",
    )
    cache_min_compile_time_secs: float = field(
        value=0.0,
        help="Minimum compilation time in seconds for cache entries. 0 means no minimum",
    )
    cache_enable_xla: str = field(
        value="none",
        help="Which XLA caches to enable",
    )


@jax.tree_util.register_dataclass
@dataclass
class CompileConfig(BaseConfig):
    compile: CompileOptions = field(CompileOptions(), help="Compilation configuration")


Config = TypeVar("Config", bound=CompileConfig)


class CompileMixin(BaseTask[Config], Generic[Config]):
    """Defines a task mixin for controlling JAX compilation behavior."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        cc = self.config.compile

        # Set basic compilation flags
        if cc.debug_nans:
            logger.info("Enabling NaNs debugging")
            jax.config.update("jax_debug_nans", True)

        if cc.disable_jit:
            logger.info("Disabling JIT compilation")
            jax.config.update("jax_disable_jit", True)

        if cc.enable_x64:
            logger.info("Enabling 64-bit precision")
            jax.config.update("jax_enable_x64", True)

        if cc.default_device is not None:
            logger.info("Setting default device to %s", cc.default_device)
            jax.config.update("jax_default_device", cc.default_device)

        # Set logging level
        logger.info("Setting JAX logging level to %s", cc.logging_level)
        jax.config.update("jax_logging_level", cc.logging_level)

        # Configure compilation cache
        if cc.cache_dir is not None:
            logger.info("Setting JAX compilation cache directory to %s", cc.cache_dir)
            jax.config.update("jax_compilation_cache_dir", cc.cache_dir)

            logger.info("Configuring JAX compilation cache parameters")
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", cc.cache_min_size_bytes)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", cc.cache_min_compile_time_secs)
            jax.config.update("jax_persistent_cache_enable_xla_caches", cc.cache_enable_xla)
