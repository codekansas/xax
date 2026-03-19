"""Tests for literal-typed config options."""

import jax.numpy as jnp
import pytest

from xax.task.loggers.wandb import WandbConfig
from xax.task.mixins.logger import LoggerConfig
from xax.utils.structured_config import merge_config_sources
from xax.utils.types.training import PrecisionConfig


def test_wandb_config_uses_literal_options() -> None:
    cfg = merge_config_sources(
        WandbConfig,
        [{"reinit": "finish_previous", "mode": "offline", "resume": "allow"}],
    )
    assert cfg.reinit == "finish_previous"
    assert cfg.mode == "offline"
    assert cfg.resume == "allow"

    with pytest.raises(TypeError, match="literal options"):
        merge_config_sources(WandbConfig, [{"reinit": "restart"}])


def test_logger_config_uses_literal_backends() -> None:
    cfg = merge_config_sources(LoggerConfig, [{"logger_backend": ["stdout", "wandb"]}])
    assert cfg.logger_backend == ["stdout", "wandb"]

    with pytest.raises(TypeError, match="literal options"):
        merge_config_sources(LoggerConfig, [{"logger_backend": ["stdout", "bad-backend"]}])


def test_precision_config_uses_literal_values() -> None:
    cfg = merge_config_sources(PrecisionConfig, [{"compute_dtype": "float16", "grad_dtype": "float32"}])
    assert cfg.compute_dtype == "float16"
    assert cfg.grad_dtype == "float32"
    assert cfg.compute_jax_dtype == jnp.float16
    assert cfg.grad_jax_dtype == jnp.float32

    with pytest.raises(TypeError, match="literal options"):
        merge_config_sources(PrecisionConfig, [{"compute_dtype": "fp64"}])
