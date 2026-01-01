"""Tests for FP8 quantization utilities."""

import jax
import jax.numpy as jnp
import pytest

from xax.nn.fp8 import (
    Fp8Linear,
    compute_amax_history,
    compute_scale,
    fp8_matmul,
    get_fp8_max,
    init_fp8_scales,
    quantize,
    quantize_dequantize,
)


@pytest.mark.parametrize("fp8_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
def test_get_fp8_max(fp8_dtype: jnp.dtype) -> None:
    fp8_max = get_fp8_max(fp8_dtype)
    assert jnp.isfinite(fp8_max)
    assert fp8_max > 0
    expected = float(jnp.finfo(fp8_dtype).max)
    assert jnp.isclose(fp8_max, expected)


def test_compute_scale() -> None:
    fp8_max = jnp.array(448.0)
    prev_scale = jnp.array(2.5)

    # Basic scaling: scale = fp8_max / amax
    scale = compute_scale(jnp.array(100.0), prev_scale, fp8_max)
    assert jnp.isclose(scale, fp8_max / 100.0)

    # Invalid amax values fall back to previous scale
    assert jnp.isclose(compute_scale(jnp.array(0.0), prev_scale, fp8_max), prev_scale)
    assert jnp.isclose(compute_scale(jnp.array(-1.0), prev_scale, fp8_max), prev_scale)
    assert jnp.isclose(compute_scale(jnp.array(jnp.inf), prev_scale, fp8_max), prev_scale)

    # Margin reduces scale by power of 2
    scale_m0 = compute_scale(jnp.array(100.0), prev_scale, fp8_max, margin=0)
    scale_m1 = compute_scale(jnp.array(100.0), prev_scale, fp8_max, margin=1)
    assert jnp.isclose(scale_m1, scale_m0 / 2)


def test_amax_history() -> None:
    x = jnp.array([1.0, -5.0, 3.0])
    history = jnp.zeros(4)
    new_history = compute_amax_history(x, history)
    # New amax (5.0) should be at position 0 after roll
    assert jnp.isclose(new_history[0], 5.0)
    assert len(new_history) == 4


@pytest.mark.parametrize("fp8_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
def test_quantize_dequantize(fp8_dtype: jnp.dtype) -> None:
    key = jax.random.key(42)
    x = jax.random.normal(key, (16, 16))
    fp8_max = get_fp8_max(fp8_dtype)
    scale = fp8_max / jnp.max(jnp.abs(x))

    # Roundtrip preserves values approximately
    result = quantize_dequantize(x, fp8_dtype, scale)
    assert result.shape == x.shape
    rel_error = jnp.abs(result - x) / (jnp.abs(x) + 1e-6)
    assert jnp.mean(rel_error) < 0.05

    # Signs and relative ordering preserved
    x_ordered = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    scale_ordered = fp8_max / jnp.max(jnp.abs(x_ordered))
    result_ordered = quantize_dequantize(x_ordered, fp8_dtype, scale_ordered)
    assert jnp.all(jnp.sign(result_ordered) == jnp.sign(x_ordered))

    # Values clipped to FP8 range
    x_large = jnp.array([1000.0, -1000.0])
    q = quantize(x_large, fp8_dtype, jnp.array(1.0))
    assert jnp.all(jnp.abs(q.astype(jnp.float32)) <= fp8_max)


def test_fp8_scales_init() -> None:
    scales = init_fp8_scales(history_length=16)
    assert jnp.isclose(scales.input_scale, 1.0)
    assert jnp.isclose(scales.kernel_scale, 1.0)
    assert scales.input_amax_history.shape == (16,)
    assert scales.kernel_amax_history.shape == (16,)


def test_fp8_linear() -> None:
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)

    # Basic output shape
    layer = Fp8Linear(32, 64, key=key1)
    x = jax.random.normal(key2, (8, 32))
    out, _ = jax.vmap(layer)(x)
    assert out.shape == (8, 64)

    # FP8 disabled matches standard matmul
    layer_std = Fp8Linear(32, 64, key=key1, use_fp8=False)
    out_std, scales_std = jax.vmap(layer_std)(x)
    assert layer_std.bias is not None
    expected = x @ layer_std.weight.T + layer_std.bias
    assert jnp.allclose(out_std, expected)
    assert scales_std is None

    # FP8 enabled is close to standard
    layer_fp8 = Fp8Linear(32, 64, key=key1, use_fp8=True)
    out_fp8, _ = jax.vmap(layer_fp8)(x)
    assert jnp.allclose(out_fp8, out_std.astype(out_fp8.dtype), atol=1.0, rtol=0.1)

    # Delayed scaling mode updates history
    scales = init_fp8_scales(history_length=16)
    x_single = jax.random.normal(key2, (32,))
    out, new_scales = layer_fp8(x_single, scales=scales)
    assert out.shape == (64,)
    assert new_scales is not None
    assert jnp.any(new_scales.input_amax_history != 0.0)


@pytest.mark.parametrize("fp8_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
def test_fp8_matmul(fp8_dtype: jnp.dtype) -> None:
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    # Use smaller matrices to limit error accumulation
    a = jax.random.normal(key1, (4, 16))
    b = jax.random.normal(key2, (16, 8))

    result = fp8_matmul(a, b, fp8_dtype=fp8_dtype)
    std_result = a @ b

    assert result.shape == (4, 8)
    # E5M2 has larger errors due to fewer mantissa bits
    atol = 1.0 if fp8_dtype == jnp.float8_e5m2 else 0.5
    assert jnp.allclose(result, std_result.astype(result.dtype), atol=atol, rtol=0.1)
