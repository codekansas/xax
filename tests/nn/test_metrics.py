"""Tests metrics functions."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

import xax


def test_dtw_cost_matches() -> None:
    def _assert_correct(distances: Array) -> None:
        min_cost, indices = xax.dynamic_time_warping(distances, return_path=True)
        assert min_cost.item() == jnp.take_along_axis(distances, indices[None, :], axis=0).sum().item()

    # Simple distance matrix.
    series_1 = jnp.array([1, 2, 3, 4, 5])
    series_2 = jnp.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    _assert_correct((series_1[:, None] - series_2[None, :]) ** 2)

    with pytest.raises(AssertionError):
        _assert_correct((series_2[:, None] - series_1[None, :]) ** 2)

    # Random matrix.
    key = jax.random.key(0)
    n, m = 5, 10
    distances = jax.random.uniform(key, (n, m))
    _assert_correct(distances)

    # Edge case - single element
    _assert_correct(jnp.array([[1.0]]))

    # Edge case - equal length sequences
    n = 5
    distances = jax.random.uniform(key, (n, n))
    _assert_correct(distances)

    # Edge case - zero distances
    _assert_correct(jnp.zeros((3, 5)))
