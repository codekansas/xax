"""Norm and metric utilities."""

from typing import Literal, cast, get_args, overload

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array

NormType = Literal["l1", "l2"]


def cast_euclidean_norm_type(norm: str) -> NormType:
    if norm not in get_args(NormType):
        raise ValueError(f"Invalid norm: {norm}")
    return cast(NormType, norm)


def get_euclidean_norm(x: Array, norm: NormType) -> Array:
    match norm:
        case "l1":
            return jnp.abs(x)
        case "l2":
            return jnp.square(x)
        case _:
            raise ValueError(f"Invalid norm: {norm}")


@overload
def dynamic_time_warping(distance_matrix_nm: Array) -> Array: ...


@overload
def dynamic_time_warping(distance_matrix_nm: Array, return_path: Literal[True]) -> tuple[Array, Array]: ...


def dynamic_time_warping(distance_matrix_nm: Array, return_path: bool = False) -> Array | tuple[Array, Array]:
    """Dynamic Time Warping.

    Args:
        distance_matrix_nm: A matrix of pairwise distances between two
            sequences, with shape (N, M), with the condition that N <= M.
        return_path: If set, return the minimum path, otherwise just return
            the cost. The latter is preferred if using this function as a
            distance metric since it avoids the backwards scan on backpointers.

    Returns:
        The cost of the minimum path from the top-left corner of the distance
        matrix to the bottom-right corner, along with the indices of that
        minimum path.
    """
    chex.assert_shape(distance_matrix_nm, (None, None))
    n, m = distance_matrix_nm.shape

    assert n <= m, f"Invalid dynamic time warping distance matrix shape: ({n}, {m})"

    # Masks values which cannot be reached.
    row_idx = jnp.arange(n)[:, None]
    col_idx = jnp.arange(m)[None, :]
    mask = row_idx > col_idx
    distance_matrix_nm = jnp.where(mask, jnp.inf, distance_matrix_nm)

    # Pre-pads with inf
    distance_matrix_nm = jnp.pad(distance_matrix_nm, ((1, 0), (0, 0)), mode="constant", constant_values=jnp.inf)
    indices = jnp.arange(n)

    # Scan over remaining rows to fill cost matrix
    def scan_fn(prev_cost: Array, cur_distances: Array) -> tuple[Array, Array]:
        same_trans = prev_cost
        prev_trans = jnp.pad(prev_cost[:-1], ((1, 0),), mode="constant", constant_values=jnp.inf)
        nc = jnp.minimum(prev_trans, same_trans) + cur_distances[1:]
        return nc, jnp.where(prev_trans < same_trans, indices - 1, indices) if return_path else nc

    init_cost = distance_matrix_nm[1:, 0]
    final_cost, back_pointers = jax.lax.scan(scan_fn, init_cost, distance_matrix_nm[:, 1:].T)

    if not return_path:
        return final_cost

    # Scan the back pointers backwards to get the minimum path.
    def scan_back_fn(carry: Array, back_pointer: Array) -> tuple[Array, Array]:
        prev_idx = back_pointer[carry]
        return prev_idx, carry

    final_index = jnp.array(n - 1)
    _, min_path = jax.lax.scan(scan_back_fn, final_index, back_pointers, reverse=True)
    min_path = jnp.pad(min_path, ((1, 0)), mode="constant", constant_values=0)

    return final_cost[-1], min_path
