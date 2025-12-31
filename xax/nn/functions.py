"""Defines helper Jax functions."""

import random
from dataclasses import is_dataclass
from typing import Any, Callable, Iterable, Mapping, ParamSpec, Sequence, TypeVar

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from xax.core.conf import load_user_config

T = TypeVar("T")
P = ParamSpec("P")


def recursive_apply(item: Any, func: Callable[[Array], Array], include_numpy: bool = False) -> Any:  # noqa: ANN401
    """Applies a function recursively to tensors in an item.

    Args:
        item: The item to apply the function to
        func: The function to apply (for the tensor)
        include_numpy: If set, include numpy arrays

    Returns:
        The same item, with the function applied
    """
    if isinstance(item, (str, int, float)):
        return item
    if include_numpy and isinstance(item, np.ndarray):
        return func(jnp.array(item))
    if isinstance(item, Array):
        return func(item)
    if is_dataclass(item):
        return item.__class__(**{k: recursive_apply(v, func, include_numpy) for k, v in item.__dict__.items()})
    if isinstance(item, Mapping):
        return {k: recursive_apply(v, func, include_numpy) for k, v in item.items()}
    if isinstance(item, Sequence):
        return [recursive_apply(i, func, include_numpy) for i in item]
    return item


def recursive_chunk(item: Any, num_chunks: int, dim: int = 0) -> Iterable[Any]:  # noqa: ANN401
    """Recursively chunk tensors N times.

    Args:
        item: The item to recursively chunk
        num_chunks: The number of splits to make
        dim: The split dimension

    Yields:
        ``num_chunks`` chunks of items
    """
    if isinstance(item, (str, int, float)):
        yield from (item for _ in range(num_chunks))
    elif isinstance(item, np.ndarray):
        yield from np.array_split(item, num_chunks, axis=dim)
    elif is_dataclass(item):
        yield from (
            item.__class__(**{k: i for k, i in zip(item.__dict__, ii, strict=True)})
            for ii in zip(*(recursive_chunk(v, num_chunks, dim) for v in item.__dict__.values()), strict=False)
        )
    elif isinstance(item, Mapping):
        yield from (
            dict(zip(item, ii, strict=False))
            for ii in zip(*(recursive_chunk(i, num_chunks, dim) for i in item.values()), strict=False)
        )
    elif isinstance(item, Sequence):
        yield from (list(ii) for ii in zip(*(recursive_chunk(i, num_chunks, dim) for i in item), strict=False))
    else:
        yield from (item for _ in range(num_chunks))


def set_random_seed(seed: int | None = None, offset: int = 0) -> None:
    if seed is None:
        seed = load_user_config().experiment.default_random_seed
    seed += offset
    random.seed(seed)
    np.random.seed(seed)
