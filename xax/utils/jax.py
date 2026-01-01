"""Defines some utility functions for interfacing with Jax."""

import functools
import logging
import os
from functools import wraps
from typing import Any, Callable, Hashable, Iterable, ParamSpec, Sequence, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import sharding_impls
from jax._src.lib import xla_client as xc
from jaxtyping import PyTree

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PyTree)

P = ParamSpec("P")  # For function parameters
R = TypeVar("R")  # For function return type

# For control flow functions.
Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")

F = TypeVar("F", bound=Callable)
AxisName = Hashable


class RecompileError(RuntimeError):
    pass


@functools.lru_cache(maxsize=None)
def disable_jit_level() -> int:
    """Gets a debugging flag for disabling jitting.

    For Xax's JIT'ed functions, we can set a JIT level which can be used to
    disable jitting when we want to debug some NaN issues.

    Returns:
        The JIT level to disable.
    """
    return int(os.environ.get("DISABLE_JIT_LEVEL", "0"))


def should_disable_jit(jit_level: int | None) -> bool:
    return jit_level is not None and jit_level < disable_jit_level()


def as_float(value: int | float | np.ndarray | jnp.ndarray) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return float(value.item())
    raise TypeError(f"Unexpected type: {type(value)}")


def get_hash(obj: object) -> int:
    """Get a hash of an object.

    If the object is hashable, use the hash. Otherwise, use the id.
    """
    if hasattr(obj, "__hash__"):
        return hash(obj)
    return id(obj)


def jit(
    in_shardings: Any = sharding_impls.UNSPECIFIED,  # noqa: ANN401
    out_shardings: Any = sharding_impls.UNSPECIFIED,  # noqa: ANN401
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    compiler_options: dict[str, Any] | None = None,
    jit_level: int | None = None,
    error_on_recompile: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrapper function that provides utility improvements over Jax's JIT.

    Specifically, this function works on class methods, is toggleable, and
    detects recompilations by matching hash values.

    This is meant to be used as a decorator factory, and the decorated function
    calls `wrapped`.
    """
    if should_disable_jit(jit_level):
        return lambda fn: fn  # Identity function.

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        jitted_fn = jax.jit(
            fn,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            donate_argnames=donate_argnames,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            compiler_options=compiler_options,
        )

        cache: dict[tuple[Any, tuple[str, ...]], int] = {}

        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if not error_on_recompile:
                return jitted_fn(*args, **kwargs)
            lowered = jitted_fn.lower(*args, **kwargs)
            in_avals = lowered.in_avals
            in_tree = lowered.in_tree
            key = (in_tree, tuple(str(a) for a in in_avals))
            compiled = lowered.compile()
            cid = id(compiled)
            if key in cache and cache[key] != cid:
                raise RecompileError(f"Recompilation detected for {key!r}. Old compiled id={cache[key]}, new id={cid}")
            return cast(R, jitted_fn(*args, **kwargs))

        return wrapped

    return decorator


def _split_module(tree: T, axis: int = 0) -> list[T]:
    """Splits a module in the same way that jax.lax.scan and jax.vmap do.

    Args:
        tree: The tree to split.
        axis: The axis to split on.

    Returns:
        A list of the split trees.
    """
    first_leaf = jax.tree.leaves(tree)[0]
    num_slices = first_leaf.shape[axis]
    result = [jax.tree.map(lambda x, idx=i: jnp.take(x, idx, axis=axis), tree) for i in range(num_slices)]
    return result


def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    jit_level: int | None = None,
) -> tuple[Carry, Y]:
    """A wrapper around jax.lax.scan that allows for more flexible tracing.

    If the provided JIT level is below the environment JIT level, we manually
    unroll the scan function as a for loop.

    Args:
        f: The function to scan.
        init: The initial value for the scan.
        xs: The input to the scan.
        length: The length of the scan.
        reverse: Whether to reverse the scan.
        unroll: The unroll factor for the scan.
        jit_level: The JIT level to use for the scan.

    Returns:
        A tuple containing the final carry and the output of the scan.
    """
    if not should_disable_jit(jit_level):
        return jax.lax.scan(f, init, xs, length, reverse, unroll)

    carry = init
    ys = []

    if xs is None:
        if length is None:
            raise ValueError("length must be provided if xs is None")
        for _ in range(length) if not reverse else range(length - 1, -1, -1):
            carry, y = f(carry, None)  # type: ignore[arg-type]
            ys.append(y)

    else:
        xlist = _split_module(xs, axis=0)
        if reverse:
            xlist = xlist[::-1]
        for x in xlist:
            carry, y = f(carry, x)
            ys.append(y)

    if reverse:
        ys = ys[::-1]

    if not ys:
        return carry, jnp.array([])  # type: ignore[return-value]

    return carry, jax.tree.map(lambda *ys: jnp.stack(ys), *ys)


def vmap(
    fun: Callable[P, R],
    in_axes: int | Sequence[int | None] = 0,
    jit_level: int | None = None,
) -> Callable[P, R]:
    """A wrapper around jax.lax.vmap that allows for more flexible tracing.

    If the provided JIT level is below the environment JIT level, we manually
    unroll the scan function as a for loop.
    """
    if not should_disable_jit(jit_level):
        return jax.vmap(fun, in_axes=in_axes)

    @functools.wraps(fun)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if kwargs:
            raise ValueError("vmap does not support keyword arguments")

        ia = in_axes
        if isinstance(ia, int):
            ia = [ia] * len(args)
        elif len(ia) != len(args):
            raise ValueError("in_axes must be the same length as args")

        if not all(isinstance(a, int) or a is None for a in ia):
            raise ValueError("in_axes must be a list of integers or None")

        ns = next((len(_split_module(a, axis=i)) for i, a in zip(ia, args, strict=True) if i is not None), None)
        if ns is None:
            return fun(*args, **kwargs)  # type: ignore[arg-type]
        split_args = [[a] * ns if i is None else _split_module(a, axis=i) for i, a in zip(ia, args, strict=True)]
        split_outputs = [fun(*sargs, **kwargs) for sargs in zip(*split_args, strict=True)]  # type: ignore[arg-type]

        if not split_outputs:
            return jnp.array([])  # type: ignore[return-value]

        return jax.tree.map(lambda *ys: jnp.stack(ys), *split_outputs)

    return wrapped


def grad(
    fun: Callable[P, R],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
    jit_level: int | None = None,
) -> Callable:
    """A wrapper around jax.grad that allows for more flexible tracing.

    We don't do anything special here, we just manually evaluate the function
    if the JIT level is below the environment JIT level.
    """
    if not should_disable_jit(jit_level):
        return jax.grad(fun, argnums, has_aux, holomorphic, allow_int, reduce_axes)

    @functools.wraps(fun)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Callable:
        # Evaluate the function once, then just return the gradient.
        fun(*args, **kwargs)

        return jax.grad(fun, argnums, has_aux, holomorphic, allow_int, reduce_axes)(*args, **kwargs)

    return wrapped
