"""Defines some utility functions for interfacing with Jax."""

import functools
import logging
import os
from functools import wraps
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    ParamSpec,
    Sequence,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    overload,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import sharding_impls
from jax._src.lib import xla_client as xc
from jax.sharding import NamedSharding
from jaxtyping import Array, PyTree

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


def fix_unspecified_sharding(arr: Array, sharding: NamedSharding) -> Array:
    if hasattr(arr, "sharding") and not hasattr(arr.sharding, "is_fully_replicated"):
        # Array has unspecified sharding - get data from first shard
        if hasattr(arr, "addressable_shards") and arr.addressable_shards:
            data = np.asarray(arr.addressable_shards[0].data)
            return jax.device_put(data, sharding)
    return arr


def to_numpy(arr: jnp.ndarray | np.ndarray) -> np.ndarray:
    """Convert a JAX or numpy array to a numpy array.

    Handles arrays with unspecified sharding from multi-GPU JIT by extracting
    data from the first addressable shard.

    Args:
        arr: A JAX or numpy array.

    Returns:
        A numpy array with the same data.
    """
    if isinstance(arr, np.ndarray):
        return arr
    try:
        return np.asarray(arr)
    except AttributeError:
        # Array has unspecified sharding from multi-GPU JIT
        if hasattr(arr, "addressable_shards") and arr.addressable_shards:
            return np.asarray(arr.addressable_shards[0].data)
        raise


def to_scalar(arr: jnp.ndarray | np.ndarray) -> int | float:
    """Convert a scalar JAX or numpy array to a Python scalar.

    Handles arrays with unspecified sharding from multi-GPU JIT.

    Args:
        arr: A scalar JAX or numpy array.

    Returns:
        A Python int or float.
    """
    return to_numpy(arr).item()


def as_float(value: int | float | np.ndarray | jnp.ndarray) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return float(to_scalar(value))
    raise TypeError(f"Unexpected type: {type(value)}")


def get_hash(obj: object) -> int:
    """Get a hash of an object.

    If the object is hashable, use the hash. Otherwise, use the id.
    """
    if hasattr(obj, "__hash__"):
        return hash(obj)
    return id(obj)


@overload
def jit(
    fn: Callable[P, R],
    *,
    in_shardings: Any = ...,  # noqa: ANN401
    out_shardings: Any = ...,  # noqa: ANN401
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
) -> Callable[P, R]: ...


@overload
def jit(
    fn: None = None,
    *,
    in_shardings: Any = ...,  # noqa: ANN401
    out_shardings: Any = ...,  # noqa: ANN401
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
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def jit(
    fn: Callable[P, R] | None = None,
    *,
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
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrapper function that provides utility improvements over Jax's JIT.

    Specifically, this function works on class methods, is toggleable, and
    detects recompilations by matching hash values.

    Can be used as a decorator factory or directly on a function:
        @jit()
        def foo(): ...

        @jit
        def bar(): ...

        jit(baz)
    """

    def make_jitted(f: Callable[P, R]) -> Callable[P, R]:
        jitted_fn = jax.jit(
            f,
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

        @wraps(f)
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

    if should_disable_jit(jit_level):
        if fn is not None:
            return fn
        return lambda f: f

    if fn is not None:
        return make_jitted(fn)

    return make_jitted


_DonateArg = Literal["all", "all-except-first", "warn", "warn-except-first", "none"]


class _JitKwargs(TypedDict, total=False):
    """Keyword arguments passed through to jax.jit."""

    in_shardings: Any  # noqa: ANN401
    out_shardings: Any  # noqa: ANN401
    static_argnums: int | Sequence[int] | None
    static_argnames: str | Iterable[str] | None
    donate_argnums: int | Sequence[int] | None
    donate_argnames: str | Iterable[str] | None
    keep_unused: bool
    device: xc.Device | None
    backend: str | None
    inline: bool
    abstracted_axes: Any  # noqa: ANN401


@overload
def filter_jit(
    fn: Callable[P, R],
    *,
    donate: _DonateArg = "none",
    jit_level: int | None = None,
    **jitkwargs: Unpack[_JitKwargs],
) -> Callable[P, R]: ...


@overload
def filter_jit(
    fn: None = None,
    *,
    donate: _DonateArg = "none",
    jit_level: int | None = None,
    **jitkwargs: Unpack[_JitKwargs],
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def filter_jit(
    fn: Callable[P, R] | None = None,
    *,
    donate: _DonateArg = "none",
    jit_level: int | None = None,
    **jitkwargs: Unpack[_JitKwargs],
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrapper around eqx.filter_jit with xax's toggleable JIT feature.

    This function combines equinox's filter_jit (which handles pytrees with
    non-array leaves by treating them as static) with xax's JIT level control.

    Args:
        fn: The function to JIT compile. If None, returns a decorator.
        donate: Buffer donation strategy passed to eqx.filter_jit.
            Options: "none", "all", "all-except-first", "warn", "warn-except-first"
        jit_level: Optional JIT level for conditional compilation.
            If set and >= DISABLE_JIT_LEVEL env var, JIT is disabled.
        **jitkwargs: Additional arguments passed to jax.jit (e.g., in_shardings, out_shardings).

    Returns:
        JIT-compiled function or decorator.
    """
    if should_disable_jit(jit_level):
        if fn is not None:
            return fn
        return lambda f: f

    if fn is not None:
        return eqx.filter_jit(fn, donate=donate, **jitkwargs)

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        return eqx.filter_jit(f, donate=donate, **jitkwargs)

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
    spmd_axis_name: AxisName | None = None,
) -> Callable[P, R]:
    """A wrapper around jax.lax.vmap that allows for more flexible tracing.

    If the provided JIT level is below the environment JIT level, we manually
    unroll the scan function as a for loop.

    Args:
        fun: The function to vectorize.
        in_axes: Specifies which input array axes to map over.
        jit_level: If set and below environment JIT level, manually unrolls.
        spmd_axis_name: If set, enables SPMD vmap for mesh-sharded inputs.
            This should match the mesh axis name used for sharding the
            batch dimension (e.g., 'batch').
    """
    if not should_disable_jit(jit_level):
        return jax.vmap(fun, in_axes=in_axes, spmd_axis_name=spmd_axis_name)

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
