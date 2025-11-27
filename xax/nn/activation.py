"""Defines a general-purpose API for activation functions."""

import math
from collections.abc import Callable
from typing import Literal, cast, get_args

import jax
import jax.numpy as jnp
from jaxtyping import Array

ActivationType = Literal[
    "no_act",
    "relu",
    "relu6",
    "relu2",
    "clamp6",
    "leaky_relu",
    "elu",
    "celu",
    "selu",
    "gelu",
    "gelu_fast",
    "gelu_quick",
    "sigmoid",
    "log_sigmoid",
    "hard_sigomid",
    "tanh",
    "softsign",
    "softplus",
    "silu",
    "mish",
    "swish",
    "hard_swish",
    "soft_sign",
    "relu_squared",
    "laplace",
]


def cast_activation_type(s: str) -> ActivationType:
    """Cast a string to ActivationType, raising an error if invalid.

    Args:
        s: String to cast

    Returns:
        ActivationType value

    Raises:
        AssertionError: If the string is not a valid activation type
    """
    args = get_args(ActivationType)
    assert s in args, f"Invalid activation type: '{s}' Valid options are {args}"
    return cast(ActivationType, s)


def clamp(x: Array, value: float | None = None, value_range: tuple[float, float] | None = None) -> Array:
    assert (value is None) != (value_range is None), "Exactly one of `value` or `value_range` must be specified."
    if value is not None:
        value_range = (-value, value)
    else:
        assert value_range is not None
    return jnp.clip(x, value_range[0], value_range[1])


def clamp6(x: Array) -> Array:
    return jnp.clip(x, -6, 6)


def relu_squared(x: Array) -> Array:
    return jnp.square(jax.nn.relu(x))


def gelu_fast(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_quick(x: Array) -> Array:
    return x * jax.nn.sigmoid(1.702 * x)


def laplace(x: Array, mu: float = 0.707107, sigma: float = 0.282095) -> Array:
    return 0.5 * (1.0 + jax.scipy.special.erf((x - mu) / (sigma * math.sqrt(2.0))))


def get_activation(act: ActivationType) -> Callable[[Array], Array]:
    """Returns an activation function from a keyword string.

    Args:
        act: The keyword for the activation function

    Returns:
        The activation function (either a callable or an Equinox module)

    Raises:
        NotImplementedError: If the activation function is invalid
    """
    match act:
        case "no_act":
            return jax.nn.identity
        case "relu":
            return jax.nn.relu
        case "relu2":
            return jax.nn.relu
        case "relu6":
            return jax.nn.relu6
        case "clamp6":
            return clamp6
        case "leaky_relu":
            return jax.nn.leaky_relu
        case "elu":
            return jax.nn.elu
        case "celu":
            return jax.nn.celu
        case "selu":
            return jax.nn.selu
        case "gelu":
            return jax.nn.gelu
        case "gelu_fast":
            return gelu_fast
        case "gelu_quick":
            return gelu_quick
        case "sigmoid":
            return jax.nn.sigmoid
        case "log_sigmoid":
            return jax.nn.log_sigmoid
        case "hard_sigomid":
            return jax.nn.hard_sigmoid
        case "tanh":
            return jnp.tanh
        case "softsign":
            return jax.nn.soft_sign
        case "softplus":
            return jax.nn.softplus
        case "silu" | "swish":
            return jax.nn.silu
        case "mish":
            return jax.nn.mish
        case "hard_swish":
            return jax.nn.hard_swish
        case "soft_sign":
            return jax.nn.soft_sign
        case "relu_squared":
            return relu_squared
        case "laplace":
            return laplace
        case _:
            raise NotImplementedError(f"Activation function '{act}' is not implemented.")
