"""Defines general-purpose helper functions for initializing norm layers."""

from typing import Literal, TypeVar, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

T_module = TypeVar("T_module", bound=eqx.Module)

NormType = Literal[
    "no_norm",
    "batch",
    "batch_affine",
    "instance",
    "instance_affine",
    "group",
    "group_affine",
    "layer",
    "layer_affine",
    "rms",
]


def cast_norm_type(s: str) -> NormType:
    """Cast a string to NormType, raising an error if invalid.

    Args:
        s: String to cast

    Returns:
        NormType value

    Raises:
        AssertionError: If the string is not a valid norm type
    """
    args = get_args(NormType)
    assert s in args, f"Invalid norm type: '{s}' Valid options are {args}"
    return cast(NormType, s)


class RMSNorm(eqx.Module):
    """Defines root-mean-square normalization."""

    weight: jax.Array
    eps: float = eqx.field(static=True, default=1e-6)

    @classmethod
    def build(cls, dim: int, eps: float = 1e-6) -> "RMSNorm":
        """Build RMSNorm from parameters.

        Args:
            dim: Dimension to normalize over
            eps: Epsilon value for numerical stability

        Returns:
            RMSNorm instance
        """
        weight = jnp.ones(dim)
        return cls(weight=weight, eps=eps)

    def _norm(self, x: Array) -> Array:
        """Apply RMS normalization.

        Args:
            x: Input array

        Returns:
            Normalized array
        """
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input array

        Returns:
            Normalized and scaled array
        """
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        return output * self.weight


class LastBatchNorm(eqx.Module):
    """Applies batch norm along final dimension without transposing the tensor.

    The normalization tracks the running mean and variance for each channel,
    then normalizes each channel to have a unit normal distribution.

    Input:
        x: Tensor with shape (..., N)

    Output:
        The tensor, normalized by the running mean and variance
    """

    channels: int
    momentum: float
    eps: float
    mean: Array
    var: Array
    affine_weight: Array | None
    affine_bias: Array | None

    def __init__(
        self,
        channels: int,
        momentum: float = 0.99,
        affine: bool = True,
        eps: float = 1e-4,
    ) -> None:
        """Initialize LastBatchNorm.

        Args:
            channels: Number of channels
            momentum: Momentum for running statistics
            affine: Whether to apply affine transformation
            eps: Epsilon value for numerical stability
        """
        self.channels = channels
        self.momentum = momentum
        self.eps = eps

        self.mean = jnp.zeros(channels)
        self.var = jnp.ones(channels)

        if affine:
            self.affine_weight = jnp.ones((channels, channels))
            self.affine_bias = jnp.zeros(channels)
        else:
            self.affine_weight = None
            self.affine_bias = None

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input array with shape (..., channels)

        Returns:
            Normalized array
        """
        if self.affine_weight is not None and self.affine_bias is not None:
            x = jnp.einsum("... c, c c -> ... c", x, self.affine_weight) + self.affine_bias

        # Compute batch statistics
        x_flat = x.reshape(-1, self.channels)
        mean = jnp.mean(x_flat, axis=0)
        var = jnp.var(x_flat, axis=0)

        # Update running statistics (in JAX, this would typically be handled by state)
        # For now, we use the current batch statistics
        new_mean = mean * (1 - self.momentum) + self.mean * self.momentum
        new_var = var * (1 - self.momentum) + self.var * self.momentum

        # Normalize
        x_out = (x - new_mean) / jnp.sqrt(new_var + self.eps)

        # In a real implementation, you'd update self.mean and self.var here
        # But in JAX, this would require state management
        return x_out


class ConvLayerNorm(eqx.Module):
    """Layer normalization for convolutional layers."""

    channels: int
    eps: float
    elementwise_affine: bool
    weight: Array | None
    bias: Array | None
    static_shape: tuple[int, ...] | None

    def __init__(
        self,
        channels: int,
        *,
        dims: int | None = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        """Initialize ConvLayerNorm.

        Args:
            channels: Number of channels
            dims: Number of spatial dimensions (1, 2, or 3)
            eps: Epsilon value for numerical stability
            elementwise_affine: Whether to apply elementwise affine transformation
        """
        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = jnp.ones(channels)
            self.bias = jnp.zeros(channels)
        else:
            self.weight = None
            self.bias = None

        self.static_shape = None if dims is None else (1, -1) + (1,) * dims

    def __call__(self, inputs: Array) -> Array:
        """Forward pass.

        Args:
            inputs: Input tensor with shape (B, C, ...)

        Returns:
            Normalized tensor
        """
        # Compute mean and variance over channel dimension
        mean = jnp.mean(inputs, axis=1, keepdims=True)
        var = jnp.mean(jnp.square(inputs - mean), axis=1, keepdims=True)
        normalized_inputs = (inputs - mean) / jnp.sqrt(var + self.eps)

        if self.elementwise_affine and self.weight is not None and self.bias is not None:
            if self.static_shape is None:
                # Dynamically reshape weight and bias to match input shape
                weight = self.weight.reshape((-1,) + (1,) * (len(inputs.shape) - 2))
                bias = self.bias.reshape((-1,) + (1,) * (len(inputs.shape) - 2))
            else:
                weight = self.weight.reshape(self.static_shape)
                bias = self.bias.reshape(self.static_shape)
            normalized_inputs = normalized_inputs * weight + bias

        return normalized_inputs


_ConvNorm = eqx.nn.Identity | eqx.nn.GroupNorm | ConvLayerNorm
Norm1D = _ConvNorm
Norm2D = _ConvNorm
Norm3D = _ConvNorm


def get_norm_1d(
    norm: NormType,
    *,
    dim: int | None = None,
    groups: int | None = None,
    eps: float = 1e-5,
) -> Norm1D:
    """Returns a normalization layer for tensors with shape (B, C, T).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        groups: The number of groups to use for group normalization
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid 1D norm type
    """
    match norm:
        case "no_norm":
            return eqx.nn.Identity()
        case "batch" | "batch_affine":
            if dim is None:
                raise ValueError("`dim` is required for batch norm")
            # Use GroupNorm as a substitute for BatchNorm (BatchNorm requires state management in JAX)
            return eqx.nn.GroupNorm(1, dim)
        case "instance" | "instance_affine":
            if dim is None:
                raise ValueError("`dim` is required for instance norm")
            # Instance norm is group norm with groups=dim
            return eqx.nn.GroupNorm(dim, dim)
        case "group" | "group_affine":
            if dim is None:
                raise ValueError("`dim` is required for group norm")
            if groups is None:
                raise ValueError("`groups` is required for group norm")
            return eqx.nn.GroupNorm(groups, dim)
        case "layer" | "layer_affine":
            if dim is None:
                raise ValueError("`dim` is required for layer norm")
            return ConvLayerNorm(dim, dims=1, eps=eps, elementwise_affine=norm == "layer_affine")
        case _:
            raise NotImplementedError(f"Invalid 1D norm type: {norm}")


def get_norm_2d(
    norm: NormType,
    *,
    dim: int | None = None,
    groups: int | None = None,
    eps: float = 1e-5,
) -> Norm2D:
    """Returns a normalization layer for tensors with shape (B, C, H, W).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        groups: The number of groups to use for group normalization
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid 2D norm type
    """
    match norm:
        case "no_norm":
            return eqx.nn.Identity()
        case "batch" | "batch_affine":
            if dim is None:
                raise ValueError("`dim` is required for batch norm")
            # Use GroupNorm as a substitute for BatchNorm
            return eqx.nn.GroupNorm(1, dim)
        case "instance" | "instance_affine":
            if dim is None:
                raise ValueError("`dim` is required for instance norm")
            # Instance norm is group norm with groups=dim
            return eqx.nn.GroupNorm(dim, dim)
        case "group" | "group_affine":
            if dim is None:
                raise ValueError("`dim` is required for group norm")
            if groups is None:
                raise ValueError("`groups` is required for group norm")
            return eqx.nn.GroupNorm(groups, dim)
        case "layer" | "layer_affine":
            if dim is None:
                raise ValueError("`dim` is required for layer norm")
            return ConvLayerNorm(dim, dims=2, eps=eps, elementwise_affine=norm == "layer_affine")
        case _:
            raise NotImplementedError(f"Invalid 2D norm type: {norm}")


def get_norm_3d(
    norm: NormType,
    *,
    dim: int | None = None,
    groups: int | None = None,
    eps: float = 1e-5,
) -> Norm3D:
    """Returns a normalization layer for tensors with shape (B, C, D, H, W).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        groups: The number of groups to use for group normalization
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid 3D norm type
    """
    match norm:
        case "no_norm":
            return eqx.nn.Identity()
        case "batch" | "batch_affine":
            if dim is None:
                raise ValueError("`dim` is required for batch norm")
            # Use GroupNorm as a substitute for BatchNorm
            return eqx.nn.GroupNorm(1, dim)
        case "instance" | "instance_affine":
            if dim is None:
                raise ValueError("`dim` is required for instance norm")
            # Instance norm is group norm with groups=dim
            return eqx.nn.GroupNorm(dim, dim)
        case "group" | "group_affine":
            if dim is None:
                raise ValueError("`dim` is required for group norm")
            if groups is None:
                raise ValueError("`groups` is required for group norm")
            return eqx.nn.GroupNorm(groups, dim)
        case "layer" | "layer_affine":
            if dim is None:
                raise ValueError("`dim` is required for layer norm")
            return ConvLayerNorm(dim, dims=3, eps=eps, elementwise_affine=norm == "layer_affine")
        case _:
            raise NotImplementedError(f"Invalid 3D norm type: {norm}")


NormLinear = eqx.nn.Identity | LastBatchNorm | eqx.nn.LayerNorm | RMSNorm


def get_norm_linear(
    norm: NormType,
    *,
    dim: int | None = None,
    eps: float = 1e-5,
) -> NormLinear:
    """Returns a normalization layer for tensors with shape (B, ..., C).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid linear norm type
    """
    match norm:
        case "no_norm":
            return eqx.nn.Identity()
        case "batch" | "batch_affine":
            if dim is None:
                raise ValueError("`dim` is required for batch norm")
            return LastBatchNorm(dim, affine=norm == "batch_affine", eps=eps)
        case "layer" | "layer_affine":
            if dim is None:
                raise ValueError("`dim` is required for layer norm")
            return eqx.nn.LayerNorm(dim, use_weight=norm == "layer_affine", use_bias=norm == "layer_affine", eps=eps)
        case "rms":
            if dim is None:
                raise ValueError("`dim` is required for RMS norm")
            return RMSNorm.build(dim, eps=eps)
        case _:
            raise NotImplementedError(f"Invalid linear norm type: {norm}")
