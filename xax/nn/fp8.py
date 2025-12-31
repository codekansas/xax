"""FP8 quantization utilities for efficient matrix multiplication.

This module implements FP8 (8-bit floating point) quantization for matrix
multiplications. XLA detects the quantize-dequantize pattern around dot
products and optimizes them to use hardware FP8 matmul (e.g., cublasLt).

The key insight is that we don't call FP8 matmul directly - instead we
set up a pattern that XLA recognizes and optimizes:
1. Quantize inputs to FP8 with scaling
2. "Fake" dequantize back to compute dtype
3. Standard dot product - XLA rewrites to FP8 matmul

Two scaling approaches are supported:
- Current scaling: Compute scale from max(abs(x)) each forward pass
- Delayed scaling: Use historical amax values for better performance

References:
- Flax FP8 guide: https://flax-linen.readthedocs.io/en/stable/guides/quantization/fp8_basics.html
- Flax fp8_ops.py: https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

# FP8 format constants
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max  # ~448
E5M2_MAX = jnp.finfo(jnp.float8_e5m2).max  # ~57344

# Default amax history length for delayed scaling
AMAX_HISTORY_LENGTH = 1024


def get_fp8_max(fp8_dtype: jnp.dtype, compute_dtype: jnp.dtype = jnp.float32) -> Array:
    """Get the maximum representable value for an FP8 dtype.

    Args:
        fp8_dtype: The FP8 dtype (e.g., jnp.float8_e4m3fn or jnp.float8_e5m2)
        compute_dtype: The dtype to cast the result to

    Returns:
        The maximum representable value as a scalar
    """
    return jnp.finfo(fp8_dtype).max.astype(compute_dtype)


def compute_scale(
    amax: Array,
    scale: Array,
    fp8_max: Array,
    margin: int = 0,
) -> Array:
    """Compute scaling factor from absolute maximum value.

    This implements the NVIDIA transformer-engine approach for computing
    scaling factors. The scale is computed as:
        scale = fp8_max / (amax * 2^margin)

    Args:
        amax: The absolute maximum value of the tensor
        scale: The previous scale (used if amax is invalid)
        fp8_max: Maximum representable value of the target FP8 dtype
        margin: Safety margin as power of 2 (default 0)

    Returns:
        The computed scale factor
    """
    # Compute new scale: fp8_max / (amax * 2^margin)
    # We store the inverse scale for efficiency
    sf = (fp8_max / amax) / (2**margin)

    # Use previous scale if amax is invalid (<=0 or non-finite)
    sf = jnp.where(amax > 0, sf, scale)
    sf = jnp.where(jnp.isfinite(amax), sf, scale)

    return sf


def compute_amax_history(x: Array, amax_history: Array) -> Array:
    """Update the rolling amax history buffer.

    Args:
        x: Input tensor to compute amax from
        amax_history: Previous amax history buffer of shape (history_length,)

    Returns:
        Updated amax history with new value at position 0
    """
    amax_update = jnp.max(jnp.abs(x)).astype(amax_history.dtype)
    new_history = jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)
    return new_history


def quantize(
    x: Array,
    fp8_dtype: jnp.dtype,
    scale: Array,
    compute_dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Quantize a tensor to FP8.

    Args:
        x: Input tensor in compute_dtype
        fp8_dtype: Target FP8 dtype (e.g., jnp.float8_e4m3fn)
        scale: Scaling factor (x will be multiplied by scale before quantizing)
        compute_dtype: The compute dtype for intermediate calculations

    Returns:
        Quantized tensor in fp8_dtype
    """
    fp8_max = get_fp8_max(fp8_dtype, compute_dtype)

    # Scale and clip to FP8 range
    scaled_x = x * jnp.broadcast_to(scale.astype(compute_dtype), x.shape)
    clipped_x = jnp.clip(scaled_x, -fp8_max, fp8_max)

    return clipped_x.astype(fp8_dtype)


def dequantize(
    x: Array,
    compute_dtype: jnp.dtype,
    scale: Array,
) -> Array:
    """Dequantize an FP8 tensor back to compute dtype.

    Args:
        x: Input tensor in FP8 dtype
        compute_dtype: Target dtype for dequantization
        scale: Scaling factor (result will be divided by scale)

    Returns:
        Dequantized tensor in compute_dtype
    """
    return x.astype(compute_dtype) / jnp.broadcast_to(scale.astype(compute_dtype), x.shape)


def quantize_dequantize(
    x: Array,
    fp8_dtype: jnp.dtype,
    scale: Array,
    compute_dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Quantize and immediately dequantize (for XLA pattern detection).

    This is the key function for FP8 matmul. XLA detects this pattern
    around dot products and rewrites them to use hardware FP8 matmul.

    Args:
        x: Input tensor in compute_dtype
        fp8_dtype: FP8 dtype for quantization
        scale: Scaling factor
        compute_dtype: Compute dtype for dequantization

    Returns:
        Tensor that has been quantized to FP8 and back to compute_dtype
    """
    q = quantize(x, fp8_dtype, scale, compute_dtype)
    return dequantize(q, compute_dtype, scale)


@dataclass
class Fp8Scales:
    """Container for FP8 scaling state.

    This holds the scaling factors and amax history for delayed scaling.
    """

    input_scale: Array
    input_amax_history: Array
    kernel_scale: Array
    kernel_amax_history: Array


def init_fp8_scales(
    history_length: int = AMAX_HISTORY_LENGTH,
    compute_dtype: jnp.dtype = jnp.float32,
) -> Fp8Scales:
    """Initialize FP8 scaling state.

    Args:
        history_length: Length of amax history buffer
        compute_dtype: Dtype for scales

    Returns:
        Initialized Fp8Scales with unit scales and zero history
    """
    return Fp8Scales(
        input_scale=jnp.ones((), dtype=compute_dtype),
        input_amax_history=jnp.zeros((history_length,), dtype=compute_dtype),
        kernel_scale=jnp.ones((), dtype=compute_dtype),
        kernel_amax_history=jnp.zeros((history_length,), dtype=compute_dtype),
    )


class Fp8Linear(eqx.Module):
    """Linear layer with FP8 quantization for matrix multiplication.

    This wraps a standard linear layer with FP8 quantization-dequantization
    around the matmul. XLA detects this pattern and optimizes it to use
    hardware FP8 matmul kernels.

    The layer supports two modes:
    - Current scaling: Compute scale from input each forward pass
    - Delayed scaling: Use historical amax values (requires external state)

    For training, E4M3 is used for both activations and weights in the
    forward pass. For backward pass, E5M2 can be used for gradients due
    to its larger dynamic range.
    """

    weight: Array
    bias: Array | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    fp8_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        use_bias: bool = True,
        use_fp8: bool = True,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        fp8_dtype: jnp.dtype = jnp.float8_e4m3fn,
    ) -> None:
        """Initialize FP8 linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            key: PRNG key for initialization
            use_bias: Whether to include bias term
            use_fp8: Whether to use FP8 quantization
            compute_dtype: Dtype for computation (default bfloat16)
            fp8_dtype: FP8 dtype to use (default E4M3)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.compute_dtype = compute_dtype
        self.fp8_dtype = fp8_dtype

        # Initialize weights with Kaiming uniform
        key1, key2 = jax.random.split(key)
        bound = 1.0 / jnp.sqrt(in_features)
        self.weight = jax.random.uniform(
            key1,
            (out_features, in_features),
            minval=-bound,
            maxval=bound,
            dtype=compute_dtype,
        )

        if use_bias:
            self.bias = jax.random.uniform(
                key2,
                (out_features,),
                minval=-bound,
                maxval=bound,
                dtype=compute_dtype,
            )
        else:
            self.bias = None

    def __call__(
        self,
        x: Array,
        scales: Fp8Scales | None = None,
    ) -> tuple[Array, Fp8Scales | None]:
        """Forward pass with optional FP8 quantization.

        Args:
            x: Input tensor of shape (..., in_features)
            scales: Optional FP8 scales for delayed scaling mode.
                If None and use_fp8=True, uses current scaling.

        Returns:
            Tuple of (output tensor, updated scales or None)
        """
        if not self.use_fp8:
            # Standard linear without FP8
            out = x @ self.weight.T
            if self.bias is not None:
                out = out + self.bias
            return out, None

        # Ensure input is in compute dtype
        x = x.astype(self.compute_dtype)
        w = self.weight.astype(self.compute_dtype)

        if scales is not None:
            # Delayed scaling mode - use provided scales
            input_scale = scales.input_scale
            kernel_scale = scales.kernel_scale

            # Update amax history
            new_input_amax_history = compute_amax_history(x, scales.input_amax_history)
            new_kernel_amax_history = compute_amax_history(w, scales.kernel_amax_history)

            # Compute new scales from history for next iteration
            fp8_max = get_fp8_max(self.fp8_dtype, self.compute_dtype)
            new_input_scale = compute_scale(
                jnp.max(new_input_amax_history),
                input_scale,
                fp8_max,
            )
            new_kernel_scale = compute_scale(
                jnp.max(new_kernel_amax_history),
                kernel_scale,
                fp8_max,
            )

            updated_scales = Fp8Scales(
                input_scale=new_input_scale,
                input_amax_history=new_input_amax_history,
                kernel_scale=new_kernel_scale,
                kernel_amax_history=new_kernel_amax_history,
            )
        else:
            # Current scaling mode - compute scale from input
            fp8_max = get_fp8_max(self.fp8_dtype, self.compute_dtype)
            input_scale = fp8_max / jnp.maximum(jnp.max(jnp.abs(x)), 1e-12)
            kernel_scale = fp8_max / jnp.maximum(jnp.max(jnp.abs(w)), 1e-12)
            updated_scales = None

        # Quantize-dequantize for XLA pattern detection
        x_qdq = quantize_dequantize(x, self.fp8_dtype, input_scale, self.compute_dtype)
        w_qdq = quantize_dequantize(w, self.fp8_dtype, kernel_scale, self.compute_dtype)

        # Standard matmul - XLA will optimize to FP8 matmul
        out = x_qdq @ w_qdq.T

        if self.bias is not None:
            out = out + self.bias

        return out, updated_scales


def fp8_matmul(
    a: Array,
    b: Array,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    fp8_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> Array:
    """Perform FP8 matrix multiplication with current scaling.

    This is a convenience function for FP8 matmul with automatic scaling.
    XLA will detect the quantize-dequantize pattern and optimize to
    hardware FP8 matmul.

    Args:
        a: First input tensor
        b: Second input tensor
        compute_dtype: Dtype for computation
        fp8_dtype: FP8 dtype to use

    Returns:
        Result of a @ b computed with FP8 quantization
    """
    a = a.astype(compute_dtype)
    b = b.astype(compute_dtype)

    fp8_max = get_fp8_max(fp8_dtype, compute_dtype)
    a_scale = fp8_max / jnp.maximum(jnp.max(jnp.abs(a)), 1e-12)
    b_scale = fp8_max / jnp.maximum(jnp.max(jnp.abs(b)), 1e-12)

    a_qdq = quantize_dequantize(a, fp8_dtype, a_scale, compute_dtype)
    b_qdq = quantize_dequantize(b, fp8_dtype, b_scale, compute_dtype)

    return a_qdq @ b_qdq
