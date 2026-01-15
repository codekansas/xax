"""Attention mechanisms for transformer models.

This module implements standard attention mechanisms for transformers, supporting:
- Self-attention and cross-attention
- Grouped Query Attention (GQA) for efficient KV caching
- Rotary Position Embeddings (RoPE) with YaRN extension
- Sliding window attention
- FP8 quantization for matrix multiplications
- cuDNN flash attention acceleration

These are general-purpose building blocks that can be used for training
transformers from scratch or fine-tuning pre-trained models (with LoRA, etc).
"""

import math
import warnings
from typing import Literal, NotRequired, TypedDict

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from xax.nn.fp8 import Fp8Linear, Fp8Scales, init_fp8_scales
from xax.nn.lora import LoRALinear
from xax.utils.jax import scan as xax_scan


def apply_linear(x: Array, linear: eqx.nn.Linear | Fp8Linear | LoRALinear) -> Array:
    """Apply linear layer with LoRA support.

    This function applies a linear transformation using einsum for better performance.
    It supports standard eqx.nn.Linear, Fp8Linear, and LoRALinear layers, enabling LoRA
    fine-tuning without changing the calling code.

    Args:
        x: Input tensor of shape (..., in_features)
        linear: Linear layer (eqx.nn.Linear, Fp8Linear, or LoRALinear)

    Returns:
        Output tensor of shape (..., out_features)
    """
    if isinstance(linear, LoRALinear):
        # LoRALinear: use weight_oi and bias_o, plus LoRA delta
        y = jnp.einsum("...d,od->...o", x, linear.weight_oi)
        if linear.bias_o is not None:
            y = y + linear.bias_o
        # Add LoRA contribution: (x @ A) @ B * alpha
        delta = (x @ linear.lora_a_ir) @ linear.lora_b_ro * linear.alpha
        return y + delta
    elif isinstance(linear, Fp8Linear):
        # For Fp8Linear, call it directly (it handles FP8 internally)
        # This simplified path doesn't track scales - use forward() for that
        y = linear(x)
        if isinstance(y, tuple):
            return y[0]
        return y
    else:
        # Standard eqx.nn.Linear
        y = jnp.einsum("...d,od->...o", x, linear.weight)
        if linear.bias is not None:
            y = y + linear.bias
        return y


class RMSNorm(eqx.Module):
    """RMSNorm over the last dimension (no bias), matching LLaMA/Qwen style.

    RMS normalization is more efficient than LayerNorm and works well for LLMs.
    It normalizes by the root mean square without centering (no mean subtraction).
    """

    weight: Array
    eps: float = eqx.field(static=True, default=1e-6)

    @classmethod
    def build(cls, dim: int, eps: float = 1e-6) -> "RMSNorm":
        """Build RMSNorm from parameters.

        Args:
            dim: Dimension of the input
            eps: Small constant for numerical stability

        Returns:
            RMSNorm instance
        """
        weight = jnp.ones((dim,), dtype=jnp.float32)
        return cls(weight=weight, eps=eps)

    def __call__(self, x: Array) -> Array:
        norm = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / norm) * self.weight


class SwiGLU(eqx.Module):
    """SwiGLU feed-forward layer (LLaMA/Qwen style).

    Uses gated linear unit with SiLU activation:
        output = down(silu(gate(x)) * up(x))

    This is more expressive than standard FFN and commonly used in modern LLMs.
    """

    gate: eqx.nn.Linear
    up: eqx.nn.Linear
    down: eqx.nn.Linear

    @classmethod
    def build(
        cls,
        embed_dim: int,
        hidden_dim: int | None = None,
        *,
        key: PRNGKeyArray,
        use_bias: bool = False,
    ) -> "SwiGLU":
        """Build SwiGLU feed-forward layer from parameters.

        Args:
            embed_dim: Input/output embedding dimension
            hidden_dim: Hidden dimension (defaults to 4 * embed_dim)
            key: PRNG key for initialization
            use_bias: Whether to use bias in linear layers

        Returns:
            SwiGLU instance
        """
        if hidden_dim is None:
            hidden_dim = embed_dim * 4

        k1, k2, k3 = jax.random.split(key, 3)
        gate = eqx.nn.Linear(embed_dim, hidden_dim, use_bias=use_bias, key=k1)
        up = eqx.nn.Linear(embed_dim, hidden_dim, use_bias=use_bias, key=k2)
        down = eqx.nn.Linear(hidden_dim, embed_dim, use_bias=use_bias, key=k3)
        return cls(gate=gate, up=up, down=down)

    def __call__(self, x: Array) -> Array:
        chex.assert_rank(x, {2, 3})
        # Use apply_linear for LoRA support
        hidden = jax.nn.silu(apply_linear(x, self.gate)) * apply_linear(x, self.up)
        return apply_linear(hidden, self.down)


def can_use_cudnn_attention(
    dtype: jnp.dtype,
    head_dim: int,
    seq_len: int,
    has_bias: bool = False,
) -> bool:
    """Check if cuDNN flash attention can be used.

    cuDNN flash attention requirements:
    - dtype: fp16, bf16, or fp8
    - head_dim: <= 128 and multiple of 8
    - sequence length: multiple of 64
    - no custom attention bias (masks are OK)

    Args:
        dtype: Data type of Q/K/V tensors
        head_dim: Dimension per attention head
        seq_len: Sequence length
        has_bias: Whether custom attention bias is used

    Returns:
        True if cuDNN can be used, False otherwise
    """
    return (
        dtype in (jnp.float16, jnp.bfloat16)
        and head_dim <= 128
        and head_dim % 8 == 0
        and seq_len % 64 == 0
        and not has_bias
    )


class AttentionCache(TypedDict):
    k: Array
    v: Array
    position: Array


class Fp8ScalesCache(TypedDict):
    """FP8 scaling state for attention projections."""

    q_proj: NotRequired[Fp8Scales]
    k_proj: NotRequired[Fp8Scales]
    v_proj: NotRequired[Fp8Scales]
    output_proj: NotRequired[Fp8Scales]


class TransformerBlockCache(TypedDict):
    """Cache for a single transformer block including FP8 scales."""

    self_attn: AttentionCache
    cross_attn: NotRequired[AttentionCache]
    self_attn_fp8: NotRequired[Fp8ScalesCache]
    cross_attn_fp8: NotRequired[Fp8ScalesCache]


class TransformerCache(TypedDict):
    """Cache for the entire transformer stack."""

    layers: dict[str, TransformerBlockCache]


def _make_linear(
    in_features: int,
    out_features: int,
    key: PRNGKeyArray,
    use_bias: bool = True,
    use_fp8: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> eqx.nn.Linear | Fp8Linear:
    """Create a linear layer, optionally with FP8 quantization.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        key: PRNG key for initialization
        use_bias: Whether to include a bias term
        use_fp8: Whether to use FP8 quantization
        compute_dtype: Compute dtype for FP8 mode

    Returns:
        Linear layer (Fp8Linear if use_fp8=True, else eqx.nn.Linear)
    """
    if use_fp8:
        return Fp8Linear(
            in_features,
            out_features,
            key=key,
            use_fp8=True,
            compute_dtype=compute_dtype,
        )
    return eqx.nn.Linear(in_features, out_features, use_bias=use_bias, key=key)


def _apply_linear_batched(
    layer: eqx.nn.Linear | Fp8Linear,
    x: Array,
    scales: Fp8Scales | None = None,
) -> tuple[Array, Fp8Scales | None]:
    """Apply a linear layer to batched input, handling FP8 scaling properly.

    For FP8 with delayed scaling, we want to compute scales from the entire
    batch/sequence, not per-element. This function handles that case.

    Args:
        layer: Linear layer (either eqx.nn.Linear or Fp8Linear)
        x: Input tensor of shape (batch, features)
        scales: Optional FP8 scales for delayed scaling mode.
            If None and layer is Fp8Linear, uses current scaling.

    Returns:
        Tuple of (output tensor, updated scales or None)
    """
    if isinstance(layer, Fp8Linear):
        # For FP8, we want scales computed from the full tensor
        # Flatten to 1D, apply linear, reshape back
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        # Apply to first element to get scales, then apply to all with those scales
        if scales is not None:
            # Delayed scaling: use provided scales, update history from full tensor
            # Apply linear with vmap but share the scales
            out_flat, new_scales = jax.vmap(lambda xi: layer(xi, scales=scales))(x_flat)
            # Take the last updated scales (all should be similar)
            new_scales = jax.tree.map(lambda x: x[-1], new_scales)
        else:
            # Current scaling: compute scale from full tensor
            # Just vmap and ignore scales (each gets its own current scale)
            out_flat, _ = jax.vmap(lambda xi: layer(xi, scales=None))(x_flat)
            new_scales = None

        out = out_flat.reshape(*batch_shape, -1)
        return out, new_scales

    # Standard linear: just vmap
    out = jax.vmap(layer)(x)
    return out, None


def _init_fp8_scales_cache(history_length: int) -> Fp8ScalesCache:
    """Initialize FP8 scales cache for all projection layers.

    Args:
        history_length: Length of amax history buffer

    Returns:
        FP8 scales cache for q, k, v, and output projections
    """
    return {
        "q_proj": init_fp8_scales(history_length),
        "k_proj": init_fp8_scales(history_length),
        "v_proj": init_fp8_scales(history_length),
        "output_proj": init_fp8_scales(history_length),
    }


RotaryEmbeddingStyle = Literal["concatenated", "interleaved"]


class RotaryEmbedding(eqx.Module):
    """Rotary Position Embedding (RoPE) for transformer attention.

    This implements the rotary position embedding as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864

    Supports YaRN (Yet another RoPE extensioN) for extended context:
    "YaRN: Efficient Context Window Extension of Large Language Models"
    https://arxiv.org/abs/2309.00071

    Two styles are supported:
    - "concatenated" (default): Used by LLaMA, Qwen, Mistral. Splits head_dim in half.
    - "interleaved": Used by GPT-NeoX. Uses even/odd indices.
    """

    head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True, default=10000.0)
    style: RotaryEmbeddingStyle = eqx.field(static=True, default="interleaved")
    # YaRN parameters
    factor: float = eqx.field(static=True, default=1.0)
    original_max_position_embeddings: int | None = eqx.field(static=True, default=None)
    beta_slow: float = eqx.field(static=True, default=1.0)
    beta_fast: float = eqx.field(static=True, default=32.0)

    @classmethod
    def build(
        cls,
        head_dim: int,
        base: float = 10000.0,
        style: RotaryEmbeddingStyle = "interleaved",
        factor: float = 1.0,
        original_max_position_embeddings: int | None = None,
        beta_slow: float = 1.0,
        beta_fast: float = 32.0,
    ) -> "RotaryEmbedding":
        """Build rotary embedding from parameters.

        Args:
            head_dim: Dimension of each attention head
            base: Base for the frequency computation
            style: RoPE style - "interleaved" (default, GPT-NeoX) or "concatenated" (LLaMA/Qwen)
            factor: YaRN scaling factor (1.0 = no scaling)
            original_max_position_embeddings: Original context length for YaRN
            beta_slow: YaRN slow dimension weight
            beta_fast: YaRN fast dimension weight

        Returns:
            RotaryEmbedding instance
        """
        return cls(
            head_dim=head_dim,
            base=base,
            style=style,
            factor=factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_slow=beta_slow,
            beta_fast=beta_fast,
        )

    @property
    def uses_yarn(self) -> bool:
        return self.factor > 1.0 and self.original_max_position_embeddings is not None

    def _get_rotary_embeddings(self, positions: Array, dtype: jnp.dtype) -> tuple[Array, Array]:
        """Get rotary embeddings for a given sequence length.

        Args:
            positions: Positions of the sequence
            dtype: Data type for the embeddings

        Returns:
            Tuple of (cos_embeddings, sin_embeddings) of shape (seq_len, head_dim//2)
        """
        dim = self.head_dim // 2

        if self.uses_yarn:
            return self._get_yarn_embeddings(positions, dtype, dim)

        # Standard RoPE: Create frequency bands
        freqs = jnp.exp(-jnp.arange(0, dim, dtype=dtype) * jnp.log(self.base) / dim)

        # Compute angles
        angles = positions[:, None] * freqs[None, :]  # (seq_len, dim)

        # Compute cos and sin embeddings
        cos_embeddings = jnp.cos(angles)
        sin_embeddings = jnp.sin(angles)

        return cos_embeddings, sin_embeddings

    def _get_yarn_embeddings(self, positions: Array, dtype: jnp.dtype, dim: int) -> tuple[Array, Array]:
        """Get YaRN RoPE embeddings for extended context.

        YaRN interpolates between original and scaled frequencies based on dimension,
        preserving high-frequency components for short-range dependencies while
        extending low-frequency components for longer context.
        """
        features = self.head_dim
        original_max_pos = self.original_max_position_embeddings
        assert original_max_pos is not None

        # Compute low/high thresholds for interpolation
        low = (features * math.log(original_max_pos / (self.beta_fast * 2 * math.pi))) / (2 * math.log(self.base))
        high = (features * math.log(original_max_pos / (self.beta_slow * 2 * math.pi))) / (2 * math.log(self.base))
        low, high = max(low, 0), min(high, features - 1)

        # Compute original and scaled frequencies
        timescale = self.base ** (jnp.arange(0, features, 2, dtype=jnp.float32) / features)
        rot_freq_extra = 1.0 / timescale  # Original frequencies
        rot_freq_inter = 1.0 / (self.factor * timescale)  # Scaled frequencies

        # Compute interpolation factor per dimension
        high = high if low != high else (high + 0.001)
        dim_indices = jnp.arange(dim, dtype=jnp.float32)
        interp_factor = 1 - jnp.clip((dim_indices - low) / (high - low), 0.0, 1.0)

        # Interpolate between scaled and original frequencies
        rotational_frequency = rot_freq_inter * (1 - interp_factor) + rot_freq_extra * interp_factor

        # Compute angles with high precision (important for YaRN)
        angles = positions[:, None].astype(jnp.float32) * rotational_frequency[None, :]

        # Compute attention scaling factor
        m_scale = 1.0
        attention_scaling = 0.1 * m_scale * math.log(self.factor) + 1.0

        # Compute cos and sin with scaling
        cos_embeddings = (jnp.cos(angles) * attention_scaling).astype(dtype)
        sin_embeddings = (jnp.sin(angles) * attention_scaling).astype(dtype)

        return cos_embeddings, sin_embeddings

    def apply_rotary_embeddings(
        self,
        x: Array,
        positions: Array | None = None,
    ) -> Array:
        """Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor of shape (seq_len, num_heads, head_dim) or (batch, seq_len, num_heads, head_dim)
            positions: Optional position indices of shape (seq_len,) or (batch, seq_len)
                If None, uses sequential positions starting from 0

        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        # Handle both 3D and 4D inputs
        if x.ndim == 4:
            # (batch, seq_len, num_heads, head_dim) - batch first
            bsz, seq_len, _, head_dim = x.shape
            if positions is None:
                positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (bsz, seq_len))
            return self._apply_rotary_4d(x, positions)
        else:
            # (seq_len, num_heads, head_dim)
            seq_len, _, head_dim = x.shape
            assert head_dim == self.head_dim, f"Expected head_dim {self.head_dim}, got {head_dim}"

            if positions is None:
                positions = jnp.arange(seq_len, dtype=x.dtype)

            cos_emb, sin_emb = self._get_rotary_embeddings(positions, x.dtype)
            cos_emb = cos_emb[:, None, :]  # (seq_len, 1, head_dim//2)
            sin_emb = sin_emb[:, None, :]

            match self.style:
                case "concatenated":
                    return self._apply_concatenated(x, cos_emb, sin_emb)
                case "interleaved":
                    return self._apply_interleaved(x, cos_emb, sin_emb)
                case _:
                    raise ValueError(f"Unknown RoPE style: {self.style}. Use 'concatenated' or 'interleaved'.")

    def _apply_rotary_4d(self, x: Array, positions: Array) -> Array:
        """Apply RoPE to 4D tensor (batch, seq, heads, head_dim)."""
        bsz, seq_len, _, head_dim = x.shape
        assert head_dim == self.head_dim

        # Get embeddings for each position in the batch
        # positions is (batch, seq_len), we need cos/sin for each
        cos_emb, sin_emb = self._get_rotary_embeddings(positions.reshape(-1), x.dtype)
        cos_emb = cos_emb.reshape(bsz, seq_len, 1, -1)  # (batch, seq, 1, head_dim//2)
        sin_emb = sin_emb.reshape(bsz, seq_len, 1, -1)

        if self.style == "concatenated":
            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]
            return jnp.concatenate([x1 * cos_emb - x2 * sin_emb, x2 * cos_emb + x1 * sin_emb], axis=-1)
        else:
            x_even = x[..., ::2]
            x_odd = x[..., 1::2]
            rotated_even = x_even * cos_emb - x_odd * sin_emb
            rotated_odd = x_even * sin_emb + x_odd * cos_emb
            result = jnp.zeros_like(x)
            result = result.at[..., ::2].set(rotated_even)
            result = result.at[..., 1::2].set(rotated_odd)
            return result

    def _apply_concatenated(self, x: Array, cos_emb: Array, sin_emb: Array) -> Array:
        """LLaMA/Qwen style: split into first half and second half."""
        head_dim = x.shape[-1]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        return jnp.concatenate([x1 * cos_emb - x2 * sin_emb, x2 * cos_emb + x1 * sin_emb], axis=-1)

    def _apply_interleaved(self, x: Array, cos_emb: Array, sin_emb: Array) -> Array:
        """GPT-NeoX style: use even/odd indices."""
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos_emb - x_odd * sin_emb
        rotated_odd = x_even * sin_emb + x_odd * cos_emb
        result = jnp.zeros_like(x)
        result = result.at[..., ::2].set(rotated_even)
        result = result.at[..., 1::2].set(rotated_odd)
        return result


class SelfAttentionBlock(eqx.Module):
    """Self-attention block with Grouped Query Attention (GQA) support.

    Supports:
    - Standard multi-head attention (num_kv_heads == num_heads)
    - Grouped Query Attention (num_kv_heads < num_heads) for efficient KV caching
    - Rotary Position Embeddings (RoPE) with YaRN extension
    - QK-Norm (Qwen3 style) for training stability
    - Sliding window attention
    - Attention sinks for improved generation quality
    - cuDNN flash attention acceleration (when conditions are met)
    - FP8 quantization
    - LoRA support via apply_linear helper
    """

    q_proj: eqx.nn.Linear | Fp8Linear
    k_proj: eqx.nn.Linear | Fp8Linear
    v_proj: eqx.nn.Linear | Fp8Linear
    output_proj: eqx.nn.Linear | Fp8Linear
    rotary_emb: RotaryEmbedding | None
    q_norm: RMSNorm | None
    k_norm: RMSNorm | None
    sinks: Array | None
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    local_window_size: int | None = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_bias: bool = True,
        causal: bool = False,
        context_length: int | None = None,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        rotary_style: RotaryEmbeddingStyle = "concatenated",
        rotary_factor: float = 1.0,
        rotary_original_max_position_embeddings: int | None = None,
        rotary_beta_slow: float = 1.0,
        rotary_beta_fast: float = 32.0,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        use_attention_sinks: bool = False,
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> "SelfAttentionBlock":
        """Build a self-attention block from configuration parameters.

        Args:
            embed_dim: Model embedding dimension
            num_heads: Number of query attention heads
            key: PRNG key for initialization
            num_kv_heads: Number of key/value heads for GQA (defaults to num_heads)
            head_dim: Dimension per head (defaults to embed_dim // num_heads)
            use_bias: Whether to include bias in projection layers
            causal: Whether to use causal masking
            context_length: Sliding window size (None = full attention)
            use_rotary_embeddings: Whether to use RoPE
            rotary_base: RoPE theta base frequency
            rotary_style: RoPE style ("concatenated" for LLaMA/Qwen, "interleaved" for GPT-NeoX)
            rotary_factor: YaRN scaling factor (1.0 = no scaling)
            rotary_original_max_position_embeddings: Original context length for YaRN
            rotary_beta_slow: YaRN slow dimension weight
            rotary_beta_fast: YaRN fast dimension weight
            use_qk_norm: Whether to apply QK-Norm (Qwen3 style)
            qk_norm_eps: Epsilon for QK-Norm RMSNorm
            use_attention_sinks: Whether to use learnable attention sinks
            use_fp8: Whether to use FP8 quantization
            compute_dtype: Compute dtype for FP8 mode

        Returns:
            A new SelfAttentionBlock instance
        """
        if context_length is not None:
            assert context_length > 1, "context_length must be at least 2"

        keys = jax.random.split(key, 4)

        actual_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        actual_head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        # GQA: num_heads must be divisible by num_kv_heads
        assert num_heads % actual_num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        q_dim = num_heads * actual_head_dim
        kv_dim = actual_num_kv_heads * actual_head_dim

        q_proj = _make_linear(embed_dim, q_dim, keys[0], use_bias, use_fp8, compute_dtype)
        k_proj = _make_linear(embed_dim, kv_dim, keys[1], use_bias, use_fp8, compute_dtype)
        v_proj = _make_linear(embed_dim, kv_dim, keys[2], use_bias, use_fp8, compute_dtype)
        output_proj = _make_linear(q_dim, embed_dim, keys[3], use_bias, use_fp8, compute_dtype)

        # Initialize rotary embeddings if requested
        if use_rotary_embeddings:
            rotary_emb = RotaryEmbedding.build(
                head_dim=actual_head_dim,
                base=rotary_base,
                style=rotary_style,
                factor=rotary_factor,
                original_max_position_embeddings=rotary_original_max_position_embeddings,
                beta_slow=rotary_beta_slow,
                beta_fast=rotary_beta_fast,
            )
        else:
            rotary_emb = None

        # QK-Norm (Qwen3 style)
        if use_qk_norm:
            q_norm = RMSNorm.build(actual_head_dim, eps=qk_norm_eps)
            k_norm = RMSNorm.build(actual_head_dim, eps=qk_norm_eps)
        else:
            q_norm = None
            k_norm = None

        # Attention sinks for softmax stability
        if use_attention_sinks:
            sinks = jnp.zeros((num_heads,), dtype=jnp.float32)
        else:
            sinks = None

        if context_length is not None and not causal:
            warnings.warn("context_length is set but causal is False; overriding causal to True", stacklevel=2)
            causal = True

        local_window_size = None if context_length is None else context_length - 1

        return cls(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=output_proj,
            rotary_emb=rotary_emb,
            q_norm=q_norm,
            k_norm=k_norm,
            sinks=sinks,
            num_heads=num_heads,
            num_kv_heads=actual_num_kv_heads,
            head_dim=actual_head_dim,
            causal=causal,
            local_window_size=local_window_size,
            use_fp8=use_fp8,
        )

    @property
    def embed_dim(self) -> int:
        return self.head_dim * self.num_heads

    def _reshape_q(self, x: Array) -> Array:
        """Reshape Q from (seq_len, q_dim) to (seq_len, num_heads, head_dim)."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    def _reshape_kv(self, x: Array) -> Array:
        """Reshape K/V from (seq_len, kv_dim) to (seq_len, num_kv_heads, head_dim)."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_kv_heads, self.head_dim)

    def _combine_heads(self, x: Array) -> Array:
        """Reshape from (seq_len, num_heads, head_dim) to (seq_len, q_dim)."""
        seq_len, _, _ = x.shape
        return x.reshape(seq_len, -1)

    def __call__(
        self,
        x_btd: Array,
        positions_bt: Array | None = None,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        """Apply self-attention with batched input (for LLM training/inference).

        This is a simplified interface for the common case of batched input
        without FP8 or caching. For caching/FP8, use the `forward` method.

        Args:
            x_btd: Input tensor of shape (batch, seq_len, embed_dim)
            positions_bt: Position indices for RoPE, shape (batch, seq_len).
                If None, uses sequential positions starting from 0.
            key: PRNG key (unused, for API compatibility)
            inference: Whether in inference mode (unused, for API compatibility)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        chex.assert_rank(x_btd, 3)
        bsz, tsz, _ = x_btd.shape

        if positions_bt is None:
            positions_bt = jnp.broadcast_to(jnp.arange(tsz)[None, :], (bsz, tsz))

        # Project to Q, K, V using apply_linear for LoRA support
        q_bthd = apply_linear(x_btd, self.q_proj).reshape(bsz, tsz, self.num_heads, self.head_dim)
        k_bthd = apply_linear(x_btd, self.k_proj).reshape(bsz, tsz, self.num_kv_heads, self.head_dim)
        v_bthd = apply_linear(x_btd, self.v_proj).reshape(bsz, tsz, self.num_kv_heads, self.head_dim)

        # Apply QK-Norm before RoPE (Qwen3 style)
        if self.q_norm is not None:
            q_bthd = self.q_norm(q_bthd)
        if self.k_norm is not None:
            k_bthd = self.k_norm(k_bthd)

        # Apply rotary embeddings
        if self.rotary_emb is not None:
            q_bthd = self.rotary_emb.apply_rotary_embeddings(q_bthd, positions=positions_bt)
            k_bthd = self.rotary_emb.apply_rotary_embeddings(k_bthd, positions=positions_bt)

        # Handle attention sinks via bias
        bias = None
        if self.sinks is not None:
            bias = self.sinks[None, None, :, None]

        # Check if cuDNN flash attention can be used
        use_cudnn = can_use_cudnn_attention(
            dtype=q_bthd.dtype,
            head_dim=self.head_dim,
            seq_len=tsz,
            has_bias=bias is not None,
        )
        implementation = "cudnn" if use_cudnn else None

        # Apply attention
        ctx_bthd = jax.nn.dot_product_attention(
            q_bthd,
            k_bthd,
            v_bthd,
            bias=bias,
            is_causal=self.causal,
            scale=1.0 / (self.head_dim**0.5),
            local_window_size=(self.local_window_size, 0) if self.local_window_size else None,
            implementation=implementation,
        )

        # Combine heads and project output
        ctx_btd = ctx_bthd.reshape(bsz, tsz, self.num_heads * self.head_dim)
        return apply_linear(ctx_btd, self.output_proj)

    def init_cache(self, max_len: int | None = None, dtype: jnp.dtype | None = None) -> AttentionCache:
        """Initialize cache for the input.

        Args:
            dtype: The dtype of the cache
            max_len: The maximum length of the cache

        Returns:
            Cache with fixed-length k and v tensors
        """
        if max_len is None:
            max_len = self.local_window_size
        if max_len is None:
            raise ValueError("context_length or max_len must be set for caching")

        # Create fixed-length cache (uses num_kv_heads for GQA efficiency)
        k_cache = jnp.zeros((max_len, self.num_kv_heads, self.head_dim), dtype=dtype)
        v_cache = jnp.zeros((max_len, self.num_kv_heads, self.head_dim), dtype=dtype)

        return {"k": k_cache, "v": v_cache, "position": jnp.array(0, dtype=jnp.int32)}

    def init_mask(
        self,
        seq_len: int,
        add_cache: bool = False,
        batch_dim: bool = False,
    ) -> Array:
        """Initialize the attention matrix mask.

        Args:
            seq_len: The length of the sequence
            add_cache: Whether to add the cache to the mask
            batch_dim: Whether to add a batch dimension to the mask

        Returns:
            The attention matrix mask of shape (bsz, 1, seq_len, seq_len + cache_len)
            if batch_dim is True, otherwise (seq_len, seq_len + cache_len).
        """
        t, s, o = seq_len, seq_len, 0
        if add_cache:
            if self.local_window_size is None:
                raise ValueError("local_window_size must be set for caching")
            s += self.local_window_size
            o -= self.local_window_size
        mask = jnp.tril(jnp.ones((t, s), dtype=jnp.bool_), k=-o)
        if self.local_window_size is not None:
            neg_mask = ~jnp.tril(jnp.ones((t, s), dtype=jnp.bool_), k=-(self.local_window_size + 1 + o))
            mask = mask & neg_mask
        mask = mask.reshape(1, 1, t, s) if batch_dim else mask.reshape(t, s)
        return mask

    def forward(
        self,
        x_tn: Array,
        *,
        mask: Array | None = None,
        cache: AttentionCache | None = None,
        fp8_scales: Fp8ScalesCache | None = None,
    ) -> tuple[Array, AttentionCache, Fp8ScalesCache | None]:
        """Apply self-attention with optional GQA and cuDNN acceleration.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: The cached key and value tensors (fixed-length)
            fp8_scales: Optional FP8 scales for delayed scaling mode.
                If None and use_fp8=True, uses current scaling.

        Returns:
            Tuple of (output tensor, updated attention cache, updated FP8 scales)
        """
        chex.assert_rank(x_tn, 2)

        # Project inputs to queries, keys, and values
        q_scales = fp8_scales.get("q_proj") if fp8_scales else None
        k_scales = fp8_scales.get("k_proj") if fp8_scales else None
        v_scales = fp8_scales.get("v_proj") if fp8_scales else None

        q, new_q_scales = _apply_linear_batched(self.q_proj, x_tn, q_scales)
        k, new_k_scales = _apply_linear_batched(self.k_proj, x_tn, k_scales)
        v, new_v_scales = _apply_linear_batched(self.v_proj, x_tn, v_scales)

        # Reshape to multihead format (Q has num_heads, K/V have num_kv_heads)
        q = self._reshape_q(q)
        k = self._reshape_kv(k)
        v = self._reshape_kv(v)

        # Apply QK-Norm before RoPE (Qwen3 style)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        seq_len = q.shape[0]
        if self.rotary_emb is not None:
            # Determine position indices for rotary embeddings
            if cache is not None:
                start_pos = cache["position"]
            else:
                start_pos = 0
            positions = jnp.arange(seq_len) + start_pos
            q = self.rotary_emb.apply_rotary_embeddings(q, positions=positions)
            k = self.rotary_emb.apply_rotary_embeddings(k, positions=positions)

        if cache is not None:
            k_cache = cache["k"]
            v_cache = cache["v"]
            p = cache["position"]

            k = jax.lax.dynamic_update_slice(k_cache, k, (p, 0, 0))
            v = jax.lax.dynamic_update_slice(v_cache, v, (p, 0, 0))

            new_position = p + seq_len

        else:
            new_position = seq_len

        # Handle attention sinks via bias
        bias = None
        if self.sinks is not None:
            bias = self.sinks[None, :, None]  # (1, num_heads, 1) for unbatched

        # Check if cuDNN can be used
        use_cudnn = can_use_cudnn_attention(
            dtype=q.dtype,
            head_dim=self.head_dim,
            seq_len=seq_len,
            has_bias=bias is not None,
        )
        implementation = "cudnn" if use_cudnn else None

        if seq_len == 1:
            attn_output = jax.nn.dot_product_attention(q, k, v, bias=bias, implementation=implementation)

        elif mask is not None:
            attn_output = jax.nn.dot_product_attention(q, k, v, mask=mask, bias=bias, implementation=implementation)

        elif cache is not None:
            attn_output = jax.nn.dot_product_attention(
                q,
                k,
                v,
                bias=bias,
                is_causal=self.causal,
                local_window_size=(self.local_window_size, 0) if self.local_window_size is not None else None,
                implementation=implementation,
            )

        else:
            attn_output = jax.nn.dot_product_attention(
                q,
                k,
                v,
                bias=bias,
                is_causal=self.causal,
                local_window_size=(self.local_window_size, 0) if self.local_window_size is not None else None,
                implementation=implementation,
            )

        attn_output = self._combine_heads(attn_output)
        out_scales = fp8_scales.get("output_proj") if fp8_scales else None
        output, new_out_scales = _apply_linear_batched(self.output_proj, attn_output, out_scales)

        if self.local_window_size is not None:
            k = k[-self.local_window_size :]
            v = v[-self.local_window_size :]

        # Build updated FP8 scales cache
        updated_fp8_scales: Fp8ScalesCache | None = None
        if fp8_scales is not None and self.use_fp8:
            updated_fp8_scales = {}
            if new_q_scales is not None:
                updated_fp8_scales["q_proj"] = new_q_scales
            if new_k_scales is not None:
                updated_fp8_scales["k_proj"] = new_k_scales
            if new_v_scales is not None:
                updated_fp8_scales["v_proj"] = new_v_scales
            if new_out_scales is not None:
                updated_fp8_scales["output_proj"] = new_out_scales

        return output, {"k": k, "v": v, "position": new_position}, updated_fp8_scales


class CrossAttentionBlock(eqx.Module):
    """Cross-attention block using jax.nn.dot_product_attention."""

    q_proj: eqx.nn.Linear | Fp8Linear
    k_proj: eqx.nn.Linear | Fp8Linear
    v_proj: eqx.nn.Linear | Fp8Linear
    output_proj: eqx.nn.Linear | Fp8Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)
    rotary_emb: RotaryEmbedding | None = eqx.field(default=None)

    @classmethod
    def build(
        cls,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> "CrossAttentionBlock":
        """Build cross-attention block from parameters.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            key: PRNG key for initialization
            use_rotary_embeddings: Whether to use rotary position embeddings
            rotary_base: Base for rotary position embeddings
            use_fp8: Whether to use FP8 quantization
            compute_dtype: Data type for computation

        Returns:
            CrossAttentionBlock instance
        """
        keys = jax.random.split(key, 4)

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        q_proj = _make_linear(embed_dim, embed_dim, keys[0], use_fp8=use_fp8, compute_dtype=compute_dtype)
        k_proj = _make_linear(embed_dim, embed_dim, keys[1], use_fp8=use_fp8, compute_dtype=compute_dtype)
        v_proj = _make_linear(embed_dim, embed_dim, keys[2], use_fp8=use_fp8, compute_dtype=compute_dtype)
        output_proj = _make_linear(embed_dim, embed_dim, keys[3], use_fp8=use_fp8, compute_dtype=compute_dtype)

        # Initialize rotary embeddings if requested
        rotary_emb: RotaryEmbedding | None = None
        if use_rotary_embeddings:
            rotary_emb = RotaryEmbedding.build(head_dim=head_dim, base=rotary_base)

        return cls(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=output_proj,
            num_heads=num_heads,
            head_dim=head_dim,
            use_fp8=use_fp8,
            rotary_emb=rotary_emb,
        )

    def _reshape_for_multihead(self, x: Array) -> Array:
        """Reshape from (seq_len, embed_dim) to (seq_len, num_heads, head_dim)."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    def _combine_heads(self, x: Array) -> Array:
        """Reshape from (seq_len, num_heads, head_dim) to (seq_len, embed_dim)."""
        seq_len, _, _ = x.shape
        return x.reshape(seq_len, -1)

    def init_cache(
        self,
        kv_sn: Array,
        fp8_scales: Fp8ScalesCache | None = None,
    ) -> tuple[AttentionCache, Fp8ScalesCache | None]:
        """Initialize cache for the input.

        Args:
            kv_sn: Key/value input tensor of shape (kv_seq_len, embed_dim)
            fp8_scales: Optional FP8 scales for delayed scaling mode

        Returns:
            Tuple of (attention cache, updated FP8 scales)
        """
        chex.assert_rank(kv_sn, 2)
        k_scales = fp8_scales.get("k_proj") if fp8_scales else None
        v_scales = fp8_scales.get("v_proj") if fp8_scales else None

        k, new_k_scales = _apply_linear_batched(self.k_proj, kv_sn, k_scales)
        v, new_v_scales = _apply_linear_batched(self.v_proj, kv_sn, v_scales)

        # Reshape to multihead format
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)

        # Build updated FP8 scales
        updated_fp8_scales: Fp8ScalesCache | None = None
        if fp8_scales is not None and self.use_fp8:
            new_fp8: Fp8ScalesCache = {}
            for key, val in fp8_scales.items():
                new_fp8[key] = val  # type: ignore[literal-required]
            if new_k_scales is not None:
                new_fp8["k_proj"] = new_k_scales
            if new_v_scales is not None:
                new_fp8["v_proj"] = new_v_scales
            updated_fp8_scales = new_fp8

        return {"k": k, "v": v, "position": 0}, updated_fp8_scales

    def forward(
        self,
        q_tn: Array,
        *,
        kv_sn: Array | None = None,
        cache: AttentionCache | None = None,
        fp8_scales: Fp8ScalesCache | None = None,
    ) -> tuple[Array, AttentionCache, Fp8ScalesCache | None]:
        """Apply cross-attention.

        Args:
            q_tn: Query input tensor of shape (q_seq_len, embed_dim)
            kv_sn: Key/value input tensor of shape (kv_seq_len, embed_dim).
                If not provided, then `cache` must be provided.
            cache: The cached key and value tensors. If not provided, then
                `kv_sn` must be provided.
            fp8_scales: Optional FP8 scales for delayed scaling mode.

        Returns:
            Tuple of (output tensor, attention cache, updated FP8 scales)
        """
        chex.assert_rank(q_tn, 2)

        # Project queries
        q_scales = fp8_scales.get("q_proj") if fp8_scales else None
        q, new_q_scales = _apply_linear_batched(self.q_proj, q_tn, q_scales)
        q = self._reshape_for_multihead(q)
        q_seq_len = q.shape[0]

        # Track updated scales
        new_k_scales = None
        new_v_scales = None

        # Use cached key/value if provided
        if cache is not None:
            k = cache["k"]
            v = cache["v"]
            q_position = cache["position"]
        elif kv_sn is not None:
            chex.assert_rank(kv_sn, 2)
            k_scales = fp8_scales.get("k_proj") if fp8_scales else None
            v_scales = fp8_scales.get("v_proj") if fp8_scales else None

            k, new_k_scales = _apply_linear_batched(self.k_proj, kv_sn, k_scales)
            v, new_v_scales = _apply_linear_batched(self.v_proj, kv_sn, v_scales)
            k = self._reshape_for_multihead(k)
            v = self._reshape_for_multihead(v)
            q_position = 0
        else:
            raise ValueError("Either `cache` or `kv_sn` must be provided.")

        # Apply rotary embeddings to queries and keys if enabled
        if self.rotary_emb is None:
            q_rot = q
            k_rot = k
        else:
            q_positions = jnp.arange(q_seq_len) + q_position
            k_positions = jnp.arange(k.shape[0])
            q_rot = self.rotary_emb.apply_rotary_embeddings(q, positions=q_positions)
            k_rot = self.rotary_emb.apply_rotary_embeddings(k, positions=k_positions)

        # Apply dot product attention
        attn_output = jax.nn.dot_product_attention(
            q_rot,
            k_rot,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            is_causal=False,
        )

        # Combine heads
        attn_output = self._combine_heads(attn_output)

        # Final projection
        out_scales = fp8_scales.get("output_proj") if fp8_scales else None
        output, new_out_scales = _apply_linear_batched(self.output_proj, attn_output, out_scales)

        # Build updated FP8 scales cache
        updated_fp8_scales: Fp8ScalesCache | None = None
        if fp8_scales is not None and self.use_fp8:
            updated_fp8_scales = {}
            if new_q_scales is not None:
                updated_fp8_scales["q_proj"] = new_q_scales
            if new_k_scales is not None:
                updated_fp8_scales["k_proj"] = new_k_scales
            if new_v_scales is not None:
                updated_fp8_scales["v_proj"] = new_v_scales
            if new_out_scales is not None:
                updated_fp8_scales["output_proj"] = new_out_scales

        return output, {"k": k, "v": v, "position": q_position + q_seq_len}, updated_fp8_scales


NormLayer = eqx.nn.LayerNorm | RMSNorm
FeedForwardLayer = eqx.nn.MLP | SwiGLU
NormType = Literal["layernorm", "rmsnorm"]
FeedForwardType = Literal["mlp", "swiglu"]


class TransformerBlock(eqx.Module):
    """Transformer block supporting both standard and LLM configurations.

    This block supports:
    - Pre-norm architecture with LayerNorm or RMSNorm
    - MLP or SwiGLU feed-forward networks
    - Self-attention with GQA, RoPE, QK-Norm, sliding window, attention sinks
    - Optional cross-attention
    - FP8 quantization
    - LoRA fine-tuning (via apply_linear in underlying modules)

    Can be constructed either:
    - Directly with pre-built components: TransformerBlock(self_attn=..., feed_forward=..., ...)
    - From configuration parameters: TransformerBlock.build(embed_dim=..., num_heads=..., ...)
    """

    self_attn: SelfAttentionBlock = eqx.field()
    feed_forward: FeedForwardLayer = eqx.field()
    attn_norm: NormLayer = eqx.field()
    mlp_norm: NormLayer = eqx.field()
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)
    cross_attn: CrossAttentionBlock | None = eqx.field(default=None)
    cross_attn_norm: NormLayer | None = eqx.field(default=None)
    context_length: int | None = eqx.field(static=True, default=None)

    @classmethod
    def build(
        cls,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        key: PRNGKeyArray,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        causal: bool = False,
        cross_attention: bool = False,
        context_length: int | None = None,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        rotary_style: RotaryEmbeddingStyle = "concatenated",
        rotary_factor: float = 1.0,
        rotary_original_max_position_embeddings: int | None = None,
        rotary_beta_slow: float = 1.0,
        rotary_beta_fast: float = 32.0,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        use_attention_sinks: bool = False,
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-6,
        feedforward_type: FeedForwardType = "mlp",
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> "TransformerBlock":
        """Build a transformer block from configuration parameters.

        Args:
            embed_dim: Model embedding dimension
            num_heads: Number of query attention heads
            ff_dim: Feed-forward hidden dimension
            key: PRNG key for initialization
            num_kv_heads: Number of key/value heads for GQA (defaults to num_heads)
            head_dim: Dimension per head (defaults to embed_dim // num_heads)
            causal: Whether to use causal masking
            cross_attention: Whether to include cross-attention
            context_length: Sliding window size (None = full attention)
            use_rotary_embeddings: Whether to use RoPE
            rotary_base: RoPE theta base frequency
            rotary_style: RoPE style ("concatenated" for LLaMA/Qwen, "interleaved" for GPT-NeoX)
            rotary_factor: YaRN scaling factor (1.0 = no scaling)
            rotary_original_max_position_embeddings: Original context length for YaRN
            rotary_beta_slow: YaRN slow dimension weight
            rotary_beta_fast: YaRN fast dimension weight
            use_qk_norm: Whether to apply QK-Norm (Qwen3 style)
            qk_norm_eps: Epsilon for QK-Norm RMSNorm
            use_attention_sinks: Whether to use learnable attention sinks
            norm_type: Normalization type ("layernorm" or "rmsnorm")
            norm_eps: Epsilon for normalization layers
            feedforward_type: Feed-forward type ("mlp" or "swiglu")
            use_fp8: Whether to use FP8 quantization
            compute_dtype: Compute dtype for FP8 mode

        Returns:
            A new TransformerBlock instance
        """
        keys = jax.random.split(key, 3)

        actual_head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        self_attn = SelfAttentionBlock.build(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            key=keys[0],
            causal=causal,
            context_length=context_length,
            use_rotary_embeddings=use_rotary_embeddings,
            rotary_base=rotary_base,
            rotary_style=rotary_style,
            rotary_factor=rotary_factor,
            rotary_original_max_position_embeddings=rotary_original_max_position_embeddings,
            rotary_beta_slow=rotary_beta_slow,
            rotary_beta_fast=rotary_beta_fast,
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_attention_sinks=use_attention_sinks,
            use_fp8=use_fp8,
            compute_dtype=compute_dtype,
        )

        if cross_attention:
            cross_attn = CrossAttentionBlock.build(
                embed_dim=embed_dim,
                num_heads=num_heads,
                key=keys[1],
                use_rotary_embeddings=use_rotary_embeddings,
                rotary_base=rotary_base,
                use_fp8=use_fp8,
                compute_dtype=compute_dtype,
            )
            cross_attn_norm: NormLayer | None = (
                RMSNorm.build(embed_dim, eps=norm_eps) if norm_type == "rmsnorm" else eqx.nn.LayerNorm(embed_dim)
            )
        else:
            cross_attn = None
            cross_attn_norm = None

        # Create normalization layers based on norm_type
        if norm_type == "rmsnorm":
            attn_norm: NormLayer = RMSNorm.build(embed_dim, eps=norm_eps)
            mlp_norm: NormLayer = RMSNorm.build(embed_dim, eps=norm_eps)
        else:
            attn_norm = eqx.nn.LayerNorm(embed_dim)
            mlp_norm = eqx.nn.LayerNorm(embed_dim)

        # Create feed-forward layer based on feedforward_type
        if feedforward_type == "swiglu":
            feed_forward: FeedForwardLayer = SwiGLU.build(embed_dim, ff_dim, key=keys[2])
        else:
            feed_forward = eqx.nn.MLP(
                in_size=embed_dim,
                out_size=embed_dim,
                width_size=ff_dim,
                depth=1,
                activation=jax.nn.gelu,
                key=keys[2],
            )

        return cls(
            self_attn=self_attn,
            feed_forward=feed_forward,
            attn_norm=attn_norm,
            mlp_norm=mlp_norm,
            num_heads=num_heads,
            head_dim=actual_head_dim,
            causal=causal,
            use_fp8=use_fp8,
            cross_attn=cross_attn,
            cross_attn_norm=cross_attn_norm,
            context_length=context_length,
        )

    @property
    def embed_dim(self) -> int:
        return self.head_dim * self.num_heads

    def init_cache(
        self,
        max_len: int | None = None,
        dtype: jnp.dtype | None = None,
        context_sn: Array | None = None,
        fp8_history_length: int | None = None,
    ) -> TransformerBlockCache:
        """Initialize cache for the input.

        Args:
            max_len: The maximum length of the cache
            dtype: Data type for cache tensors
            context_sn: Context sequence for cross-attention
            fp8_history_length: If provided, also initialize FP8 scales for delayed scaling

        Returns:
            Cache containing attention state and optionally FP8 scales
        """
        if dtype is None and context_sn is not None:
            dtype = context_sn.dtype
        self_attn_cache = self.self_attn.init_cache(max_len=max_len, dtype=dtype)
        cache: TransformerBlockCache = {"self_attn": self_attn_cache}
        if self.cross_attn is not None:
            if context_sn is None:
                raise ValueError("context_sn must be provided if cross_attn is not None")
            cross_cache, _ = self.cross_attn.init_cache(kv_sn=context_sn)
            cache["cross_attn"] = cross_cache

        # Initialize FP8 scales if requested
        if fp8_history_length is not None:
            if self.self_attn.use_fp8:
                cache["self_attn_fp8"] = _init_fp8_scales_cache(fp8_history_length)
            if self.cross_attn is not None and self.cross_attn.use_fp8:
                cache["cross_attn_fp8"] = _init_fp8_scales_cache(fp8_history_length)

        return cache

    def init_mask(
        self,
        seq_len: int,
        add_cache: bool = False,
        batch_dim: bool = False,
    ) -> Array:
        return self.self_attn.init_mask(
            seq_len,
            add_cache=add_cache,
            batch_dim=batch_dim,
        )

    def __call__(
        self,
        x_btd: Array,
        positions_bt: Array | None = None,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        """Apply transformer block with batched input (for LLM training/inference).

        This is a simplified interface for the common case of batched input
        without FP8 or caching. For caching/FP8, use the `forward` method.

        Args:
            x_btd: Input tensor of shape (batch, seq_len, embed_dim)
            positions_bt: Position indices for RoPE, shape (batch, seq_len).
                If None, uses sequential positions starting from 0.
            key: PRNG key for dropout (if applicable)
            inference: Whether in inference mode

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        attn_key = None
        if key is not None:
            attn_key, _ = jax.random.split(key, 2)

        # Self-attention with pre-norm
        normed = self.attn_norm(x_btd)
        y_btd = x_btd + self.self_attn(normed, positions_bt, key=attn_key, inference=inference)

        # Feed-forward with pre-norm
        y_btd = y_btd + self.feed_forward(self.mlp_norm(y_btd))

        return y_btd

    def forward(
        self,
        x_tn: Array,
        *,
        context_sn: Array | None = None,
        mask: Array | None = None,
        cache: TransformerBlockCache | None = None,
    ) -> tuple[Array, TransformerBlockCache]:
        """Apply transformer block with unbatched input and caching support.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            context_sn: Optional context for cross-attention
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: Optional dictionary containing cached key and value tensors
                and FP8 scales

        Returns:
            The output tensor and the updated cache
        """
        chex.assert_rank(x_tn, 2)

        # Self-attention block with pre-norm
        norm_x = jax.vmap(self.attn_norm)(x_tn)

        self_attn_fp8 = cache.get("self_attn_fp8") if cache else None
        attn_output, self_attn_cache, updated_self_fp8 = self.self_attn.forward(
            x_tn=norm_x,
            mask=mask,
            cache=None if cache is None else cache.get("self_attn"),
            fp8_scales=self_attn_fp8,
        )
        updated_cache: TransformerBlockCache = {"self_attn": self_attn_cache}
        if updated_self_fp8 is not None:
            updated_cache["self_attn_fp8"] = updated_self_fp8

        x_tn = x_tn + attn_output

        # Cross-attention block (if enabled) with pre-norm
        if self.cross_attn is not None:
            assert self.cross_attn_norm is not None

            norm_x = jax.vmap(self.cross_attn_norm)(x_tn)

            cross_attn_fp8 = cache.get("cross_attn_fp8") if cache else None
            cross_attn_output, cross_cache, updated_cross_fp8 = self.cross_attn.forward(
                q_tn=norm_x,
                kv_sn=context_sn,
                cache=None if cache is None else cache.get("cross_attn"),
                fp8_scales=cross_attn_fp8,
            )
            updated_cache["cross_attn"] = cross_cache
            if updated_cross_fp8 is not None:
                updated_cache["cross_attn_fp8"] = updated_cross_fp8

            x_tn = x_tn + cross_attn_output

        # Feed-forward block with pre-norm
        norm_x_tn = self.mlp_norm(x_tn)
        ff_output_tn = self.feed_forward(norm_x_tn)
        x_tn = x_tn + ff_output_tn

        return x_tn, updated_cache


class TransformerStack(eqx.Module):
    """A stack of transformer blocks."""

    layers: tuple[TransformerBlock, ...]
    num_layers: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        cross_attention: bool = False,
        context_length: int | None = None,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> "TransformerStack":
        """Build transformer stack from parameters.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            num_layers: Number of transformer blocks
            key: PRNG key for initialization
            causal: Whether to use causal masking
            cross_attention: Whether to include cross-attention
            context_length: Maximum context length (for sliding window)
            use_rotary_embeddings: Whether to use rotary position embeddings
            rotary_base: Base for rotary position embeddings
            use_fp8: Whether to use FP8 quantization
            compute_dtype: Data type for computation

        Returns:
            TransformerStack instance
        """
        keys = jax.random.split(key, num_layers)

        layers = tuple(
            TransformerBlock.build(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                key=keys[i],
                causal=causal,
                cross_attention=cross_attention,
                context_length=context_length,
                use_rotary_embeddings=use_rotary_embeddings,
                rotary_base=rotary_base,
                use_fp8=use_fp8,
                compute_dtype=compute_dtype,
            )
            for i in range(num_layers)
        )

        return cls(
            layers=layers,
            num_layers=num_layers,
            causal=causal,
            use_fp8=use_fp8,
        )

    def init_cache(
        self,
        dtype: jnp.dtype | None = None,
        x_tn: Array | None = None,
        fp8_history_length: int | None = None,
    ) -> TransformerCache:
        """Initialize cache for all layers.

        Args:
            dtype: Data type for cache tensors
            x_tn: Context sequence for cross-attention
            fp8_history_length: If provided, also initialize FP8 scales for delayed scaling

        Returns:
            Cache containing attention state for all layers
        """
        cache = {}
        for i, layer in enumerate(self.layers):
            cache[f"layer_{i}"] = layer.init_cache(dtype=dtype, context_sn=x_tn, fp8_history_length=fp8_history_length)
        return {"layers": cache}

    def init_mask(
        self,
        seq_len: int,
        add_cache: bool = False,
        batch_dim: bool = False,
    ) -> Array:
        return self.layers[0].init_mask(
            seq_len,
            add_cache=add_cache,
            batch_dim=batch_dim,
        )

    def forward(
        self,
        x_tn: Array,
        *,
        context_sn: Array | None = None,
        mask: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Apply transformer stack.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            context_sn: Optional context for cross-attention
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The output tensor and the updated cache
        """
        # Initialize layer caches
        if cache is None:
            cache = {"layers": {}}

        # Updated cache will be built
        updated_cache: TransformerCache = {"layers": {}}

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache["layers"].get(f"layer_{i}")

            x_tn, updated_cache["layers"][f"layer_{i}"] = layer.forward(
                x_tn,
                context_sn=context_sn,
                mask=mask,
                cache=layer_cache,
            )

        return x_tn, updated_cache


class Transformer(eqx.Module):
    token_embedding: eqx.nn.Embedding
    layers: TransformerStack
    layer_norm: eqx.nn.LayerNorm
    embed_dim: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)
    output_layer: eqx.nn.Linear | Fp8Linear | None = eqx.field(default=None)
    context_length: int | None = eqx.field(static=True, default=None)

    @classmethod
    def build(
        cls,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        output_size: int | None = None,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        cross_attention: bool = False,
        context_length: int | None = None,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> "Transformer":
        """Build transformer from parameters.

        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            num_layers: Number of transformer blocks
            output_size: Size of output projection (None for no projection)
            key: PRNG key for initialization
            causal: Whether to use causal masking
            cross_attention: Whether to include cross-attention
            context_length: Maximum context length (for sliding window)
            use_rotary_embeddings: Whether to use rotary position embeddings
            rotary_base: Base for rotary position embeddings
            use_fp8: Whether to use FP8 quantization
            compute_dtype: Data type for computation

        Returns:
            Transformer instance
        """
        # Calculate number of keys needed
        num_keys = 3 if output_size is None else 4
        keys = jax.random.split(key, num_keys)

        token_embedding = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])

        layers = TransformerStack.build(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            key=keys[2],
            causal=causal,
            cross_attention=cross_attention,
            context_length=context_length,
            use_rotary_embeddings=use_rotary_embeddings,
            rotary_base=rotary_base,
            use_fp8=use_fp8,
            compute_dtype=compute_dtype,
        )

        layer_norm = eqx.nn.LayerNorm(embed_dim)
        output_layer: eqx.nn.Linear | Fp8Linear | None = None
        if output_size is not None:
            output_layer = _make_linear(embed_dim, output_size, keys[3], use_fp8=use_fp8, compute_dtype=compute_dtype)

        return cls(
            token_embedding=token_embedding,
            layers=layers,
            layer_norm=layer_norm,
            embed_dim=embed_dim,
            causal=causal,
            use_fp8=use_fp8,
            output_layer=output_layer,
            context_length=context_length,
        )

    def init_cache(
        self,
        dtype: jnp.dtype | None = None,
        x_tn: Array | None = None,
        fp8_history_length: int | None = None,
    ) -> TransformerCache:
        """Initialize cache for the input."""
        return self.layers.init_cache(dtype=dtype, x_tn=x_tn, fp8_history_length=fp8_history_length)

    def init_mask(
        self,
        seq_len: int,
        add_cache: bool = False,
        batch_dim: bool = False,
    ) -> Array:
        return self.layers.init_mask(
            seq_len,
            add_cache=add_cache,
            batch_dim=batch_dim,
        )

    def encode(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Encode the input sequence.

        Args:
            x: Input token indices of shape (seq_len)
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The encoded representation and the updated cache
        """
        x_embedded = jax.vmap(self.token_embedding)(x)
        x_embedded, updated_cache = self.layers.forward(x_embedded, mask=mask, cache=cache)
        output = jax.vmap(self.layer_norm)(x_embedded)
        return output, updated_cache

    def decode(
        self,
        x_t: Array,
        context_s: Array,
        *,
        mask: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Decode with self-attention and cross-attention.

        Args:
            x_t: Input token indices, shape (seq_len)
            context_s: Context from encoder (token indices or embedded),
                shape (context_len, embed_dim)
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The decoded representation and the updated cache
        """
        x_embedded = jax.vmap(self.token_embedding)(x_t)
        context_embedded = jax.vmap(self.token_embedding)(context_s)
        x_embedded, updated_cache = self.layers.forward(
            x_embedded,
            context_sn=context_embedded,
            mask=mask,
            cache=cache,
        )
        output = jax.vmap(self.layer_norm)(x_embedded)
        return output, updated_cache

    def forward(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Forward pass for encoder-only or decoder-only transformers.

        Args:
            x: Input token indices of shape (seq_len)
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The output representation and the updated cache
        """
        chex.assert_rank(x, 1)
        output, updated_cache = self.encode(x, mask=mask, cache=cache)
        if self.output_layer is not None:
            output, _ = _apply_linear_batched(self.output_layer, output)
        return output, updated_cache

    def predict_sequence(self, x_seq: Array) -> Array:
        output, _ = self.forward(x=x_seq)
        return output

    def generate_sequence(
        self,
        prompt_seq: Array,
        max_len: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        key: PRNGKeyArray | None = None,
        jit_level: int | None = None,
    ) -> Array:
        """Generate a sequence autoregressively with KV caching.

        Args:
            prompt_seq: Input token indices of shape (prompt_len,)
            max_len: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Optional top-k sampling parameter
            key: PRNG key for sampling
            jit_level: JIT level for the scan function

        Returns:
            Generated sequence of shape (prompt_len + max_len,)
        """
        if key is None:
            key = jax.random.key(0)

        prompt_len = prompt_seq.shape[0]

        total_len = prompt_len + max_len
        output_seq = jnp.zeros(total_len, dtype=prompt_seq.dtype)
        output_seq = output_seq.at[:prompt_len].set(prompt_seq)

        # Initialize cache with prompt
        dtype = self.token_embedding.weight.dtype
        cache = self.init_cache(dtype=dtype)
        mask = self.init_mask(prompt_len, add_cache=True, batch_dim=False)
        _, cache = self.encode(prompt_seq, cache=cache, mask=mask)

        # Define scan function for autoregressive generation
        def scan_fn(
            carry: tuple[Array, int, TransformerCache, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[Array, int, TransformerCache, PRNGKeyArray], Array]:
            output_seq, pos, cache, rng = carry
            current_token = jax.lax.dynamic_slice(output_seq, (pos,), (1,))

            # Forward pass with cache update
            logits, new_cache = self.forward(
                x=current_token,
                cache=cache,
            )

            logits = logits[-1] / temperature
            if top_k is not None:
                top_logits, top_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.full_like(logits, float("-inf"))
                logits = logits.at[top_indices].set(top_logits)
            rng, subrng = jax.random.split(rng)
            next_token = jax.random.categorical(subrng, logits[None, ...])[0]
            new_output_seq = jax.lax.dynamic_update_slice(output_seq, next_token[None], (pos + 1,))

            return (new_output_seq, pos + 1, new_cache, rng), next_token

        init_carry = (output_seq, prompt_len - 1, cache, key)
        (final_seq, _, _, _), _ = xax_scan(scan_fn, init_carry, length=max_len, jit_level=jit_level)
        return final_seq
