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


class RMSNorm(eqx.Module):
    """RMSNorm over the last dimension (no bias), matching LLaMA/Qwen style.

    RMS normalization is more efficient than LayerNorm and works well for LLMs.
    It normalizes by the root mean square without centering (no mean subtraction).
    """

    weight: Array
    eps: float = 1e-6

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.weight = jnp.ones((dim,), dtype=jnp.float32)
        self.eps = eps

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

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        *,
        key: PRNGKeyArray,
        use_bias: bool = False,
    ) -> None:
        """Initialize SwiGLU feed-forward layer.

        Args:
            embed_dim: Input/output embedding dimension
            hidden_dim: Hidden dimension (defaults to 4 * embed_dim)
            key: PRNG key for initialization
            use_bias: Whether to use bias in linear layers
        """
        if hidden_dim is None:
            hidden_dim = embed_dim * 4

        k1, k2, k3 = jax.random.split(key, 3)
        self.gate = eqx.nn.Linear(embed_dim, hidden_dim, use_bias=use_bias, key=k1)
        self.up = eqx.nn.Linear(embed_dim, hidden_dim, use_bias=use_bias, key=k2)
        self.down = eqx.nn.Linear(hidden_dim, embed_dim, use_bias=use_bias, key=k3)

    def __call__(self, x: Array) -> Array:
        chex.assert_rank(x, {2, 3})
        gate_out = jax.nn.silu(x @ self.gate.weight.T)
        up_out = x @ self.up.weight.T
        if self.gate.bias is not None:
            gate_out = gate_out + self.gate.bias
        if self.up.bias is not None:
            up_out = up_out + self.up.bias
        hidden = gate_out * up_out
        out = hidden @ self.down.weight.T
        if self.down.bias is not None:
            out = out + self.down.bias
        return out


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
    position: int  # Position counter for rotary embeddings


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
    use_fp8: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> eqx.nn.Linear | Fp8Linear:
    """Create a linear layer, optionally with FP8 quantization.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        key: PRNG key for initialization
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
    return eqx.nn.Linear(in_features, out_features, key=key)


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

    head_dim: int = eqx.field()
    base: float = eqx.field()
    style: RotaryEmbeddingStyle = eqx.field(static=True)
    # YaRN parameters
    factor: float = eqx.field()
    original_max_position_embeddings: int | None = eqx.field()
    beta_slow: float = eqx.field()
    beta_fast: float = eqx.field()

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        style: RotaryEmbeddingStyle = "interleaved",
        factor: float = 1.0,
        original_max_position_embeddings: int | None = None,
        beta_slow: float = 1.0,
        beta_fast: float = 32.0,
    ) -> None:
        """Initialize rotary embedding.

        Args:
            head_dim: Dimension of each attention head
            base: Base for the frequency computation
            style: RoPE style - "interleaved" (default, GPT-NeoX) or "concatenated" (LLaMA/Qwen)
            factor: YaRN scaling factor (1.0 = no scaling)
            original_max_position_embeddings: Original context length for YaRN
            beta_slow: YaRN slow dimension weight
            beta_fast: YaRN fast dimension weight
        """
        self.head_dim = head_dim
        self.base = base
        self.style = style
        self.factor = factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast

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


def llm_linear(x_btd: Array, linear: eqx.nn.Linear) -> Array:
    """Apply linear layer with einsum for better performance.

    Also supports LoRALinear layers which have weight_oi/bias_o instead of weight/bias.
    This enables LoRA fine-tuning of pre-trained models.
    """
    if isinstance(linear, LoRALinear):
        # LoRALinear: use weight_oi and bias_o, plus LoRA delta
        y_bto = jnp.einsum("...d,od->...o", x_btd, linear.weight_oi)
        if linear.bias_o is not None:
            y_bto = y_bto + linear.bias_o
        # Add LoRA contribution: (x @ A) @ B * alpha
        delta_bto = (x_btd @ linear.lora_a_ir) @ linear.lora_b_ro * linear.alpha
        return y_bto + delta_bto
    else:
        # Standard eqx.nn.Linear
        y_bto = jnp.einsum("...d,od->...o", x_btd, linear.weight)
        if linear.bias is not None:
            y_bto = y_bto + linear.bias
        return y_bto


class LLMAttention(eqx.Module):
    """Grouped-query attention with rotary embeddings for LLMs.

    Features:
    - Grouped Query Attention (GQA) for efficient KV caching
    - Rotary Position Embeddings (RoPE) with YaRN extension
    - QK-Norm (Qwen3 style) for training stability
    - Sliding window attention for long contexts
    - Attention sinks for improved generation quality
    - LoRA support via llm_linear helper
    - cuDNN flash attention acceleration
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    rotary: RotaryEmbedding
    q_norm: RMSNorm | None  # QK-Norm for Qwen3
    k_norm: RMSNorm | None  # QK-Norm for Qwen3
    q_heads: int
    kv_heads: int
    head_dim: int
    dropout_rate: float
    sliding_window_size: int | None = None  # None = full attention, int = window size
    sinks: Array | None = None  # Learnable attention sinks for softmax stability

    def __call__(
        self,
        x_btd: Array,
        positions_bt: Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        chex.assert_rank(x_btd, 3)
        chex.assert_rank(positions_bt, 2)
        bsz, tsz, _ = x_btd.shape
        chex.assert_shape(positions_bt, (bsz, tsz))

        q_bthd = llm_linear(x_btd, self.q_proj).reshape(bsz, tsz, self.q_heads, self.head_dim)
        k_bthd = llm_linear(x_btd, self.k_proj).reshape(bsz, tsz, self.kv_heads, self.head_dim)
        v_bthd = llm_linear(x_btd, self.v_proj).reshape(bsz, tsz, self.kv_heads, self.head_dim)

        # Apply QK-Norm before RoPE (Qwen3 style)
        if self.q_norm is not None:
            q_bthd = self.q_norm(q_bthd)
        if self.k_norm is not None:
            k_bthd = self.k_norm(k_bthd)

        # Apply rotary embeddings
        positions_flat = positions_bt.reshape(-1)
        q_flat = q_bthd.reshape(-1, self.q_heads, self.head_dim)
        k_flat = k_bthd.reshape(-1, self.kv_heads, self.head_dim)

        q_flat = self.rotary.apply_rotary_embeddings(q_flat, positions=positions_flat)
        k_flat = self.rotary.apply_rotary_embeddings(k_flat, positions=positions_flat)

        q_bthd = q_flat.reshape(bsz, tsz, self.q_heads, self.head_dim)
        k_bthd = k_flat.reshape(bsz, tsz, self.kv_heads, self.head_dim)

        # Use jax.nn.dot_product_attention for efficient attention computation.
        # Shape is already (batch, seq, heads, dim) which matches JAX's expected (B, T, N, H).
        # Note: attention sinks are handled via bias if present.
        bias = None
        if self.sinks is not None:
            # Attention sinks add a learnable bias to attention logits per head.
            # Shape: (1, 1, heads, 1) broadcast to (batch, seq, heads, seq)
            bias = self.sinks[None, None, :, None]

        # Check if cuDNN flash attention can be used
        use_cudnn = can_use_cudnn_attention(
            dtype=q_bthd.dtype,
            head_dim=self.head_dim,
            seq_len=tsz,
            has_bias=bias is not None,
        )
        implementation = "cudnn" if use_cudnn else None

        ctx_bthd = jax.nn.dot_product_attention(
            q_bthd,
            k_bthd,
            v_bthd,
            bias=bias,
            is_causal=True,
            scale=1.0 / (self.head_dim**0.5),
            local_window_size=(self.sliding_window_size, 0) if self.sliding_window_size else None,
            implementation=implementation,
        )

        ctx_btd = ctx_bthd.reshape(bsz, tsz, self.q_heads * self.head_dim)
        return llm_linear(ctx_btd, self.o_proj)


class LLMFeedForward(eqx.Module):
    """SwiGLU feed-forward layer for LLMs with LoRA support.

    Uses gated linear unit with SiLU activation:
        output = down(silu(gate(x)) * up(x))

    Supports LoRA fine-tuning via llm_linear helper.
    """

    gate: eqx.nn.Linear
    up: eqx.nn.Linear
    down: eqx.nn.Linear

    def __call__(self, x_btd: Array) -> Array:
        chex.assert_rank(x_btd, {2, 3})
        gated_btd = jax.nn.silu(llm_linear(x_btd, self.gate)) * llm_linear(x_btd, self.up)
        return llm_linear(gated_btd, self.down)


class LLMBlock(eqx.Module):
    """Single transformer layer with pre-norm for LLMs.

    Uses RMSNorm and SwiGLU feed-forward, which are standard in modern LLMs
    like LLaMA, Qwen, and Mistral.
    """

    attn: LLMAttention
    attn_norm: RMSNorm
    mlp_norm: RMSNorm
    mlp: LLMFeedForward

    def __call__(
        self,
        x_btd: Array,
        positions_bt: Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        attn_key = None
        if key is not None:
            attn_key, _ = jax.random.split(key, 2)
        normed = self.attn_norm(x_btd)
        y_btd = x_btd + self.attn(normed, positions_bt, key=attn_key, inference=inference)
        y_btd = y_btd + self.mlp(self.mlp_norm(y_btd))
        return y_btd


class SelfAttentionBlock(eqx.Module):
    """Self-attention block with Grouped Query Attention (GQA) support.

    Supports:
    - Standard multi-head attention (num_kv_heads == num_heads)
    - Grouped Query Attention (num_kv_heads < num_heads) for efficient KV caching
    - Rotary Position Embeddings (RoPE)
    - Sliding window attention
    - cuDNN flash attention acceleration (when conditions are met)
    - FP8 quantization
    """

    q_proj: eqx.nn.Linear | Fp8Linear = eqx.field()
    k_proj: eqx.nn.Linear | Fp8Linear = eqx.field()
    v_proj: eqx.nn.Linear | Fp8Linear = eqx.field()
    output_proj: eqx.nn.Linear | Fp8Linear = eqx.field()
    rotary_emb: RotaryEmbedding | None = eqx.field()
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    local_window_size: int | None = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        causal: bool = False,
        context_length: int | None = None,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        rotary_style: RotaryEmbeddingStyle = "concatenated",
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        """Initialize self-attention block.

        Args:
            embed_dim: Model embedding dimension
            num_heads: Number of query attention heads
            key: PRNG key for initialization
            num_kv_heads: Number of key/value heads for GQA (defaults to num_heads)
            head_dim: Dimension per head (defaults to embed_dim // num_heads)
            causal: Whether to use causal masking
            context_length: Sliding window size (None = full attention)
            use_rotary_embeddings: Whether to use RoPE
            rotary_base: RoPE theta base frequency
            rotary_style: RoPE style ("concatenated" for LLaMA/Qwen, "interleaved" for GPT-NeoX)
            use_fp8: Whether to use FP8 quantization
            compute_dtype: Compute dtype for FP8 mode
        """
        if context_length is not None:
            assert context_length > 1, "context_length must be at least 2"

        keys = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.use_fp8 = use_fp8

        # GQA: num_heads must be divisible by num_kv_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        q_dim = num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = _make_linear(embed_dim, q_dim, keys[0], use_fp8, compute_dtype)
        self.k_proj = _make_linear(embed_dim, kv_dim, keys[1], use_fp8, compute_dtype)
        self.v_proj = _make_linear(embed_dim, kv_dim, keys[2], use_fp8, compute_dtype)
        self.output_proj = _make_linear(q_dim, embed_dim, keys[3], use_fp8, compute_dtype)

        # Initialize rotary embeddings if requested
        if use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_dim,
                base=rotary_base,
                style=rotary_style,
            )
        else:
            self.rotary_emb = None

        if context_length is not None and not causal:
            warnings.warn("context_length is set but causal is False; overriding causal to True", stacklevel=2)
            causal = True

        self.causal = causal
        self.local_window_size = None if context_length is None else context_length - 1

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

    def init_cache(self, dtype: jnp.dtype | None = None) -> AttentionCache:
        """Initialize cache for the input.

        Args:
            dtype: The dtype of the cache

        Returns:
            Cache with fixed-length k and v tensors
        """
        if self.local_window_size is None:
            raise ValueError("context_length must be set for caching")

        # Create fixed-length cache (uses num_kv_heads for GQA efficiency)
        k_cache = jnp.zeros((self.local_window_size, self.num_kv_heads, self.head_dim), dtype=dtype)
        v_cache = jnp.zeros((self.local_window_size, self.num_kv_heads, self.head_dim), dtype=dtype)

        return {"k": k_cache, "v": v_cache, "position": 0}

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
            k = jnp.concatenate([k_cache, k], axis=0)
            v = jnp.concatenate([v_cache, v], axis=0)

            new_position = cache["position"] + seq_len

        else:
            new_position = seq_len

        # Check if cuDNN can be used
        use_cudnn = can_use_cudnn_attention(
            dtype=q.dtype,
            head_dim=self.head_dim,
            seq_len=seq_len,
            has_bias=False,
        )
        implementation = "cudnn" if use_cudnn else None

        if seq_len == 1:
            attn_output = jax.nn.dot_product_attention(q, k, v, implementation=implementation)

        elif mask is not None:
            attn_output = jax.nn.dot_product_attention(q, k, v, mask=mask, implementation=implementation)

        elif cache is not None:
            raise NotImplementedError("For training with a cache, provide a mask instead.")

        else:
            attn_output = jax.nn.dot_product_attention(
                q,
                k,
                v,
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
    rotary_emb: RotaryEmbedding | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        keys = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_fp8 = use_fp8
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = _make_linear(embed_dim, embed_dim, keys[0], use_fp8, compute_dtype)
        self.k_proj = _make_linear(embed_dim, embed_dim, keys[1], use_fp8, compute_dtype)
        self.v_proj = _make_linear(embed_dim, embed_dim, keys[2], use_fp8, compute_dtype)
        self.output_proj = _make_linear(embed_dim, embed_dim, keys[3], use_fp8, compute_dtype)

        # Initialize rotary embeddings if requested
        if use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_dim,
                base=rotary_base,
            )
        else:
            self.rotary_emb = None

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


class TransformerBlock(eqx.Module):
    self_attn: SelfAttentionBlock
    cross_attn: CrossAttentionBlock | None
    feed_forward: eqx.nn.MLP
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    layer_norm3: eqx.nn.LayerNorm | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    context_length: int | None = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        cross_attention: bool = False,
        context_length: int | None = None,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
        use_fp8: bool = False,
        compute_dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        keys = jax.random.split(key, 3)

        self.use_fp8 = use_fp8

        self.self_attn = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            key=keys[0],
            causal=causal,
            context_length=context_length,
            use_rotary_embeddings=use_rotary_embeddings,
            rotary_base=rotary_base,
            use_fp8=use_fp8,
            compute_dtype=compute_dtype,
        )

        if cross_attention:
            self.cross_attn = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                key=keys[1],
                use_rotary_embeddings=use_rotary_embeddings,
                rotary_base=rotary_base,
                use_fp8=use_fp8,
                compute_dtype=compute_dtype,
            )
            self.layer_norm3 = eqx.nn.LayerNorm(embed_dim)

        else:
            self.cross_attn = None
            self.layer_norm3 = None

        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)

        self.feed_forward = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=ff_dim,
            depth=1,
            activation=jax.nn.gelu,
            key=keys[2],
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.context_length = context_length

    @property
    def embed_dim(self) -> int:
        return self.head_dim * self.num_heads

    def init_cache(
        self,
        dtype: jnp.dtype | None = None,
        context_sn: Array | None = None,
        fp8_history_length: int | None = None,
    ) -> TransformerBlockCache:
        """Initialize cache for the input.

        Args:
            dtype: Data type for cache tensors
            context_sn: Context sequence for cross-attention
            fp8_history_length: If provided, also initialize FP8 scales for delayed scaling

        Returns:
            Cache containing attention state and optionally FP8 scales
        """
        if dtype is None and context_sn is not None:
            dtype = context_sn.dtype
        cache: TransformerBlockCache = {"self_attn": self.self_attn.init_cache(dtype=dtype)}
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

    def forward(
        self,
        x_tn: Array,
        *,
        context_sn: Array | None = None,
        mask: Array | None = None,
        cache: TransformerBlockCache | None = None,
    ) -> tuple[Array, TransformerBlockCache]:
        """Apply transformer block.

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
        norm_x = jax.vmap(self.layer_norm1)(x_tn)

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
            assert self.layer_norm3 is not None

            norm_x = jax.vmap(self.layer_norm3)(x_tn)

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
        norm_x = jax.vmap(self.layer_norm2)(x_tn)
        ff_output = jax.vmap(self.feed_forward)(norm_x)
        x_tn = x_tn + ff_output

        return x_tn, updated_cache


class TransformerStack(eqx.Module):
    """A stack of transformer blocks."""

    layers: tuple[TransformerBlock, ...]
    num_layers: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    use_fp8: bool = eqx.field(static=True)

    def __init__(
        self,
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
    ) -> None:
        keys = jax.random.split(key, num_layers)

        self.layers = tuple(
            TransformerBlock(
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

        self.num_layers = num_layers
        self.causal = causal
        self.use_fp8 = use_fp8

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
    output_layer: eqx.nn.Linear | Fp8Linear | None
    layer_norm: eqx.nn.LayerNorm
    embed_dim: int = eqx.field()
    causal: bool = eqx.field()
    context_length: int | None = eqx.field()
    use_fp8: bool = eqx.field(static=True)

    def __init__(
        self,
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
    ) -> None:
        # Calculate number of keys needed
        num_keys = 3 if output_size is None else 4
        keys = jax.random.split(key, num_keys)

        self.token_embedding = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])
        self.use_fp8 = use_fp8

        self.layers = TransformerStack(
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

        self.layer_norm = eqx.nn.LayerNorm(embed_dim)
        if output_size is not None:
            self.output_layer = _make_linear(embed_dim, output_size, keys[3], use_fp8, compute_dtype)
        else:
            self.output_layer = None

        self.embed_dim = embed_dim
        self.causal = causal
        self.context_length = context_length

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
