"""Attention mechanisms for transformer models.

This module implements standard attention mechanisms for transformers, but
supporting a fixed-size context window and caching that can be used to train
transformers which can be unrolled with a fixed-length cache.
"""

import math
import warnings
from typing import NotRequired, TypedDict

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from xax.utils.jax import scan as xax_scan


class RotaryEmbedding(eqx.Module):
    """Rotary Position Embedding (RoPE) for transformer attention.

    This implements the rotary position embedding as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """

    head_dim: int = eqx.field()
    base: float = eqx.field()

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
    ) -> None:
        """Initialize rotary embedding.

        Args:
            head_dim: Dimension of each attention head
            base: Base for the frequency computation
        """
        self.head_dim = head_dim
        self.base = base

    def _get_rotary_embeddings(self, positions: Array, dtype: jnp.dtype) -> tuple[Array, Array]:
        """Get rotary embeddings for a given sequence length.

        Args:
            positions: Positions of the sequence
            dtype: Data type for the embeddings

        Returns:
            Tuple of (cos_embeddings, sin_embeddings) of shape (seq_len, head_dim//2)
        """
        # Create frequency bands
        dim = self.head_dim // 2
        freqs = jnp.exp(-jnp.arange(0, dim, dtype=dtype) * jnp.log(self.base) / dim)

        # Compute angles
        angles = positions[:, None] * freqs[None, :]  # (seq_len, dim)

        # Compute cos and sin embeddings
        cos_embeddings = jnp.cos(angles)
        sin_embeddings = jnp.sin(angles)

        return cos_embeddings, sin_embeddings

    def apply_rotary_embeddings(
        self,
        x: Array,
        positions: Array | None = None,
    ) -> Array:
        """Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor of shape (seq_len, num_heads, head_dim)
            positions: Optional position indices of shape (seq_len,)
                If None, uses sequential positions starting from 0

        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        seq_len, _, head_dim = x.shape
        assert head_dim == self.head_dim, f"Expected head_dim {self.head_dim}, got {head_dim}"

        # Get rotary embeddings
        if positions is None:
            positions = jnp.arange(seq_len, dtype=x.dtype)
        cos_emb, sin_emb = self._get_rotary_embeddings(positions, x.dtype)

        # Reshape to (seq_len, 1, head_dim//2) for broadcasting
        cos_emb = cos_emb[:, None, :]  # (seq_len, 1, head_dim//2)
        sin_emb = sin_emb[:, None, :]  # (seq_len, 1, head_dim//2)

        # Split input into even and odd dimensions
        x_even = x[..., ::2]  # (seq_len, num_heads, head_dim//2)
        x_odd = x[..., 1::2]  # (seq_len, num_heads, head_dim//2)

        # Apply rotation
        rotated_even = x_even * cos_emb - x_odd * sin_emb
        rotated_odd = x_even * sin_emb + x_odd * cos_emb

        # Interleave back together
        result = jnp.zeros_like(x)
        result = result.at[..., ::2].set(rotated_even)
        result = result.at[..., 1::2].set(rotated_odd)

        return result


class AttentionCache(TypedDict):
    k: Array
    v: Array
    position: int  # Position counter for rotary embeddings


class AttentionCacheDict(TypedDict):
    self_attn: AttentionCache
    cross_attn: NotRequired[AttentionCache]


class TransformerCache(TypedDict):
    """Cache for the entire transformer stack."""

    layers: dict[str, AttentionCacheDict]


class SelfAttentionBlock(eqx.Module):
    """Self-attention block using jax.nn.dot_product_attention."""

    q_proj: eqx.nn.Linear = eqx.field()
    k_proj: eqx.nn.Linear = eqx.field()
    v_proj: eqx.nn.Linear = eqx.field()
    output_proj: eqx.nn.Linear = eqx.field()
    rotary_emb: RotaryEmbedding | None = eqx.field()
    num_heads: int = eqx.field()
    head_dim: int = eqx.field()
    causal: bool = eqx.field()
    local_window_size: int | None = eqx.field()

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        context_length: int | None = None,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
    ) -> None:
        if context_length is not None:
            assert context_length > 1, "context_length must be at least 2"

        keys = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.output_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])

        # Initialize rotary embeddings if requested
        if use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_dim,
                base=rotary_base,
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

    def _reshape_for_multihead(self, x: Array) -> Array:
        """Reshape from (seq_len, embed_dim) to (seq_len, num_heads, head_dim)."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    def _combine_heads(self, x: Array) -> Array:
        """Reshape from (seq_len, num_heads, head_dim) to (seq_len, embed_dim)."""
        _, n, h = x.shape
        return x.reshape(-1, n * h)

    def init_cache(self, dtype: jnp.dtype | None = None) -> AttentionCache:
        """Initialize cache for the input.

        Args:
            dtype: The dtype of the cache

        Returns:
            Cache with fixed-length k and v tensors
        """
        if self.local_window_size is None:
            raise ValueError("context_length must be set for caching")

        # Create fixed-length cache
        k_cache = jnp.zeros((self.local_window_size, self.num_heads, self.head_dim), dtype=dtype)
        v_cache = jnp.zeros((self.local_window_size, self.num_heads, self.head_dim), dtype=dtype)

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
    ) -> tuple[Array, AttentionCache]:
        """Apply self-attention.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: The cached key and value tensors (fixed-length)

        Returns:
            The output tensor of shape (seq_len, embed_dim) and updated cache
        """
        chex.assert_rank(x_tn, 2)

        # Project inputs to queries, keys, and values
        q = jax.vmap(self.q_proj)(x_tn)
        k = jax.vmap(self.k_proj)(x_tn)
        v = jax.vmap(self.v_proj)(x_tn)

        # Reshape to multihead format
        q = self._reshape_for_multihead(q)
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)

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

        if seq_len == 1:
            attn_output = jax.nn.dot_product_attention(q, k, v)

        elif mask is not None:
            attn_output = jax.nn.dot_product_attention(q, k, v, mask=mask)

        elif cache is not None:
            raise NotImplementedError("For training with a cache, provide a mask instead.")

        else:
            attn_output = jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=self.causal,
                local_window_size=(self.local_window_size, 0) if self.local_window_size is not None else None,
            )

        attn_output = self._combine_heads(attn_output)
        output = jax.vmap(self.output_proj)(attn_output)

        if self.local_window_size is not None:
            k = k[-self.local_window_size :]
            v = v[-self.local_window_size :]

        return output, {"k": k, "v": v, "position": new_position}


class CrossAttentionBlock(eqx.Module):
    """Cross-attention block using jax.nn.dot_product_attention."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    rotary_emb: RotaryEmbedding | None
    num_heads: int = eqx.field()
    head_dim: int = eqx.field()

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        use_rotary_embeddings: bool = False,
        rotary_base: float = 10000.0,
    ) -> None:
        keys = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.output_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])

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

    def init_cache(self, kv_sn: Array) -> AttentionCache:
        """Initialize cache for the input."""
        chex.assert_rank(kv_sn, 2)
        k = jax.vmap(self.k_proj)(kv_sn)
        v = jax.vmap(self.v_proj)(kv_sn)
        # Reshape to multihead format
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)
        return {"k": k, "v": v, "position": 0}

    def forward(
        self,
        q_tn: Array,
        *,
        kv_sn: Array | None = None,
        cache: AttentionCache | None = None,
    ) -> tuple[Array, AttentionCache]:
        """Apply cross-attention.

        Args:
            q_tn: Query input tensor of shape (q_seq_len, embed_dim)
            kv_sn: Key/value input tensor of shape (kv_seq_len, embed_dim).
                If not provided, then `cache` must be provided.
            cache: The cached key and value tensors. If not provided, then
                `kv_sn` must be provided.

        Returns:
            The output tensor of shape (q_seq_len, embed_dim)
        """
        chex.assert_rank(q_tn, 2)

        # Project inputs to queries, keys, and values
        q = jax.vmap(self.q_proj)(q_tn)
        q = self._reshape_for_multihead(q)
        q_seq_len = q.shape[0]

        # Use cached key/value if provided
        if cache is not None:
            k = cache["k"]
            v = cache["v"]
            q_position = cache["position"]
        elif kv_sn is not None:
            chex.assert_rank(kv_sn, 2)
            k = jax.vmap(self.k_proj)(kv_sn)
            v = jax.vmap(self.v_proj)(kv_sn)
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
        output = jax.vmap(self.output_proj)(attn_output)

        return output, {"k": k, "v": v, "position": q_position + q_seq_len}


class TransformerBlock(eqx.Module):
    self_attn: SelfAttentionBlock
    cross_attn: CrossAttentionBlock | None
    feed_forward: eqx.nn.MLP
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    layer_norm3: eqx.nn.LayerNorm | None
    num_heads: int = eqx.field()
    head_dim: int = eqx.field()
    causal: bool = eqx.field()
    context_length: int | None = eqx.field()

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
    ) -> None:
        keys = jax.random.split(key, 3)

        self.self_attn = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            key=keys[0],
            causal=causal,
            context_length=context_length,
            use_rotary_embeddings=use_rotary_embeddings,
            rotary_base=rotary_base,
        )

        if cross_attention:
            self.cross_attn = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                key=keys[1],
                use_rotary_embeddings=use_rotary_embeddings,
                rotary_base=rotary_base,
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

    def init_cache(self, dtype: jnp.dtype | None = None, context_sn: Array | None = None) -> AttentionCacheDict:
        """Initialize cache for the input."""
        if dtype is None and context_sn is not None:
            dtype = context_sn.dtype
        cache: AttentionCacheDict = {"self_attn": self.self_attn.init_cache(dtype=dtype)}
        if self.cross_attn is not None:
            if context_sn is None:
                raise ValueError("x_tn must be provided if cross_attn is not None")
            cache["cross_attn"] = self.cross_attn.init_cache(kv_sn=context_sn)
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
        cache: AttentionCacheDict | None = None,
    ) -> tuple[Array, AttentionCacheDict]:
        """Apply transformer block.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            context_sn: Optional context for cross-attention
            mask: Optional mask of shape (batch_size, num_heads, seq_len,
                seq_len + cache_len)
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The output tensor and the updated cache
        """
        chex.assert_rank(x_tn, 2)

        # Self-attention block with pre-norm
        norm_x = jax.vmap(self.layer_norm1)(x_tn)

        attn_output, self_attn_cache = self.self_attn.forward(
            x_tn=norm_x,
            mask=mask,
            cache=None if cache is None else cache["self_attn"],
        )
        updated_cache: AttentionCacheDict = {"self_attn": self_attn_cache}

        x_tn = x_tn + attn_output

        # Cross-attention block (if enabled) with pre-norm
        if self.cross_attn is not None:
            assert self.layer_norm3 is not None

            norm_x = jax.vmap(self.layer_norm3)(x_tn)

            cross_attn_output, updated_cache["cross_attn"] = self.cross_attn.forward(
                q_tn=norm_x,
                kv_sn=context_sn,
                cache=None if cache is None else cache.get("cross_attn"),
            )

            x_tn = x_tn + cross_attn_output

        # Feed-forward block with pre-norm
        norm_x = jax.vmap(self.layer_norm2)(x_tn)
        ff_output = jax.vmap(self.feed_forward)(norm_x)
        x_tn = x_tn + ff_output

        return x_tn, updated_cache


class TransformerStack(eqx.Module):
    """A stack of transformer blocks."""

    layers: tuple[TransformerBlock, ...]
    num_layers: int = eqx.field()
    causal: bool = eqx.field()

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
            )
            for i in range(num_layers)
        )

        self.num_layers = num_layers
        self.causal = causal

    def init_cache(self, dtype: jnp.dtype | None = None, x_tn: Array | None = None) -> TransformerCache:
        """Initialize cache for the input."""
        cache = {}
        for i, layer in enumerate(self.layers):
            cache[f"layer_{i}"] = layer.init_cache(dtype=dtype, context_sn=x_tn)
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
    output_layer: eqx.nn.Linear | None
    layer_norm: eqx.nn.LayerNorm
    embed_dim: int = eqx.field()
    causal: bool = eqx.field()
    context_length: int | None = eqx.field()

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
    ) -> None:
        # Calculate number of keys needed
        num_keys = 3 if output_size is None else 4
        keys = jax.random.split(key, num_keys)

        self.token_embedding = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])

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
        )

        self.layer_norm = eqx.nn.LayerNorm(embed_dim)
        if output_size is not None:
            self.output_layer = eqx.nn.Linear(embed_dim, output_size, key=keys[3])
        else:
            self.output_layer = None

        self.embed_dim = embed_dim
        self.causal = causal
        self.context_length = context_length

    def init_cache(self, dtype: jnp.dtype | None = None, x_tn: Array | None = None) -> TransformerCache:
        """Initialize cache for the input."""
        return self.layers.init_cache(dtype=dtype, x_tn=x_tn)

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
        # Token embedding
        x_embedded = jax.vmap(self.token_embedding)(x)

        # Apply transformer stack
        x_embedded, updated_cache = self.layers.forward(x_embedded, mask=mask, cache=cache)

        # Apply final layer norm
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
        # Token embedding for x
        x_embedded = jax.vmap(self.token_embedding)(x_t)

        # Token embedding for context if needed
        context_embedded = jax.vmap(self.token_embedding)(context_s)

        # Apply transformer stack with cross-attention
        x_embedded, updated_cache = self.layers.forward(
            x_embedded,
            context_sn=context_embedded,
            mask=mask,
            cache=cache,
        )

        # Apply final layer norm
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

        # Apply output layer if it exists
        if self.output_layer is not None:
            output = jax.vmap(self.output_layer)(output)

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
            key = jax.random.PRNGKey(0)

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
