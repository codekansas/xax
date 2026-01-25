"""Lightweight JAX LLM reference implementations."""

import functools
import json
import logging
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Literal, overload

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray
from omegaconf import MISSING
from pydantic import AliasChoices, BaseModel, ConfigDict, Field as PydanticField

from xax.arch.attention import (
    AttentionCache,
    CrossAttentionBlock,
    RMSNorm,
    SelfAttentionBlock,
    SwiGLU,
    TransformerBlock,
    TransformerBlockCache,
    apply_linear,
)
from xax.core.conf import field
from xax.utils.jax import filter_jit as xax_filter_jit

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Please install huggingface_hub: pip install huggingface-hub") from e

try:
    from safetensors import safe_open
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Please install safetensors: pip install safetensors") from e

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """Model hyperparameters for decoder-only LLMs."""

    vocab_size: int = field(MISSING, help="Vocabulary size for token embeddings")
    embed_dim: int = field(MISSING, help="Model embedding dimension")
    q_heads: int = field(MISSING, help="Number of query attention heads")
    kv_heads: int = field(MISSING, help="Number of key/value attention heads (for GQA)")
    head_dim: int = field(MISSING, help="Dimension per attention head")
    num_layers: int = field(MISSING, help="Number of transformer layers")
    max_tsz: int = field(32768, help="Maximum sequence length")
    rope_theta: float = field(10_000.0, help="RoPE theta base frequency")
    dropout_rate: float = field(0.0, help="Dropout rate")
    mlp_mult: int = field(4, help="MLP hidden dimension multiplier")
    mlp_hidden_dim: int | None = field(None, help="Explicit MLP hidden dimension (overrides mlp_mult)")
    rms_eps: float = field(1e-6, help="RMSNorm epsilon for numerical stability")
    attention_bias: bool = field(False, help="Whether attention projections have bias")
    mlp_bias: bool = field(False, help="Whether MLP projections have bias")
    rope_factor: float = field(1.0, help="RoPE scaling factor (1.0 = no scaling)")
    rope_original_max_position_embeddings: int | None = field(None, help="Original training context length for YaRN")
    rope_beta_slow: float = field(1.0, help="YaRN slow dimension interpolation weight")
    rope_beta_fast: float = field(32.0, help="YaRN fast dimension interpolation weight")
    sliding_window_size: int | None = field(None, help="Sliding window size (None = full attention)")
    layer_attention_types: tuple[str, ...] | None = field(None, help="Per-layer attention type ('sliding' or 'full')")
    use_attention_sinks: bool = field(False, help="Use learnable attention sinks for stability")
    use_qk_norm: bool = field(False, help="Apply QK normalization in attention (Qwen3 style)")

    def with_vocab(self, vocab_size: int) -> "LLMConfig":
        return replace(self, vocab_size=vocab_size)

    @property
    def uses_yarn(self) -> bool:
        return self.rope_factor > 1.0 and self.rope_original_max_position_embeddings is not None


class LLM(eqx.Module):
    """Minimal decoder-only LLM."""

    embed: eqx.nn.Embedding
    blocks: tuple[TransformerBlock, ...]
    norm: RMSNorm
    lm_head: eqx.nn.Linear
    config: LLMConfig = eqx.field(static=True)

    # These are used to support adding extra tokens to the vocabulary.
    extra_embed: eqx.nn.Embedding | None = None
    extra_lm_head: eqx.nn.Linear | None = None
    tied_extra_embed: bool = eqx.field(static=True, default=False)

    @classmethod
    def build(
        cls,
        config: LLMConfig,
        *,
        extra_tokens: int | None = None,
        tied_extra_embed: bool = False,
        key: PRNGKeyArray,
    ) -> "LLM":
        """Initialize LLM with random weights."""
        key, k_emb = jax.random.split(key)
        embed = eqx.nn.Embedding(config.vocab_size, config.embed_dim, key=k_emb)
        blocks: list[TransformerBlock] = []
        mlp_width = config.mlp_hidden_dim or (config.embed_dim * config.mlp_mult)
        attn_bias = config.attention_bias
        mlp_bias = config.mlp_bias

        for i in range(config.num_layers):
            key, k_attn, k_mlp = jax.random.split(key, 3)

            # Determine sliding window for this layer
            layer_window: int | None = None
            if config.layer_attention_types is not None and i < len(config.layer_attention_types):
                if "sliding" in config.layer_attention_types[i]:
                    layer_window = config.sliding_window_size

            # Create attention block with proper settings
            self_attn = SelfAttentionBlock.build(
                embed_dim=config.embed_dim,
                num_heads=config.q_heads,
                num_kv_heads=config.kv_heads,
                head_dim=config.head_dim,
                key=k_attn,
                use_bias=attn_bias,
                causal=True,
                context_length=layer_window + 1 if layer_window else None,
                use_rotary_embeddings=True,
                rotary_base=config.rope_theta,
                rotary_style="concatenated",
                rotary_factor=config.rope_factor,
                rotary_original_max_position_embeddings=config.rope_original_max_position_embeddings,
                rotary_beta_slow=config.rope_beta_slow,
                rotary_beta_fast=config.rope_beta_fast,
                use_qk_norm=config.use_qk_norm,
                use_attention_sinks=config.use_attention_sinks,
            )

            # Create feed-forward layer
            mlp = SwiGLU.build(config.embed_dim, mlp_width, key=k_mlp, use_bias=mlp_bias)

            blocks.append(
                TransformerBlock(
                    self_attn=self_attn,
                    feed_forward=mlp,
                    attn_norm=RMSNorm.build(config.embed_dim, eps=config.rms_eps),
                    mlp_norm=RMSNorm.build(config.embed_dim, eps=config.rms_eps),
                    num_heads=config.q_heads,
                    head_dim=config.head_dim,
                    causal=True,
                    use_fp8=False,
                    cross_attn=None,
                    cross_attn_norm=None,
                    context_length=layer_window + 1 if layer_window else None,
                )
            )

        norm = RMSNorm.build(config.embed_dim, eps=config.rms_eps)
        key, k_head = jax.random.split(key)
        lm_head = eqx.nn.Linear(config.embed_dim, config.vocab_size, use_bias=False, key=k_head)

        if extra_tokens is None:
            return cls(
                embed=embed,
                blocks=tuple(blocks),
                norm=norm,
                lm_head=lm_head,
                config=config,
            )

        # Adds extra embeddings for the extra tokens.
        key, k_extra_embed, k_extra_lm_head = jax.random.split(key, 3)
        extra_embed = eqx.nn.Embedding(
            extra_tokens,
            config.embed_dim,
            key=k_extra_embed,
            dtype=embed.weight.dtype,
        )
        extra_lm_head = (
            None
            if tied_extra_embed
            else eqx.nn.Linear(
                config.embed_dim,
                extra_tokens,
                use_bias=False,
                dtype=lm_head.weight.dtype,
                key=k_extra_lm_head,
            )
        )
        return cls(
            embed=embed,
            blocks=tuple(blocks),
            norm=norm,
            lm_head=lm_head,
            config=config,
            extra_embed=extra_embed,
            extra_lm_head=extra_lm_head,
            tied_extra_embed=tied_extra_embed,
        )

    def embed_tokens(self, tokens_t: Array) -> Array:
        if self.extra_embed is None:
            return jax.vmap(self.embed)(tokens_t)
        mask_t = tokens_t >= self.config.vocab_size
        text_embeds_td = jax.vmap(self.embed)(jnp.clip(tokens_t, max=self.config.vocab_size - 1))
        extra_embeds_td = jax.vmap(self.extra_embed)(jnp.clip(tokens_t - self.config.vocab_size, min=0))
        return jnp.where(mask_t[:, None], extra_embeds_td, text_embeds_td)

    def get_logits(self, hidden_td: Array) -> Array:
        extra_head = self.extra_embed if self.tied_extra_embed else self.extra_lm_head
        if extra_head is None:
            return apply_linear(hidden_td, self.lm_head)
        text_logits_tv = apply_linear(hidden_td, self.lm_head)
        extra_logits_tv = apply_linear(hidden_td, extra_head)
        return jnp.concatenate([text_logits_tv, extra_logits_tv], axis=1)

    def init_cache(self, max_len: int | None = None, dtype: jnp.dtype = jnp.bfloat16) -> list[TransformerBlockCache]:
        """Initialize KV cache for autoregressive generation.

        Args:
            max_len: Maximum sequence length to cache.
            dtype: Data type for cache arrays.

        Returns:
            List of TransformerBlockCache dictionaries, one per layer.
        """
        caches = []
        for block in self.blocks:
            caches.append(block.init_cache(max_len=max_len, dtype=dtype))
        return caches

    @overload
    def forward_hidden(
        self,
        tokens_t: Array,
        context_tn: Array | None = None,
    ) -> Array: ...

    @overload
    def forward_hidden(
        self,
        tokens_t: Array,
        context_tn: Array | None = None,
        *,
        caches: list[TransformerBlockCache],
    ) -> tuple[Array, list[TransformerBlockCache]]: ...

    def forward_hidden(
        self,
        tokens_t: Array,
        context_tn: Array | None = None,
        *,
        caches: list[TransformerBlockCache] | None = None,
    ) -> tuple[Array, list[TransformerBlockCache]] | Array:
        """Forward pass returning hidden states shaped (bsz, tsz, embed_dim).

        This is useful for memory-efficient loss computation where you want to
        avoid materializing the full (bsz, tsz, vocab_size) logits tensor.

        Args:
            tokens_t: Tokens, shape (seq_len,).
            context_tn: Contextual embeddings, shape (seq_len, context_dim).
            caches: List of KV caches, one per layer.

        Returns:
            Tuple of (hidden states, updated caches).
        """
        chex.assert_rank(tokens_t, 1)
        x_tn = self.embed_tokens(tokens_t)
        if context_tn is not None:
            x_tn = context_tn + x_tn
        if caches is None:
            for block in self.blocks:
                x_tn, cache = block.forward(x_tn, cache=None)
            return self.norm(x_tn)
        else:
            caches_out: list[TransformerBlockCache] = []
            for block, cache in zip(self.blocks, caches, strict=True):
                x_tn, cache = block.forward(x_tn, cache=cache)
                caches_out.append(cache)
            return self.norm(x_tn), caches_out

    @overload
    def forward(
        self,
        tokens_t: Array,
        *,
        context_tn: Array | None = None,
    ) -> Array: ...

    @overload
    def forward(
        self,
        tokens_t: Array,
        context_tn: Array | None = None,
        *,
        caches: list[TransformerBlockCache],
    ) -> tuple[Array, list[TransformerBlockCache]]: ...

    def forward(
        self,
        tokens_t: Array,
        context_tn: Array | None = None,
        *,
        caches: list[TransformerBlockCache] | None = None,
    ) -> tuple[Array, list[TransformerBlockCache]] | Array:
        if caches is None:
            x_td = self.forward_hidden(
                tokens_t,
                context_tn=context_tn,
            )
            return self.get_logits(x_td)
        else:
            x_td, cache = self.forward_hidden(
                tokens_t,
                context_tn=context_tn,
                caches=caches,
            )
            logits_tv = self.get_logits(x_td)
            return logits_tv, cache

    def get_loss(
        self,
        tokens_t: Array,
        targets_t: Array,
        context_tn: Array | None = None,
        mask_t: Array | None = None,
        chunk_size: int = 1,
    ) -> Array:
        hidden_td = self.forward_hidden(tokens_t, context_tn=context_tn)
        return chunked_cross_entropy_loss(hidden_td, targets_t, self.get_logits, mask_t, chunk_size)

    def get_accuracy(
        self,
        tokens_t: Array,
        targets_t: Array,
        context_tn: Array | None = None,
        mask_t: Array | None = None,
        chunk_size: int = 1,
    ) -> Array:
        hidden_td = self.forward_hidden(tokens_t, context_tn=context_tn)
        return chunked_cross_entropy_acc(hidden_td, targets_t, self.get_logits, mask_t, chunk_size)

    def get_loss_and_accuracy(
        self,
        tokens_t: Array,
        targets_t: Array,
        context_tn: Array | None = None,
        mask_t: Array | None = None,
        chunk_size: int = 1,
    ) -> tuple[Array, Array, Array]:
        hidden_td = self.forward_hidden(tokens_t, context_tn=context_tn)
        loss = chunked_cross_entropy_loss(hidden_td, targets_t, self.get_logits, mask_t, chunk_size)
        accuracy = chunked_cross_entropy_acc(hidden_td, targets_t, self.get_logits, mask_t, chunk_size)
        return loss, accuracy, hidden_td


class CrossAttentionLLM(eqx.Module):
    """LLM with shared cross-attention layer for encoder-decoder style models.

    This wraps a base LLM and adds a single CrossAttentionBlock that is shared
    across all transformer layers. The cross-attention is applied after each
    self-attention block, allowing the decoder to attend to encoder outputs.

    This is useful for tasks like text-to-speech where we want to cross-attend
    to text embeddings while generating audio tokens.
    """

    llm: LLM
    cross_attn: CrossAttentionBlock
    cross_attn_norm: RMSNorm
    config: LLMConfig

    @classmethod
    def build(
        cls,
        config: LLMConfig,
        *,
        key: PRNGKeyArray,
        extra_tokens: int | None = None,
    ) -> "CrossAttentionLLM":
        """Build CrossAttentionLLM with a shared cross-attention layer.

        Args:
            config: LLM configuration
            key: PRNG key for initialization
            extra_tokens: Number of extra tokens to add to the vocabulary.
                If None, no extra tokens are added.

        Returns:
            CrossAttentionLLM with shared cross-attention
        """
        k1, k2 = jax.random.split(key)

        llm = LLM.build(config, extra_tokens=extra_tokens, key=k1)
        cross_attn = CrossAttentionBlock.build(
            embed_dim=config.embed_dim,
            num_heads=config.q_heads,
            key=k2,
            use_rotary_embeddings=False,
        )
        cross_attn_norm = RMSNorm.build(config.embed_dim, eps=config.rms_eps)

        return cls(
            llm=llm,
            cross_attn=cross_attn,
            cross_attn_norm=cross_attn_norm,
            config=config,
        )

    @classmethod
    def from_llm(
        cls,
        llm: LLM,
        *,
        key: PRNGKeyArray,
    ) -> "CrossAttentionLLM":
        """Create CrossAttentionLLM from an existing LLM.

        Args:
            llm: Existing LLM model (can be pretrained)
            key: PRNG key for cross-attention initialization

        Returns:
            CrossAttentionLLM wrapping the provided LLM
        """
        cross_attn = CrossAttentionBlock.build(
            embed_dim=llm.config.embed_dim,
            num_heads=llm.config.q_heads,
            key=key,
            use_rotary_embeddings=False,
        )
        cross_attn_norm = RMSNorm.build(llm.config.embed_dim, eps=llm.config.rms_eps)

        return cls(
            llm=llm,
            cross_attn=cross_attn,
            cross_attn_norm=cross_attn_norm,
            config=llm.config,
        )

    def init_cache(
        self,
        max_len: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        encoder_output_sn: Array | None = None,
    ) -> tuple[list[TransformerBlockCache], AttentionCache | None]:
        """Initialize KV caches for autoregressive generation.

        Args:
            max_len: Maximum sequence length for self-attention cache
            dtype: Data type for cache arrays
            encoder_output_sn: Encoder output for cross-attention cache.
                Shape (encoder_seq_len, embed_dim). If provided, computes
                and caches K/V projections for cross-attention.

        Returns:
            Tuple of (self-attention caches, cross-attention cache)
        """
        self_caches = self.llm.init_cache(max_len=max_len, dtype=dtype)

        cross_cache = None
        if encoder_output_sn is not None:
            cross_cache, _ = self.cross_attn.init_cache(kv_sn=encoder_output_sn)

        return self_caches, cross_cache

    @overload
    def forward_hidden(self, tokens_t: Array, *, context_tn: Array | None = None) -> Array: ...

    @overload
    def forward_hidden(
        self,
        tokens_t: Array,
        *,
        context_tn: Array | None = None,
        encoder_output_sn: Array,
    ) -> Array: ...

    @overload
    def forward_hidden(
        self,
        tokens_t: Array,
        *,
        context_tn: Array | None = None,
        caches: list[TransformerBlockCache],
        cross_cache: AttentionCache,
    ) -> tuple[Array, list[TransformerBlockCache], AttentionCache]: ...

    def forward_hidden(
        self,
        tokens_t: Array,
        *,
        encoder_output_sn: Array | None = None,
        context_tn: Array | None = None,
        caches: list[TransformerBlockCache] | None = None,
        cross_cache: AttentionCache | None = None,
    ) -> tuple[Array, list[TransformerBlockCache], AttentionCache] | Array:
        """Forward pass with cross-attention to encoder output.

        Args:
            tokens_t: Input token IDs, shape (seq_len,)
            encoder_output_sn: Encoder output for cross-attention,
                shape (encoder_seq_len, embed_dim). Required if no cross_cache.
            caches: Self-attention KV caches per layer
            cross_cache: Cross-attention KV cache (precomputed from encoder)

        Returns:
            Hidden states and updated caches if caching, else just hidden states
        """
        chex.assert_rank(tokens_t, 1)
        x_tn = self.llm.embed_tokens(tokens_t)
        if context_tn is not None:
            x_tn = context_tn + x_tn

        if caches is None:
            # No caching - simple forward pass
            for block in self.llm.blocks:
                x_tn, _ = block.forward(x_tn, cache=None)

                # Apply shared cross-attention if encoder output provided
                if encoder_output_sn is not None:
                    norm_x = jax.vmap(self.cross_attn_norm)(x_tn)
                    cross_out, _, _ = self.cross_attn.forward(
                        q_tn=norm_x,
                        kv_sn=encoder_output_sn,
                    )
                    x_tn = x_tn + cross_out

            return self.llm.norm(x_tn)

        else:
            # With caching for autoregressive generation
            caches_out: list[TransformerBlockCache] = []
            updated_cross_cache = cross_cache

            for block, cache in zip(self.llm.blocks, caches, strict=True):
                x_tn, cache = block.forward(x_tn, cache=cache)
                caches_out.append(cache)

                # Apply shared cross-attention using cached K/V
                if cross_cache is not None or encoder_output_sn is not None:
                    norm_x = jax.vmap(self.cross_attn_norm)(x_tn)
                    cross_out, updated_cross_cache, _ = self.cross_attn.forward(
                        q_tn=norm_x,
                        cache=cross_cache,
                        kv_sn=encoder_output_sn if cross_cache is None else None,
                    )
                    x_tn = x_tn + cross_out

            return self.llm.norm(x_tn), caches_out, updated_cross_cache  # type: ignore[return-value]

    @overload
    def forward(
        self,
        tokens_t: Array,
        *,
        context_tn: Array | None = None,
    ) -> Array: ...

    @overload
    def forward(
        self,
        tokens_t: Array,
        *,
        context_tn: Array | None = None,
        encoder_output_sn: Array,
    ) -> Array: ...

    @overload
    def forward(
        self,
        tokens_t: Array,
        *,
        context_tn: Array | None = None,
        caches: list[TransformerBlockCache],
        cross_cache: AttentionCache,
    ) -> tuple[Array, list[TransformerBlockCache], AttentionCache]: ...

    def forward(
        self,
        tokens_t: Array,
        *,
        context_tn: Array | None = None,
        encoder_output_sn: Array | None = None,
        caches: list[TransformerBlockCache] | None = None,
        cross_cache: AttentionCache | None = None,
    ) -> tuple[Array, list[TransformerBlockCache], AttentionCache] | Array:
        """Forward pass returning logits.

        Args:
            tokens_t: Input token IDs, shape (seq_len,)
            context_tn: Contextual embeddings, shape (seq_len, context_dim).
            encoder_output_sn: Encoder output for cross-attention
            caches: Self-attention KV caches per layer
            cross_cache: Cross-attention KV cache

        Returns:
            Logits and updated caches if caching, else just logits
        """
        if caches is None:
            x_td = self.forward_hidden(
                tokens_t,
                context_tn=context_tn,
                encoder_output_sn=encoder_output_sn,
            )
            return apply_linear(x_td, self.llm.lm_head)
        else:
            x_td, caches_out, cross_cache_out = self.forward_hidden(
                tokens_t,
                context_tn=context_tn,
                encoder_output_sn=encoder_output_sn,
                caches=caches,
                cross_cache=cross_cache,
            )
            logits_tv = apply_linear(x_td, self.llm.lm_head)
            return logits_tv, caches_out, cross_cache_out


class LLMRepo(Enum):
    QWEN3_600M = "Qwen/Qwen3-0.6B"
    QWEN3_1_7B = "Qwen/Qwen3-1.7B"
    QWEN3_4B = "Qwen/Qwen3-4B"
    QWEN3_8B = "Qwen/Qwen3-8B"
    QWEN3_14B = "Qwen/Qwen3-14B"
    QWEN3_32B = "Qwen/Qwen3-32B"


@overload
def build_pretrained_llm(
    repo: LLMRepo,
    dtype: jnp.dtype | None = None,
    *,
    extra_tokens: int | None = None,
    tied_extra_embed: bool = False,
    key: PRNGKeyArray | None = None,
) -> LLM: ...


@overload
def build_pretrained_llm(
    repo: LLMRepo,
    dtype: jnp.dtype | None = None,
    *,
    use_cross_attention: Literal[True],
    key: PRNGKeyArray,
    extra_tokens: int | None = None,
    tied_extra_embed: bool = False,
) -> CrossAttentionLLM: ...


def build_pretrained_llm(
    repo: LLMRepo,
    dtype: jnp.dtype | None = None,
    *,
    use_cross_attention: bool = False,
    extra_tokens: int | None = None,
    tied_extra_embed: bool = False,
    key: PRNGKeyArray | None = None,
) -> LLM | CrossAttentionLLM:
    """Loads a pretrained model, optionally with cross-attention support.

    For tensor parallelism, set up a mesh with a 'model' axis before calling:

        mesh = setup_tensor_parallel_mesh()
        jax.set_mesh(mesh)
        model = build_pretrained_model(repo)

    Args:
        repo: Pretrained model repository.
        dtype: Optional dtype for the model.
        use_cross_attention: If True, wraps the LLM with a CrossAttentionLLM
            that adds a shared cross-attention layer. The cross-attention
            weights are randomly initialized.
        extra_tokens: Number of extra tokens to add to the vocabulary. If None,
            no extra tokens are added.
        tied_extra_embed: If True, tie the extra embeddings to the main
            embeddings. If False, use a separate embedding matrix for the extra
            tokens.
        key: PRNG key for initializing the model. If not provided, defaults
            to jax.random.key(0).

    Returns:
        LLM model with loaded weights, optionally wrapped with cross-attention.
    """
    config = hf_config_to_llm_config(cfg=load_hf_config(repo.value))
    if key is None:
        if use_cross_attention or extra_tokens is not None:
            raise ValueError("key is required if use_cross_attention is True or extra_tokens is not None.")
        key = jax.random.key(0)
    model_shape = eqx.filter_eval_shape(
        LLM.build,
        config,
        extra_tokens=extra_tokens,
        tied_extra_embed=tied_extra_embed,
        key=key,
    )
    llm = load_hf_weights_into_llm(model_shape, repo.value, dtype=dtype)

    if use_cross_attention:
        assert key is not None, "key is required if use_cross_attention is True."
        return CrossAttentionLLM.from_llm(llm, key=key)

    return llm


def tie_embedding_and_head(model: LLM) -> LLM:
    """Returns a model with tied embedding and lm_head weights."""
    return eqx.tree_at(lambda mm: mm.lm_head.weight, model, model.embed.weight)


def chunked_cross_entropy_loss(
    hidden_td: Array,
    targets_t: Array,
    lm_head: Callable[[Array], Array],
    mask_t: Array | None = None,
    chunk_size: int = 1,
) -> Array:
    """Compute cross-entropy loss in chunks to save memory.

    Instead of materializing the full (batch, seq, vocab) logits tensor,
    this function processes the sequence in chunks of size `chunk_size`,
    computing logits and loss for each chunk before discarding the logits.

    For a vocab of 150K and sequence of 512:
    - Full logits: 512 * 150000 * 4 bytes = ~300MB per sample
    - Chunked (8): 8 * 150000 * 4 bytes = ~4.8MB per chunk

    Numerical stability:
    - All computations are performed in float32 regardless of input dtype
    - Uses log_softmax which implements the numerically stable log-sum-exp trick
    - Accumulates loss and count in float32 to avoid precision loss

    Args:
        hidden_td: Hidden states from model.forward_hidden(), shape (seq, hidden_dim)
        targets_t: Target token indices, shape (seq,)
        lm_head: The lm_head function, shape (hidden_dim) -> (vocab_size)
        mask_t: Optional mask for valid positions, shape (seq,). If None,
            all positions are valid. "True" indicates that the value is
            included in the loss computation.
        chunk_size: Number of sequence positions to process at once.

    Returns:
        Scalar loss value (mean cross-entropy over valid positions) in float32.
    """
    tsz, hidden_dim = hidden_td.shape

    if mask_t is None:
        mask_t = jnp.ones((tsz), dtype=jnp.bool_)

    # Pad sequence to be divisible by chunk_size for static shapes
    pad_size = (chunk_size - tsz % chunk_size) % chunk_size
    if pad_size > 0:
        hidden_td = jnp.pad(hidden_td, ((0, pad_size), (0, 0)))
        targets_t = jnp.pad(targets_t, ((0, pad_size)))
        mask_t = jnp.pad(mask_t, ((0, pad_size)), constant_values=False)

    padded_tsz = tsz + pad_size
    num_chunks = padded_tsz // chunk_size

    # Reshape to (num_chunks, chunk_size, ...)
    hidden_ccd = hidden_td.reshape(num_chunks, chunk_size, hidden_dim)
    targets_cc = targets_t.reshape(num_chunks, chunk_size)
    mask_cc = mask_t.reshape(num_chunks, chunk_size)

    def process_chunk(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        total_loss, total_count = carry
        chunk_hidden, chunk_targets, chunk_mask = inputs

        # Cast to float32 for numerical stability (inputs may be bfloat16)
        chunk_hidden_f32 = chunk_hidden.astype(jnp.float32)

        # Compute logits: (batch, chunk, hidden) @ (hidden, vocab) -> (batch, chunk, vocab)
        chunk_logits = lm_head(chunk_hidden_f32)

        # log_softmax is numerically stable: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
        log_probs = jax.nn.log_softmax(chunk_logits, axis=-1)

        # Gather log prob of target tokens
        target_log_probs = jnp.take_along_axis(log_probs, chunk_targets[..., None], axis=-1).squeeze(-1)
        chunk_loss = -target_log_probs

        # Mask and accumulate in float32
        masked_loss = jnp.where(chunk_mask, chunk_loss, 0.0)
        total_loss = total_loss + masked_loss.sum()
        total_count = total_count + chunk_mask.astype(jnp.float32).sum()

        return (total_loss, total_count), None

    init_carry = (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
    (total_loss, total_count), _ = jax.lax.scan(
        process_chunk,
        init_carry,
        (hidden_ccd, targets_cc, mask_cc),
    )

    # Safe division - if no valid tokens, return 0
    return jnp.where(total_count > 0, total_loss / total_count, 0.0)


def chunked_cross_entropy_acc(
    hidden_td: Array,
    targets_t: Array,
    lm_head: Callable[[Array], Array],
    mask_t: Array | None = None,
    chunk_size: int = 1,
) -> Array:
    """Compute accuracy in chunks to save memory.

    This is the accuracy counterpart to chunked_cross_entropy_loss. It computes
    the fraction of correctly predicted tokens without materializing full logits.

    Args:
        hidden_td: Hidden states from model.forward_hidden(), shape (seq, hidden_dim)
        targets_t: Target token indices, shape (seq,)
        lm_head: The lm_head function, shape (hidden_dim) -> (vocab_size)
        mask_t: Optional mask for valid positions, shape (seq,). If None, all positions are valid.
        chunk_size: Number of sequence positions to process at once.

    Returns:
        Scalar accuracy value (fraction of correct predictions) in float32.
    """
    tsz, hidden_dim = hidden_td.shape

    if mask_t is None:
        mask_t = jnp.ones((tsz), dtype=jnp.bool_)

    # Pad sequence to be divisible by chunk_size for static shapes
    pad_size = (chunk_size - tsz % chunk_size) % chunk_size
    if pad_size > 0:
        hidden_td = jnp.pad(hidden_td, ((0, pad_size), (0, 0)))
        targets_t = jnp.pad(targets_t, ((0, pad_size)))
        mask_t = jnp.pad(mask_t, ((0, pad_size)), constant_values=False)

    padded_tsz = tsz + pad_size
    num_chunks = padded_tsz // chunk_size

    # Reshape to (num_chunks, chunk_size, ...)
    hidden_ccd = hidden_td.reshape(num_chunks, chunk_size, hidden_dim)
    targets_cc = targets_t.reshape(num_chunks, chunk_size)
    mask_cc = mask_t.reshape(num_chunks, chunk_size)

    def process_chunk(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        total_correct, total_count = carry
        chunk_hidden, chunk_targets, chunk_mask = inputs

        # Cast to float32 for numerical stability (inputs may be bfloat16)
        chunk_hidden_f32 = chunk_hidden.astype(jnp.float32)

        # Compute logits: (chunk, hidden) @ (hidden, vocab) -> (chunk, vocab)
        chunk_logits = lm_head(chunk_hidden_f32)

        # Compute accuracy: check if argmax matches target
        predictions = jnp.argmax(chunk_logits, axis=-1)
        correct = predictions == chunk_targets

        # Mask and accumulate
        masked_correct = jnp.where(chunk_mask, correct.astype(jnp.float32), 0.0)
        total_correct = total_correct + masked_correct.sum()
        total_count = total_count + chunk_mask.astype(jnp.float32).sum()

        return (total_correct, total_count), None

    init_carry = (
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    (total_correct, total_count), _ = jax.lax.scan(
        process_chunk,
        init_carry,
        (hidden_ccd, targets_cc, mask_cc),
    )

    # Safe division - if no valid tokens, return 0
    return jnp.where(total_count > 0, total_correct / total_count, 0.0)


@functools.lru_cache(maxsize=16)
def download_repo(repo_id: str, revision: str | None = None, cache_dir: str | None = None) -> Path:
    """Downloads a repo snapshot from the Huggingface Hub and returns the local path.

    Results are cached in-memory to avoid redundant HuggingFace Hub API calls
    when the same repo is accessed multiple times (e.g., config + weights).
    """
    return Path(snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir))


def load_hf_config(repo_id: str, revision: str | None = None) -> dict[str, object]:
    """Loads HF config.json as dict."""
    path = download_repo(repo_id, revision=revision)
    with open(path / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def setup_model_parallel_mesh(mp_size: int | None = None) -> Mesh:
    """Set up a mesh for model parallelism.

    For model parallelism, we shard the model weights across GPUs along the
    'model' axis. This allows running models that don't fit on a single GPU.

    Args:
        mp_size: Number of GPUs for model parallelism. If None, uses all GPUs.

    Returns:
        A JAX mesh configured for model parallelism.
    """
    devices = jax.local_devices()
    if mp_size is None:
        mp_size = len(devices)
    if mp_size > len(devices):
        raise ValueError(f"mp_size ({mp_size}) > available devices ({len(devices)})")

    # Use the first mp_size devices for model parallelism
    mp_devices = devices[:mp_size]
    mesh = Mesh(np.array(mp_devices), axis_names=("model",))
    return mesh


def get_model_parallel_sharding_spec(
    weight_shape: tuple[int, ...],
    weight_name: str,
    mesh: Mesh,
) -> NamedSharding:
    """Get sharding spec for a weight tensor in tensor parallel mode.

    For transformer models, we use the following sharding strategy:
    - Attention Q/K/V projections: Column parallel (shard output dim)
    - Attention O projection: Row parallel (shard input dim)
    - MLP gate/up projections: Column parallel (shard output dim)
    - MLP down projection: Row parallel (shard input dim)
    - Embeddings and norms: Replicated

    Args:
        weight_shape: Shape of the weight tensor
        weight_name: Name of the weight (used to determine sharding strategy)
        mesh: The tensor parallel mesh

    Returns:
        NamedSharding for the weight
    """
    # Determine sharding based on weight name and shape
    if len(weight_shape) == 1:
        # 1D tensors (biases, norm weights) are replicated
        return NamedSharding(mesh, P())

    # 2D weight matrices
    if len(weight_shape) == 2:
        # Column parallel: shard output dimension (first dim of weight matrix)
        # Used for Q, K, V projections and MLP gate/up projections
        if any(weight_name.endswith(name) for name in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
            return NamedSharding(mesh, P("model", None))

        # Row parallel: shard input dimension (second dim of weight matrix)
        # Used for O projection and MLP down projection
        if any(weight_name.endswith(name) for name in ["o_proj", "down_proj"]):
            return NamedSharding(mesh, P(None, "model"))

        # LM head: column parallel
        if "lm_head" in weight_name:
            return NamedSharding(mesh, P("model", None))

    # Default: replicate
    return NamedSharding(mesh, P())


def _fetch_state_dict(repo_id: str, revision: str | None = None) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Fetch safetensors state dict from HuggingFace repository.

    Loads weights as numpy arrays to avoid unnecessary device transfers.
    The final conversion to JAX arrays with proper sharding happens in load_hf_weights_into_llm.
    """
    snapshot_path = download_repo(repo_id, revision=revision)
    safes = [str(snapshot_path / f) for f in snapshot_path.iterdir() if f.suffix == ".safetensors"]
    if not safes:
        raise FileNotFoundError("No .safetensors files found in snapshot.")

    state: dict[str, np.ndarray] = {}
    for sf in safes:
        # Load directly as numpy - more efficient than jax->numpy conversion
        with safe_open(sf, framework="numpy") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)

    with open(snapshot_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config, state


def _maybe_get(key_sub: str, state: dict[str, np.ndarray]) -> np.ndarray:
    """Find a weight by substring matching."""
    matches: list[str] = [k for k in state if key_sub in k]
    if not matches:
        raise KeyError(f"Missing parameter containing '{key_sub}'")
    if len(matches) > 1:
        matches.sort(key=len)
    return state[matches[0]]


def _get_weight(state: dict[str, np.ndarray], *keys: str) -> np.ndarray:
    """Get weight from state, trying each key in order."""
    for key in keys:
        if any(key in k for k in state):
            return _maybe_get(key, state)
    raise KeyError(f"Missing parameter for keys: {keys}")


def _get_bias(state: dict[str, np.ndarray], *keys: str) -> np.ndarray | None:
    """Get optional bias from state, trying each key in order."""
    for key in keys:
        if key in state:
            return state[key]
    return None


class HFConfig(BaseModel):
    """Subset of HuggingFace config fields needed for LLMConfig derivation."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    hidden_size: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("hidden_size", "n_embd"),
    )
    num_attention_heads: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("num_attention_heads", "n_head"),
    )
    num_key_value_heads: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("num_key_value_heads", "num_kv_heads", "n_kv_head"),
    )
    num_hidden_layers: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("num_hidden_layers", "n_layer"),
    )
    vocab_size: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("vocab_size", "n_vocab"),
    )
    intermediate_size: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("intermediate_size", "n_inner", "ffn_hidden_size"),
    )
    rope_theta: float | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("rope_theta", "rotary_emb_base"),
    )
    rms_norm_eps: float | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("rms_norm_eps", "layer_norm_epsilon"),
    )
    attention_bias: bool | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("attention_bias"),
    )
    mlp_bias: bool | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("mlp_bias"),
    )
    head_dim: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("head_dim"),
    )
    sliding_window: int | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("sliding_window", "sliding_window_size"),
    )
    layer_types: list[str] | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("layer_types", "sliding_attention_map"),
    )
    model_type: str | None = PydanticField(
        default=None,
        validation_alias=AliasChoices("model_type"),
    )


def _parse_rope_scaling(cfg: dict[str, object]) -> tuple[float, int | None, float, float]:
    """Parse YaRN rope_scaling dict from HuggingFace config."""
    rope_scaling_raw = cfg.get("rope_scaling")
    if rope_scaling_raw is None or not isinstance(rope_scaling_raw, dict):
        return 1.0, None, 1.0, 32.0

    # Cast to dict[str, object] for proper typing after isinstance check
    rope_scaling = dict(rope_scaling_raw)

    rope_type = rope_scaling.get("rope_type")
    if rope_type != "yarn":
        return 1.0, None, 1.0, 32.0

    factor_val = rope_scaling.get("factor")
    factor = float(factor_val) if factor_val is not None else 1.0

    original_max_pos_val = rope_scaling.get("original_max_position_embeddings")
    original_max_pos = int(original_max_pos_val) if original_max_pos_val is not None else None

    beta_slow_val = rope_scaling.get("beta_slow")
    beta_slow = float(beta_slow_val) if beta_slow_val is not None else 1.0

    beta_fast_val = rope_scaling.get("beta_fast")
    beta_fast = float(beta_fast_val) if beta_fast_val is not None else 32.0

    return factor, original_max_pos, beta_slow, beta_fast


def hf_config_to_llm_config(cfg: dict[str, object]) -> LLMConfig:
    """Derive an LLMConfig from a HuggingFace config dict.

    Args:
        cfg: HuggingFace config dictionary (from load_hf_config).

    Returns:
        LLMConfig derived from the HuggingFace config.

    Raises:
        ValueError: If required fields are missing from the config.
    """
    parsed = HFConfig.model_validate(cfg)

    # Required fields - raise errors if missing
    if parsed.hidden_size is None:
        raise ValueError("HuggingFace config missing required field: hidden_size")
    if parsed.num_attention_heads is None:
        raise ValueError("HuggingFace config missing required field: num_attention_heads")
    if parsed.num_hidden_layers is None:
        raise ValueError("HuggingFace config missing required field: num_hidden_layers")
    if parsed.vocab_size is None:
        raise ValueError("HuggingFace config missing required field: vocab_size")

    embed_dim = parsed.hidden_size
    q_heads = parsed.num_attention_heads
    kv_heads = parsed.num_key_value_heads or max(1, q_heads // 2)
    head_dim = parsed.head_dim or (embed_dim // q_heads)
    num_layers = parsed.num_hidden_layers
    vocab_size = parsed.vocab_size
    mlp_hidden_dim = parsed.intermediate_size or (embed_dim * 4)
    mlp_mult = max(1, mlp_hidden_dim // embed_dim)
    rope_theta = parsed.rope_theta or 10_000.0
    rms_eps = parsed.rms_norm_eps or 1e-6
    attention_bias = parsed.attention_bias if parsed.attention_bias is not None else False
    mlp_bias = parsed.mlp_bias if parsed.mlp_bias is not None else False

    # YaRN RoPE scaling
    rope_factor, rope_original_max, rope_beta_slow, rope_beta_fast = _parse_rope_scaling(cfg)

    # Sliding window attention
    sliding_window_size = parsed.sliding_window
    layer_attention_types = tuple(parsed.layer_types) if parsed.layer_types else None
    use_attention_sinks = layer_attention_types is not None

    # Qwen3 models use QK normalization
    use_qk_norm = parsed.model_type in ("qwen3", "qwen3_moe")

    return LLMConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        rope_theta=rope_theta,
        dropout_rate=0.0,
        mlp_mult=mlp_mult,
        mlp_hidden_dim=mlp_hidden_dim,
        rms_eps=rms_eps,
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
        rope_factor=rope_factor,
        rope_original_max_position_embeddings=rope_original_max,
        rope_beta_slow=rope_beta_slow,
        rope_beta_fast=rope_beta_fast,
        sliding_window_size=sliding_window_size,
        layer_attention_types=layer_attention_types,
        use_attention_sinks=use_attention_sinks,
        use_qk_norm=use_qk_norm,
    )


def load_hf_weights_into_llm(
    model: LLM,
    repo_id: str,
    revision: str | None = None,
    dtype: jnp.dtype | None = None,
) -> LLM:
    """Load HuggingFace weights into an LLM model.

    This function maps HF causal LM checkpoint weights into the LLM structure.
    Supports models using familiar naming conventions:
    - embeddings: `embed_tokens.weight` or `wte.weight`
    - per-layer projections: `layers.{i}.self_attn.{q,k,v,o}_proj.weight/bias`
    - MLP: `gate_proj`, `up_proj`, `down_proj`
    - Norms: `input_layernorm` and `post_attention_layernorm`
    - Final norm: `norm` or `ln_f`

    Supports tensor parallelism when the mesh has a 'model' axis:
    - Q/K/V projections: column parallel (shard output dim)
    - O projection: row parallel (shard input dim)
    - Gate/Up: column parallel
    - Down: row parallel
    - Embeddings and norms: replicated

    Args:
        model: The LLM model to load weights into.
        repo_id: HuggingFace repository ID.
        revision: Optional revision/branch.
        dtype: Target dtype for weights (default: jnp.bfloat16).

    Returns:
        Model with loaded weights.

    Raises:
        ValueError: If config dimensions don't match.
        KeyError: If required weights are missing.
    """
    if dtype is None:
        dtype = jnp.bfloat16

    mesh = jax.sharding.get_mesh()
    if mesh is None:
        raise ValueError("Mesh is not set. Please set the mesh via jax.set_mesh() before loading weights.")

    config_dict, state = _fetch_state_dict(repo_id, revision)

    # Validate that model config matches the HF config
    hf_parsed = HFConfig.model_validate(config_dict)
    if hf_parsed.hidden_size is not None and hf_parsed.hidden_size != model.config.embed_dim:
        raise ValueError(f"HF hidden_size {hf_parsed.hidden_size} != model.embed_dim {model.config.embed_dim}")

    # Check if we're using model parallelism (mesh has 'model' axis)
    use_mp = "model" in mesh.axis_names

    def make_sharded_array(arr: np.ndarray, weight_name: str) -> jnp.ndarray:
        """Create a JAX array with proper sharding based on weight name."""
        arr_np = np.asarray(arr, dtype=dtype)
        shape = arr_np.shape

        if use_mp:
            sharding = get_model_parallel_sharding_spec(shape, weight_name, mesh)
        else:
            # Replicate across all devices
            sharding = NamedSharding(mesh, P())

        if mesh.devices.size == 1:
            return jnp.asarray(arr_np)

        # For sharded arrays, we need to provide the correct shard for each device
        def callback(idx: tuple[slice, ...] | None) -> np.ndarray:
            # idx tells us which slice this device needs
            if idx is None:
                return arr_np
            return arr_np[idx]

        return jax.make_array_from_callback(shape, sharding, callback)

    def to_dtype_replicated(arr: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """Convert array to target dtype with replicated sharding."""
        return make_sharded_array(np.asarray(arr), "replicated")

    def map_linear(
        eq_lin: eqx.nn.Linear,
        w: np.ndarray | jnp.ndarray,
        b: np.ndarray | jnp.ndarray | None,
        weight_name: str,
    ) -> eqx.nn.Linear:
        """Map weight/bias to linear layer with model parallel sharding."""
        w_mapped = w if w.shape == eq_lin.weight.shape else w.T
        chex.assert_shape(w_mapped, eq_lin.weight.shape)
        w_sharded = make_sharded_array(np.asarray(w_mapped), weight_name)
        eq_lin = eqx.tree_at(lambda lin: lin.weight, eq_lin, w_sharded)
        if b is not None:
            b_sharded = make_sharded_array(np.asarray(b), weight_name + ".bias")
            if eq_lin.bias is not None:
                chex.assert_shape(b, eq_lin.bias.shape)
                eq_lin = eqx.tree_at(lambda lin: lin.bias, eq_lin, b_sharded)
            else:
                eq_lin = eqx.tree_at(lambda lin: lin.bias, eq_lin, b_sharded, is_leaf=lambda x: x is None)
        return eq_lin

    # Load embedding (always replicated)
    embed_w = _get_weight(state, "embed_tokens.weight", "wte.weight")
    model = eqx.tree_at(lambda m: m.embed.weight, model, to_dtype_replicated(embed_w))

    # Load lm_head (may be tied to embeddings) - can be sharded for TP
    has_separate_lm_head = any("lm_head.weight" in k for k in state)
    if has_separate_lm_head:
        lm_head_w = _get_weight(state, "lm_head.weight")
        model = eqx.tree_at(lambda m: m.lm_head.weight, model, make_sharded_array(lm_head_w, "lm_head"))
    elif model.lm_head.weight.shape == embed_w.shape:
        model = tie_embedding_and_head(model)

    # Load per-layer weights
    for idx, block in enumerate(model.blocks):
        pfx = f"layers.{idx}.self_attn"
        pfx_alt = f"model.layers.{idx}.self_attn"
        mlp_pfx = f"layers.{idx}.mlp"
        mlp_pfx_alt = f"model.layers.{idx}.mlp"
        blk_pfx = f"layers.{idx}"
        blk_pfx_alt = f"model.layers.{idx}"

        # Attention projections - use model parallel sharding
        q_w = _get_weight(state, f"{pfx}.q_proj.weight", f"{pfx_alt}.q_proj.weight")
        k_w = _get_weight(state, f"{pfx}.k_proj.weight", f"{pfx_alt}.k_proj.weight")
        v_w = _get_weight(state, f"{pfx}.v_proj.weight", f"{pfx_alt}.v_proj.weight")
        o_w = _get_weight(state, f"{pfx}.o_proj.weight", f"{pfx_alt}.o_proj.weight")
        q_b = _get_bias(state, f"{pfx}.q_proj.bias", f"{pfx_alt}.q_proj.bias")
        k_b = _get_bias(state, f"{pfx}.k_proj.bias", f"{pfx_alt}.k_proj.bias")
        v_b = _get_bias(state, f"{pfx}.v_proj.bias", f"{pfx_alt}.v_proj.bias")
        o_b = _get_bias(state, f"{pfx}.o_proj.bias", f"{pfx_alt}.o_proj.bias")

        block = eqx.tree_at(
            lambda b: b.self_attn.q_proj,
            block,
            map_linear(block.self_attn.q_proj, q_w, q_b, "q_proj"),
        )
        block = eqx.tree_at(
            lambda b: b.self_attn.k_proj,
            block,
            map_linear(block.self_attn.k_proj, k_w, k_b, "k_proj"),
        )
        block = eqx.tree_at(
            lambda b: b.self_attn.v_proj,
            block,
            map_linear(block.self_attn.v_proj, v_w, v_b, "v_proj"),
        )
        block = eqx.tree_at(
            lambda b: b.self_attn.output_proj,
            block,
            map_linear(block.self_attn.output_proj, o_w, o_b, "o_proj"),
        )

        # QK-Norm (Qwen3 style) - only update if weights are present (always replicated)
        q_norm_w = state.get(f"{pfx_alt}.q_norm.weight")
        k_norm_w = state.get(f"{pfx_alt}.k_norm.weight")

        def is_qk_norm_leaf(x: object) -> bool:
            return isinstance(x, RMSNorm) or x is None

        if q_norm_w is not None:
            q_norm = RMSNorm(weight=to_dtype_replicated(q_norm_w), eps=model.config.rms_eps)
            block = eqx.tree_at(lambda b: b.self_attn.q_norm, block, q_norm, is_leaf=is_qk_norm_leaf)
        if k_norm_w is not None:
            k_norm = RMSNorm(weight=to_dtype_replicated(k_norm_w), eps=model.config.rms_eps)
            block = eqx.tree_at(lambda b: b.self_attn.k_norm, block, k_norm, is_leaf=is_qk_norm_leaf)

        # Layer norms (always replicated)
        pre_gamma = _get_weight(
            state,
            f"{blk_pfx}.input_layernorm.weight",
            f"{blk_pfx_alt}.input_layernorm.weight",
        )
        post_gamma = _get_weight(
            state,
            f"{blk_pfx}.post_attention_layernorm.weight",
            f"{blk_pfx_alt}.post_attention_layernorm.weight",
        )
        block = eqx.tree_at(lambda b: b.attn_norm.weight, block, to_dtype_replicated(pre_gamma))
        block = eqx.tree_at(lambda b: b.mlp_norm.weight, block, to_dtype_replicated(post_gamma))

        # MLP projections (SwiGLU) - use model parallel sharding
        gate_w = _get_weight(state, f"{mlp_pfx}.gate_proj.weight", f"{mlp_pfx_alt}.gate_proj.weight")
        up_w = _get_weight(state, f"{mlp_pfx}.up_proj.weight", f"{mlp_pfx_alt}.up_proj.weight")
        down_w = _get_weight(state, f"{mlp_pfx}.down_proj.weight", f"{mlp_pfx_alt}.down_proj.weight")
        gate_b = _get_bias(state, f"{mlp_pfx}.gate_proj.bias", f"{mlp_pfx_alt}.gate_proj.bias")
        up_b = _get_bias(state, f"{mlp_pfx}.up_proj.bias", f"{mlp_pfx_alt}.up_proj.bias")
        down_b = _get_bias(state, f"{mlp_pfx}.down_proj.bias", f"{mlp_pfx_alt}.down_proj.bias")

        block = eqx.tree_at(
            lambda b: b.feed_forward.gate,
            block,
            map_linear(block.feed_forward.gate, gate_w, gate_b, "gate_proj"),
        )
        block = eqx.tree_at(
            lambda b: b.feed_forward.up,
            block,
            map_linear(block.feed_forward.up, up_w, up_b, "up_proj"),
        )
        block = eqx.tree_at(
            lambda b: b.feed_forward.down,
            block,
            map_linear(block.feed_forward.down, down_w, down_b, "down_proj"),
        )

        model = eqx.tree_at(lambda mod, i=idx: mod.blocks[i], model, block)

    # Final norm (always replicated)
    final_gamma = _get_bias(state, "norm.weight", "model.norm.weight", "ln_f.weight")
    if final_gamma is not None:
        model = eqx.tree_at(lambda m: m.norm.weight, model, to_dtype_replicated(final_gamma))

    # Initializes the extra embeddigns to random values.
    if model.extra_embed is not None:
        model = eqx.tree_at(
            lambda m: m.extra_embed.weight,
            model,
            jax.random.normal(jax.random.key(0), model.extra_embed.weight.shape, dtype) * 0.01,
        )
    if model.extra_lm_head is not None:
        model = eqx.tree_at(
            lambda m: m.extra_lm_head.weight,
            model,
            jax.random.normal(jax.random.key(0), model.extra_lm_head.weight.shape, dtype) * 0.01,
        )

    return model


def llm_generate(
    model: LLM,
    tokens: list[int],
    eos_id: int | None,
    max_new_tokens: int = 20,
    *,
    context_tn: Array | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    key: PRNGKeyArray | None = None,
) -> list[int]:
    """Sampling-based decoding for quick sanity checks (non-JIT version)."""
    if key is None:
        key = jax.random.key(0)
    tokens_arr, final_len = xax_filter_jit(llm_generate_jit)(
        model,
        jnp.array(tokens, dtype=jnp.int32),
        eos_id if eos_id is not None else -1,
        max_new_tokens,
        context_tn,
        temperature,
        top_p,
        key,
    )
    return tokens_arr[: int(final_len)].tolist()


@xax_filter_jit(donate="all")
def _sample_next_token(
    logits_tv: Array,
    temperature: float,
    top_p: float,
    key: PRNGKeyArray,
    num_samples: int = 1,
) -> Array:
    """Sample next token from logits using temperature and top-p sampling.

    Args:
        logits_tv: Logits for vocabulary, shape (tsz, vocab_size).
        temperature: Sampling temperature (>0).
        top_p: Top-p (nucleus) sampling probability.
        key: PRNG key for sampling.
        num_samples: The number of samples to draw from the logits.

    Returns:
        Next token, shape (tsz, num_samples).
    """
    logits_tv = logits_tv.astype(jnp.float32)

    # True greedy decoding when temperature is 0
    is_greedy = temperature <= 0
    greedy_tokens = jnp.argmax(logits_tv, axis=-1, keepdims=True)

    # Temperature scaling for sampling
    scaled_logits = jnp.where(temperature > 0, logits_tv / temperature, logits_tv)

    # Top-p nucleus sampling
    sort_idx_tv = jnp.argsort(scaled_logits, axis=-1, descending=True)
    sort_logits_tv = jnp.take_along_axis(scaled_logits, sort_idx_tv, axis=-1)
    sort_probs_tv = jax.nn.softmax(sort_logits_tv, axis=-1)
    cum_probs_tv = jnp.cumsum(sort_probs_tv, axis=-1)
    mask_tv = cum_probs_tv > top_p
    mask_tv = mask_tv.at[..., 0].set(False)
    masked_logits_tv = jnp.where(mask_tv, -jnp.inf, sort_logits_tv)

    shape = masked_logits_tv.shape[:-1] + (num_samples,)
    sampled_idx_tn = jax.random.categorical(key, masked_logits_tv, axis=-1, shape=shape)
    next_token_tn = jnp.take_along_axis(sort_idx_tv, sampled_idx_tn, axis=-1)

    # Use greedy result when temperature <= 0
    return jnp.where(is_greedy, greedy_tokens, next_token_tn)


def llm_generate_jit(
    model: LLM,
    tokens_t: Array,
    eos_id: int,
    max_new_tokens: int,
    context_tn: Array | None,
    temperature: float,
    top_p: float,
    key: PRNGKeyArray,
) -> tuple[Array, Array]:
    """JIT-compiled autoregressive generation with optional context.

    Args:
        model: The LLM model.
        tokens_t: Initial token sequence, shape (initial_len,).
        eos_id: End-of-sequence token ID (-1 to disable).
        max_new_tokens: Maximum number of new tokens to generate.
        context_tn: Optional contextual embeddings, shape (max_len, context_dim).
            Must be padded to max_len = initial_len + max_new_tokens if provided.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling probability.
        key: PRNG key for sampling.

    Returns:
        Tuple of (generated tokens, final sequence length).
    """
    initial_len = tokens_t.shape[-1]
    max_len = initial_len + max_new_tokens

    # Initialize output token buffer
    padded_tokens = jnp.zeros(max_len, dtype=jnp.int32)
    padded_tokens = padded_tokens.at[:initial_len].set(tokens_t)

    dtype = model.embed.weight.dtype
    init_caches = model.init_cache(max_len, dtype)
    init_state = (padded_tokens, jnp.int32(initial_len), key, jnp.bool_(False), init_caches)

    def cond_fn(state: tuple[Array, Array, Array, Array, list[TransformerBlockCache]]) -> Array:
        _, cur_pos, _, done, _ = state
        return (cur_pos < max_len) & ~done

    def body_fn(
        state: tuple[Array, Array, Array, Array, list[TransformerBlockCache]],
    ) -> tuple[Array, Array, Array, Array, list[TransformerBlockCache]]:
        tokens_t, cur_pos, key, _, caches = state

        # Forward pass on full buffer - recompute everything each step
        key, subkey = jax.random.split(key)
        logits_tv, caches = model.forward(tokens_t[:-1], context_tn=context_tn, caches=caches)
        logits = logits_tv[..., cur_pos - 1, :]

        key, subkey = jax.random.split(key)
        next_token = _sample_next_token(logits, temperature, top_p, subkey, num_samples=1)[..., 0]
        new_tokens = tokens_t.at[cur_pos].set(next_token)
        done = jnp.bool_((eos_id >= 0) & (next_token == eos_id))

        return (new_tokens, cur_pos + 1, key, done, caches)

    final_tokens, final_pos, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_tokens, final_pos


def llm_generate_stream(
    model: LLM,
    tokens: list[int],
    eos_id: int | None,
    max_new_tokens: int = 256,
    *,
    context_tn: Array | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    key: PRNGKeyArray | None = None,
) -> Iterator[int]:
    """Streaming token generation that yields tokens one at a time.

    This is a non-JIT generator function that uses KV caching for efficient
    generation. Each token is yielded as soon as it's generated.

    Args:
        model: The LLM model.
        tokens: Initial token sequence as a list of integers.
        eos_id: End-of-sequence token ID (None to disable).
        max_new_tokens: Maximum number of new tokens to generate.
        context_tn: Optional contextual embeddings, shape (max_len, context_dim).
            Must be padded to max_len = len(tokens) + max_new_tokens if provided.
        temperature: Sampling temperature (>0).
        top_p: Top-p (nucleus) sampling probability.
        key: Optional PRNG key for sampling (defaults to key(0)).

    Yields:
        Generated tokens one at a time (does not include input tokens).
    """
    if key is None:
        key = jax.random.key(0)
    tokens_t = jnp.array(tokens, dtype=jnp.int32)

    initial_len = len(tokens)
    max_len = initial_len + max_new_tokens

    # Initialize KV cache using model's dtype
    model_dtype = model.embed.weight.dtype
    caches = model.init_cache(max_len, dtype=model_dtype)

    # Use eqx.filter_jit to properly handle equinox modules without capturing
    # model weights as constants (which would cause OOM for large models).
    @eqx.filter_jit(donate="all-except-first")
    def step(
        model: LLM,
        tokens_t: Array,
        context_tn: Array | None,
        caches: list[TransformerBlockCache],
        key: PRNGKeyArray,
    ) -> tuple[Array, list[TransformerBlockCache], PRNGKeyArray]:
        logits_tv, caches = model.forward(tokens_t, context_tn=context_tn, caches=caches)
        logits_1v = logits_tv[..., -1:, :]
        key, subkey = jax.random.split(key)
        next_token_1 = _sample_next_token(logits_1v, temperature, top_p, subkey, num_samples=1)[..., 0]
        return next_token_1, caches, key

    next_token_1, caches, key = step(model, tokens_t, context_tn, caches, key)
    next_token_int = int(next_token_1.item())

    # Check EOS
    if eos_id is not None and next_token_int == eos_id:
        return

    yield next_token_int

    # Continue generating - context_tn is only used for initial forward pass with caching
    for _ in range(max_new_tokens - 1):
        # Forward with cache - no context needed since it's already incorporated
        next_token_1, caches, key = step(model, next_token_1, None, caches, key)
        next_token_int = int(next_token_1.item())

        # Check EOS
        if eos_id is not None and next_token_int == eos_id:
            return

        yield next_token_int


def main() -> None:
    """Run LLM inference from command line."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install transformers to run the LLM demo: `pip install transformers`") from e

    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run lightweight LLM with HuggingFace weights.")
    parser.add_argument("--repo", type=LLMRepo, required=True, help="LLM repository")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (print all at once).")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode for Qwen3 models.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--mp", type=int, default=None, help="Enable model parallelism with N GPUs (default: auto)")
    args = parser.parse_args()

    # Parse dtype
    dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}
    dtype = dtype_map[args.dtype]

    # Set up tensor parallelism if multiple GPUs are available or explicitly requested
    num_devices = jax.local_device_count()
    if args.mp is not None or num_devices > 1:
        mp_size = args.mp if args.mp is not None else num_devices
        logger.info("Setting up tensor parallelism with %d GPUs", mp_size)
        mesh = setup_model_parallel_mesh(mp_size)
        jax.set_mesh(mesh)
    else:
        # Single device: still need a mesh for weight loading
        mesh = Mesh(np.array(jax.local_devices()[:1]), axis_names=("batch",))
        jax.set_mesh(mesh)

    # Loads the model repository.
    logger.info("Loading weights from %s...", args.repo.value)
    model = build_pretrained_llm(args.repo, dtype=dtype)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.repo.value, revision=args.revision)

    # Prepare input tokens.
    # For Qwen3 models, enable_thinking controls whether the model uses chain-of-thought reasoning.
    # When disabled, the model behaves like Qwen2.5-Instruct without <think>...</think> blocks.
    chat_template_kwargs: dict[str, object] = {"add_generation_prompt": True}
    if args.no_think:
        chat_template_kwargs["enable_thinking"] = False

    tokens = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        **chat_template_kwargs,
    )

    # Generate with streaming or batch mode
    if args.no_stream:
        # Batch mode: generate all then print
        output_tokens = llm_generate(
            model,
            tokens=tokens,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(text)

    else:
        # Streaming mode: print tokens as they're generated
        generated_tokens: list[int] = []
        prev_text = ""
        for token in llm_generate_stream(
            model,
            tokens=tokens,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            generated_tokens.append(token)
            # Decode all tokens and print only the new part
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if len(text) > len(prev_text):
                new_text = text[len(prev_text) :]
                sys.stdout.write(new_text)
                sys.stdout.flush()
                prev_text = text
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
