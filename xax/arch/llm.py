"""Lightweight JAX LLM reference implementations."""

import functools
import json
import logging
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterator

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import Array
from omegaconf import MISSING
from pydantic import AliasChoices, BaseModel, ConfigDict, Field as PydanticField

from xax.arch.attention import (
    AttentionCache,
    RMSNorm,
    SelfAttentionBlock,
    SwiGLU,
    TransformerBlock,
    apply_linear,
)
from xax.core.conf import field

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Please install huggingface_hub to access pre-trained LLM weights: `pip install huggingface-hub`"
    ) from e

try:
    from safetensors import safe_open
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Please install safetensors to access pre-trained LLM weights: `pip install safetensors`"
    ) from e

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
    use_remat: bool = field(True, help="Recompute activations during backward to save memory")
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
    config: LLMConfig

    @classmethod
    def build(cls, config: LLMConfig, *, key: jax.Array) -> "LLM":
        """Initialize LLM with random weights."""
        k_emb, *block_keys, k_head = jax.random.split(key, config.num_layers + 2)
        embed = eqx.nn.Embedding(config.vocab_size, config.embed_dim, key=k_emb)
        blocks: list[TransformerBlock] = []
        mlp_width = config.mlp_hidden_dim or (config.embed_dim * config.mlp_mult)
        attn_bias = config.attention_bias
        mlp_bias = config.mlp_bias

        for i in range(config.num_layers):
            k_attn, k_mlp = jax.random.split(block_keys[i], 2)

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
        lm_head = eqx.nn.Linear(config.embed_dim, config.vocab_size, use_bias=False, key=k_head)
        return cls(embed=embed, blocks=tuple(blocks), norm=norm, lm_head=lm_head, config=config)

    def forward_hidden(
        self,
        tokens_bt: Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        """Forward pass returning hidden states shaped (bsz, tsz, embed_dim).

        This is useful for memory-efficient loss computation where you want to
        avoid materializing the full (bsz, tsz, vocab_size) logits tensor.
        """
        chex.assert_rank(tokens_bt, 2)
        bsz, tsz = tokens_bt.shape
        positions_bt = jnp.broadcast_to(jnp.arange(tsz)[None, :], (bsz, tsz))

        key_seq = None
        if key is not None:
            key_seq = jax.random.split(key, len(self.blocks))

        x_btd = jnp.take(self.embed.weight, tokens_bt, axis=0)
        for i, block in enumerate(self.blocks):
            block_key = None if key_seq is None else key_seq[i]
            if self.config.use_remat:
                block_arrays, block_static = eqx.partition(block, eqx.is_array)

                @jax.checkpoint
                def remat_block(
                    arrays: TransformerBlock,
                    x: Array,
                    pos: Array,
                    k: jax.Array | None,
                    inf: bool,
                    *,
                    static: TransformerBlock = block_static,
                ) -> Array:
                    b: TransformerBlock = eqx.combine(arrays, static)
                    return b(x, pos, key=k, inference=inf)

                x_btd = remat_block(block_arrays, x_btd, positions_bt, block_key, inference)
            else:
                x_btd = block(x_btd, positions_bt, key=block_key, inference=inference)
        return self.norm(x_btd)

    def __call__(
        self,
        tokens_bt: Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        """Forward pass returning logits shaped (bsz, tsz, vocab_size)."""
        x_btd = self.forward_hidden(tokens_bt, key=key, inference=inference)
        logits_btv = apply_linear(x_btd, self.lm_head)
        return logits_btv

    def init_cache(self, max_seq_len: int, dtype: jnp.dtype = jnp.bfloat16) -> list[AttentionCache]:
        """Initialize KV cache for autoregressive generation.

        Args:
            max_seq_len: Maximum sequence length to cache.
            dtype: Data type for cache arrays.

        Returns:
            List of AttentionCache dictionaries, one per layer.
        """
        caches = []
        for block in self.blocks:
            k_cache = jnp.zeros(
                (max_seq_len, block.self_attn.num_kv_heads, block.self_attn.head_dim),
                dtype=dtype,
            )
            v_cache = jnp.zeros(
                (max_seq_len, block.self_attn.num_kv_heads, block.self_attn.head_dim),
                dtype=dtype,
            )
            caches.append({"k": k_cache, "v": v_cache, "position": 0})
        return caches

    def forward_with_cache(
        self,
        token: Array,
        caches: list[AttentionCache],
        position: Array | int,
    ) -> tuple[Array, list[AttentionCache]]:
        """Forward pass for a single token using KV cache.

        Args:
            token: Single token, shape () or (1,).
            caches: List of KV caches, one per layer.
            position: Current position in the sequence (can be traced).

        Returns:
            Tuple of (logits for next token, updated caches).
        """
        # Ensure token is scalar
        token = token.reshape(())
        position = jnp.asarray(position, dtype=jnp.int32)

        # Embed token - add batch dim for module compatibility
        x_1d = self.embed.weight[token][None, :]  # Shape: (1, embed_dim)

        # Get max cache length for masking (static)
        max_cache_len = caches[0]["k"].shape[0]

        updated_caches = []
        for block, cache in zip(self.blocks, caches, strict=True):
            attn = block.self_attn

            # Pre-norm for attention (modules expect 2D input)
            normed_1d = block.attn_norm(x_1d)

            # Project to Q, K, V from normed input
            q_1qd = apply_linear(normed_1d, attn.q_proj)  # (1, q_heads * head_dim)
            k_1kd = apply_linear(normed_1d, attn.k_proj)  # (1, kv_heads * head_dim)
            v_1kd = apply_linear(normed_1d, attn.v_proj)  # (1, kv_heads * head_dim)

            # Reshape to (num_heads, head_dim) - drop batch dim
            q_hd = q_1qd[0].reshape(attn.num_heads, attn.head_dim)
            k_hd = k_1kd[0].reshape(attn.num_kv_heads, attn.head_dim)
            v_hd = v_1kd[0].reshape(attn.num_kv_heads, attn.head_dim)

            # QK-Norm (expects 2D input)
            if attn.q_norm is not None:
                q_hd = attn.q_norm(q_hd[None, :])[0]
            if attn.k_norm is not None:
                k_hd = attn.k_norm(k_hd[None, :])[0]

            # RoPE - use dynamic position
            if attn.rotary_emb is not None:
                pos_arr = position.reshape(1)
                q_hd = attn.rotary_emb.apply_rotary_embeddings(q_hd[None, :, :], positions=pos_arr)[0]
                k_hd = attn.rotary_emb.apply_rotary_embeddings(k_hd[None, :, :], positions=pos_arr)[0]

            # Update cache at current position
            k_cache = cache["k"].at[position].set(k_hd)
            v_cache = cache["v"].at[position].set(v_hd)
            updated_caches.append({"k": k_cache, "v": v_cache, "position": position + 1})

            # Attention computation over full cache with masking
            num_groups = attn.num_heads // attn.num_kv_heads

            # GQA: expand Q to match KV head structure
            # q_hd: (num_heads, head_dim) -> (kv_heads, groups, head_dim)
            q_gkd = q_hd.reshape(attn.num_kv_heads, num_groups, attn.head_dim)

            # Use full cache and mask invalid positions
            # k_cache: (max_len, kv_heads, head_dim)
            # Compute scores over all positions
            scores = jnp.einsum("kgd,skd->kgs", q_gkd, k_cache)
            scores = scores / jnp.sqrt(attn.head_dim).astype(scores.dtype)

            # Create mask for valid positions (0 to position inclusive)
            # positions > current position should be masked out
            pos_indices = jnp.arange(max_cache_len)
            mask = pos_indices <= position  # Shape: (max_len,)
            mask = mask[None, None, :]  # Shape: (1, 1, max_len) for broadcasting

            # Apply mask: set invalid positions to -inf before softmax
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

            # Softmax
            attn_weights = jax.nn.softmax(scores, axis=-1)

            # Context: (kv_heads, groups, head_dim)
            ctx_kgd = jnp.einsum("kgs,skd->kgd", attn_weights, v_cache)
            ctx_d = ctx_kgd.reshape(-1)

            # Output projection and residual (add back batch dim)
            attn_out = apply_linear(ctx_d[None, :], attn.output_proj)  # (1, embed_dim)
            x_1d = x_1d + attn_out

            # MLP with pre-norm (modules expect 2D input)
            mlp_normed = block.mlp_norm(x_1d)
            mlp_out = block.feed_forward(mlp_normed)
            x_1d = x_1d + mlp_out

        # Final norm and lm_head
        x_1d = self.norm(x_1d)
        logits_1v = apply_linear(x_1d, self.lm_head)

        # Remove batch dimension for output
        return logits_1v[0], updated_caches


class LLMRepo(Enum):
    QWEN3_600M = "Qwen/Qwen3-0.6B"
    QWEN3_1_5B = "Qwen/Qwen3-1.5B"
    QWEN3_3B = "Qwen/Qwen3-3B"
    QWEN3_7B = "Qwen/Qwen3-7B"
    QWEN3_14B = "Qwen/Qwen3-14B"
    QWEN3_32B = "Qwen/Qwen3-32B"


def build_pretrained_model(repo: LLMRepo, dtype: jnp.dtype | None = None) -> LLM:
    """Loads a pretrained model.

    Args:
        repo: Pretrained model repository.
        dtype: Optional dtype for the model.
    """
    config = hf_config_to_llm_config(cfg=load_hf_config(repo.value))
    model = LLM.build(config, key=jax.random.key(0))
    return load_hf_weights_into_llm(model, repo.value, dtype=dtype)


def tie_embedding_and_head(model: LLM) -> LLM:
    """Returns a model with tied embedding and lm_head weights."""
    return eqx.tree_at(lambda mm: mm.lm_head.weight, model, model.embed.weight)


def chunked_cross_entropy_loss(
    hidden_btd: Array,
    targets_bt: Array,
    lm_head_weight: Array,
    mask_bt: Array | None = None,
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
        hidden_btd: Hidden states from model.forward_hidden(), shape (batch, seq, hidden_dim)
        targets_bt: Target token indices, shape (batch, seq)
        lm_head_weight: The lm_head weight matrix, shape (vocab_size, hidden_dim)
        mask_bt: Optional mask for valid positions, shape (batch, seq). If None, all positions are valid.
        chunk_size: Number of sequence positions to process at once.

    Returns:
        Scalar loss value (mean cross-entropy over valid positions) in float32.
    """
    bsz, tsz, hidden_dim = hidden_btd.shape

    if mask_bt is None:
        mask_bt = jnp.ones((bsz, tsz), dtype=jnp.bool_)

    # Pad sequence to be divisible by chunk_size for static shapes
    pad_size = (chunk_size - tsz % chunk_size) % chunk_size
    if pad_size > 0:
        hidden_btd = jnp.pad(hidden_btd, ((0, 0), (0, pad_size), (0, 0)))
        targets_bt = jnp.pad(targets_bt, ((0, 0), (0, pad_size)))
        mask_bt = jnp.pad(mask_bt, ((0, 0), (0, pad_size)), constant_values=False)

    padded_tsz = tsz + pad_size
    num_chunks = padded_tsz // chunk_size

    # Reshape to (batch, num_chunks, chunk_size, ...)
    hidden_bccd = hidden_btd.reshape(bsz, num_chunks, chunk_size, hidden_dim)
    targets_bcc = targets_bt.reshape(bsz, num_chunks, chunk_size)
    mask_bcc = mask_bt.reshape(bsz, num_chunks, chunk_size)

    def process_chunk(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        total_loss, total_count = carry
        chunk_hidden, chunk_targets, chunk_mask = inputs

        # Cast to float32 for numerical stability (inputs may be bfloat16)
        chunk_hidden_f32 = chunk_hidden.astype(jnp.float32)
        lm_head_f32 = lm_head_weight.astype(jnp.float32)

        # Compute logits: (batch, chunk, hidden) @ (hidden, vocab) -> (batch, chunk, vocab)
        chunk_logits = chunk_hidden_f32 @ lm_head_f32.T

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

    # Transpose to (num_chunks, batch, chunk_size, ...) for scan
    hidden_cbcd = jnp.transpose(hidden_bccd, (1, 0, 2, 3))
    targets_cbc = jnp.transpose(targets_bcc, (1, 0, 2))
    mask_cbc = jnp.transpose(mask_bcc, (1, 0, 2))

    init_carry = (
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    (total_loss, total_count), _ = jax.lax.scan(
        process_chunk,
        init_carry,
        (hidden_cbcd, targets_cbc, mask_cbc),
    )

    # Safe division - if no valid tokens, return 0
    return jnp.where(total_count > 0, total_loss / total_count, 0.0)


def chunked_cross_entropy_acc(
    hidden_btd: Array,
    targets_bt: Array,
    lm_head_weight: Array,
    mask_bt: Array | None = None,
    chunk_size: int = 1,
) -> Array:
    """Compute accuracy in chunks to save memory.

    This is the accuracy counterpart to chunked_cross_entropy_loss. It computes
    the fraction of correctly predicted tokens without materializing full logits.

    Args:
        hidden_btd: Hidden states from model.forward_hidden(), shape (batch, seq, hidden_dim)
        targets_bt: Target token indices, shape (batch, seq)
        lm_head_weight: The lm_head weight matrix, shape (vocab_size, hidden_dim)
        mask_bt: Optional mask for valid positions, shape (batch, seq). If None, all positions are valid.
        chunk_size: Number of sequence positions to process at once.

    Returns:
        Scalar accuracy value (fraction of correct predictions) in float32.
    """
    bsz, tsz, hidden_dim = hidden_btd.shape

    if mask_bt is None:
        mask_bt = jnp.ones((bsz, tsz), dtype=jnp.bool_)

    # Pad sequence to be divisible by chunk_size for static shapes
    pad_size = (chunk_size - tsz % chunk_size) % chunk_size
    if pad_size > 0:
        hidden_btd = jnp.pad(hidden_btd, ((0, 0), (0, pad_size), (0, 0)))
        targets_bt = jnp.pad(targets_bt, ((0, 0), (0, pad_size)))
        mask_bt = jnp.pad(mask_bt, ((0, 0), (0, pad_size)), constant_values=False)

    padded_tsz = tsz + pad_size
    num_chunks = padded_tsz // chunk_size

    # Reshape to (batch, num_chunks, chunk_size, ...)
    hidden_bccd = hidden_btd.reshape(bsz, num_chunks, chunk_size, hidden_dim)
    targets_bcc = targets_bt.reshape(bsz, num_chunks, chunk_size)
    mask_bcc = mask_bt.reshape(bsz, num_chunks, chunk_size)

    def process_chunk(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        total_correct, total_count = carry
        chunk_hidden, chunk_targets, chunk_mask = inputs

        # Cast to float32 for numerical stability (inputs may be bfloat16)
        chunk_hidden_f32 = chunk_hidden.astype(jnp.float32)
        lm_head_f32 = lm_head_weight.astype(jnp.float32)

        # Compute logits: (batch, chunk, hidden) @ (hidden, vocab) -> (batch, chunk, vocab)
        chunk_logits = chunk_hidden_f32 @ lm_head_f32.T

        # Compute accuracy: check if argmax matches target
        predictions = jnp.argmax(chunk_logits, axis=-1)
        correct = predictions == chunk_targets

        # Mask and accumulate
        masked_correct = jnp.where(chunk_mask, correct.astype(jnp.float32), 0.0)
        total_correct = total_correct + masked_correct.sum()
        total_count = total_count + chunk_mask.astype(jnp.float32).sum()

        return (total_correct, total_count), None

    # Transpose to (num_chunks, batch, chunk_size, ...) for scan
    hidden_cbcd = jnp.transpose(hidden_bccd, (1, 0, 2, 3))
    targets_cbc = jnp.transpose(targets_bcc, (1, 0, 2))
    mask_cbc = jnp.transpose(mask_bcc, (1, 0, 2))

    init_carry = (
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    (total_correct, total_count), _ = jax.lax.scan(
        process_chunk,
        init_carry,
        (hidden_cbcd, targets_cbc, mask_cbc),
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

    if mesh.devices.size > 1:
        # Replicate weights across all devices (no partitioning)
        sharding = NamedSharding(mesh, P())

        def to_dtype(arr: np.ndarray | jnp.ndarray) -> jnp.ndarray:
            # Create JAX array with proper sharding in one step
            # Using jax.make_array_from_callback ensures proper replication
            shape = arr.shape
            arr_np = np.asarray(arr, dtype=dtype)

            def callback(idx: tuple[slice, ...] | None) -> np.ndarray:
                # Each device gets the full array (replicated)
                return arr_np

            return jax.make_array_from_callback(shape, sharding, callback)

    else:

        def to_dtype(arr: np.ndarray | jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(arr, dtype=dtype)

    def map_linear(
        eq_lin: eqx.nn.Linear,
        w: np.ndarray | jnp.ndarray,
        b: np.ndarray | jnp.ndarray | None,
    ) -> eqx.nn.Linear:
        """Map weight/bias to linear layer, handling shape transposition if needed."""
        w_mapped = w if w.shape == eq_lin.weight.shape else w.T
        chex.assert_shape(w_mapped, eq_lin.weight.shape)
        eq_lin = eqx.tree_at(lambda lin: lin.weight, eq_lin, to_dtype(w_mapped))
        if b is not None:
            if eq_lin.bias is not None:
                chex.assert_shape(b, eq_lin.bias.shape)
                eq_lin = eqx.tree_at(lambda lin: lin.bias, eq_lin, to_dtype(b))
            else:
                eq_lin = eqx.tree_at(lambda lin: lin.bias, eq_lin, to_dtype(b), is_leaf=lambda x: x is None)
        return eq_lin

    # Load embedding
    embed_w = _get_weight(state, "embed_tokens.weight", "wte.weight")
    model = eqx.tree_at(lambda m: m.embed.weight, model, to_dtype(embed_w))

    # Load lm_head (may be tied to embeddings)
    has_separate_lm_head = any("lm_head.weight" in k for k in state)
    if has_separate_lm_head:
        lm_head_w = _get_weight(state, "lm_head.weight")
        model = eqx.tree_at(lambda m: m.lm_head.weight, model, to_dtype(lm_head_w))
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

        # Attention projections
        q_w = _get_weight(state, f"{pfx}.q_proj.weight", f"{pfx_alt}.q_proj.weight")
        k_w = _get_weight(state, f"{pfx}.k_proj.weight", f"{pfx_alt}.k_proj.weight")
        v_w = _get_weight(state, f"{pfx}.v_proj.weight", f"{pfx_alt}.v_proj.weight")
        o_w = _get_weight(state, f"{pfx}.o_proj.weight", f"{pfx_alt}.o_proj.weight")
        q_b = _get_bias(state, f"{pfx}.q_proj.bias", f"{pfx_alt}.q_proj.bias")
        k_b = _get_bias(state, f"{pfx}.k_proj.bias", f"{pfx_alt}.k_proj.bias")
        v_b = _get_bias(state, f"{pfx}.v_proj.bias", f"{pfx_alt}.v_proj.bias")
        o_b = _get_bias(state, f"{pfx}.o_proj.bias", f"{pfx_alt}.o_proj.bias")

        block = eqx.tree_at(lambda b: b.self_attn.q_proj, block, map_linear(block.self_attn.q_proj, q_w, q_b))
        block = eqx.tree_at(lambda b: b.self_attn.k_proj, block, map_linear(block.self_attn.k_proj, k_w, k_b))
        block = eqx.tree_at(lambda b: b.self_attn.v_proj, block, map_linear(block.self_attn.v_proj, v_w, v_b))
        block = eqx.tree_at(lambda b: b.self_attn.output_proj, block, map_linear(block.self_attn.output_proj, o_w, o_b))

        # QK-Norm (Qwen3 style) - only update if weights are present
        q_norm_w = state.get(f"{pfx_alt}.q_norm.weight")
        k_norm_w = state.get(f"{pfx_alt}.k_norm.weight")

        def is_qk_norm_leaf(x: object) -> bool:
            return isinstance(x, RMSNorm) or x is None

        if q_norm_w is not None:
            q_norm = RMSNorm(weight=to_dtype(q_norm_w), eps=model.config.rms_eps)
            block = eqx.tree_at(lambda b: b.self_attn.q_norm, block, q_norm, is_leaf=is_qk_norm_leaf)
        if k_norm_w is not None:
            k_norm = RMSNorm(weight=to_dtype(k_norm_w), eps=model.config.rms_eps)
            block = eqx.tree_at(lambda b: b.self_attn.k_norm, block, k_norm, is_leaf=is_qk_norm_leaf)

        # Layer norms
        pre_gamma = _get_weight(state, f"{blk_pfx}.input_layernorm.weight", f"{blk_pfx_alt}.input_layernorm.weight")
        post_gamma = _get_weight(
            state, f"{blk_pfx}.post_attention_layernorm.weight", f"{blk_pfx_alt}.post_attention_layernorm.weight"
        )
        block = eqx.tree_at(lambda b: b.attn_norm.weight, block, to_dtype(pre_gamma))
        block = eqx.tree_at(lambda b: b.mlp_norm.weight, block, to_dtype(post_gamma))

        # MLP projections (SwiGLU)
        gate_w = _get_weight(state, f"{mlp_pfx}.gate_proj.weight", f"{mlp_pfx_alt}.gate_proj.weight")
        up_w = _get_weight(state, f"{mlp_pfx}.up_proj.weight", f"{mlp_pfx_alt}.up_proj.weight")
        down_w = _get_weight(state, f"{mlp_pfx}.down_proj.weight", f"{mlp_pfx_alt}.down_proj.weight")
        gate_b = _get_bias(state, f"{mlp_pfx}.gate_proj.bias", f"{mlp_pfx_alt}.gate_proj.bias")
        up_b = _get_bias(state, f"{mlp_pfx}.up_proj.bias", f"{mlp_pfx_alt}.up_proj.bias")
        down_b = _get_bias(state, f"{mlp_pfx}.down_proj.bias", f"{mlp_pfx_alt}.down_proj.bias")

        block = eqx.tree_at(lambda b: b.feed_forward.gate, block, map_linear(block.feed_forward.gate, gate_w, gate_b))
        block = eqx.tree_at(lambda b: b.feed_forward.up, block, map_linear(block.feed_forward.up, up_w, up_b))
        block = eqx.tree_at(lambda b: b.feed_forward.down, block, map_linear(block.feed_forward.down, down_w, down_b))

        model = eqx.tree_at(lambda mod, i=idx: mod.blocks[i], model, block)

    # Final norm
    final_gamma = _get_bias(state, "norm.weight", "model.norm.weight", "ln_f.weight")
    if final_gamma is not None:
        model = eqx.tree_at(lambda m: m.norm.weight, model, to_dtype(final_gamma))

    return model


def llm_generate(
    model: LLM,
    tokens: list[int],
    eos_id: int | None,
    max_new_tokens: int = 20,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[int]:
    """Sampling-based decoding for quick sanity checks (non-JIT version)."""
    result = llm_generate_jit(
        model,
        jnp.array(tokens, dtype=jnp.int32),
        eos_id if eos_id is not None else -1,
        max_new_tokens,
        temperature,
        top_p,
        jax.random.key(0),
        use_cache=True,
        return_cache=False,
    )
    tokens_arr, final_len = result[0], result[1]
    # Trim to actual generated length (excluding padding after EOS)
    return tokens_arr[: int(final_len)].tolist()


def _sample_next_token(
    logits_v: Array,
    temperature: float,
    top_p: float,
    key: Array,
) -> tuple[Array, Array]:
    """Sample next token from logits using temperature and top-p sampling.

    Args:
        logits_v: Logits for vocabulary, shape (vocab_size,).
        temperature: Sampling temperature (>0).
        top_p: Top-p (nucleus) sampling probability.
        key: PRNG key for sampling.

    Returns:
        Tuple of (next_token, new_key).
    """
    logits = logits_v.astype(jnp.float32)

    # Temperature scaling
    logits = jnp.where(temperature > 0, logits / temperature, logits)

    # Top-p nucleus sampling
    sort_idx = jnp.argsort(logits)[::-1]
    sort_logits = logits[sort_idx]
    sort_probs = jax.nn.softmax(sort_logits)
    cum_probs = jnp.cumsum(sort_probs)
    mask = cum_probs > top_p
    mask = mask.at[0].set(False)
    masked_logits = jnp.where(mask, -jnp.inf, sort_logits)

    key, subkey = jax.random.split(key)
    sampled_idx = jax.random.categorical(subkey, masked_logits)
    next_token = sort_idx[sampled_idx]

    return next_token, key


def llm_generate_jit(
    model: LLM,
    tokens_t: Array,
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    key: Array,
    *,
    use_cache: bool = True,
    return_cache: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, list[AttentionCache]]:
    """JIT-compilable sampling-based decoding.

    Args:
        model: The LLM model.
        tokens_t: Initial token sequence, shape (seq_len,).
        eos_id: End-of-sequence token ID (-1 to disable).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (>0).
        top_p: Top-p (nucleus) sampling probability.
        key: PRNG key for sampling.
        use_cache: If True (default), use KV caching for O(n) generation.
            If False, recompute full sequence each step (O(n²) but simpler).
        return_cache: If True, return the KV cache along with tokens.
            Only valid when use_cache=True.

    Returns:
        If return_cache=False: Tuple of (tokens, final_length)
        If return_cache=True: Tuple of (tokens, final_length, caches)
        - tokens: Generated token sequence including input tokens, shape (seq_len + max_new_tokens,).
        - final_length: Number of valid tokens (excluding padding after EOS).
        - caches: List of KV caches per layer (only if return_cache=True).
    """
    initial_len = tokens_t.shape[0]
    max_len = initial_len + max_new_tokens

    # Initialize output token buffer
    padded_tokens = jnp.zeros(max_len, dtype=jnp.int32)
    padded_tokens = padded_tokens.at[:initial_len].set(tokens_t)

    if use_cache:
        # Cached generation: O(n) complexity
        return _llm_generate_jit_cached(
            model,
            padded_tokens,
            tokens_t,
            initial_len,
            max_len,
            eos_id,
            temperature,
            top_p,
            key,
            return_cache,
        )
    else:
        # Non-cached generation: O(n²) complexity, simpler implementation
        if return_cache:
            raise ValueError("return_cache=True requires use_cache=True")
        return _llm_generate_jit_no_cache(
            model,
            padded_tokens,
            initial_len,
            max_len,
            eos_id,
            temperature,
            top_p,
            key,
        )


def _llm_generate_jit_cached(
    model: LLM,
    padded_tokens: Array,
    tokens_t: Array,
    initial_len: int,
    max_len: int,
    eos_id: int,
    temperature: float,
    top_p: float,
    key: Array,
    return_cache: bool,
) -> tuple[Array, Array] | tuple[Array, Array, list[AttentionCache]]:
    """KV-cached generation implementation."""
    # Initialize KV cache using model's dtype
    model_dtype = model.embed.weight.dtype
    caches = model.init_cache(max_len, dtype=model_dtype)

    # Prefill: process the initial prompt to fill the cache
    vocab_size = model.config.vocab_size

    def prefill_step(
        carry: tuple[Array, list[AttentionCache], Array],
        token: Array,
    ) -> tuple[tuple[Array, list[AttentionCache], Array], None]:
        pos, caches, _ = carry
        logits, new_caches = model.forward_with_cache(token, caches, pos)
        return (pos + 1, new_caches, logits), None

    # Dummy initial logits - match model's dtype
    dummy_logits = jnp.zeros(vocab_size, dtype=model_dtype)
    (_, caches, logits_v), _ = jax.lax.scan(
        prefill_step,
        (jnp.int32(0), caches, dummy_logits),
        tokens_t,
    )

    # Sample first token
    first_token, key = _sample_next_token(logits_v, temperature, top_p, key)
    padded_tokens = padded_tokens.at[initial_len].set(first_token)
    first_done = jnp.bool_((eos_id >= 0) & (first_token == eos_id))

    # State for while loop
    init_state = (padded_tokens, jnp.int32(initial_len + 1), caches, key, first_done)

    def cond_fn(state: tuple[Array, Array, list[AttentionCache], Array, Array]) -> Array:
        _, cur_pos, _, _, done = state
        return (cur_pos < max_len) & ~done

    def body_fn(
        state: tuple[Array, Array, list[AttentionCache], Array, Array],
    ) -> tuple[Array, Array, list[AttentionCache], Array, Array]:
        tokens, cur_pos, caches, key, _ = state
        prev_token = tokens[cur_pos - 1]
        logits_v, new_caches = model.forward_with_cache(prev_token, caches, cur_pos - 1)
        next_token, new_key = _sample_next_token(logits_v, temperature, top_p, key)
        new_tokens = tokens.at[cur_pos].set(next_token)
        done = jnp.bool_((eos_id >= 0) & (next_token == eos_id))
        return (new_tokens, cur_pos + 1, new_caches, new_key, done)

    final_tokens, final_pos, final_caches, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    if return_cache:
        return final_tokens, final_pos, final_caches
    return final_tokens, final_pos


def _llm_generate_jit_no_cache(
    model: LLM,
    padded_tokens: Array,
    initial_len: int,
    max_len: int,
    eos_id: int,
    temperature: float,
    top_p: float,
    key: Array,
) -> tuple[Array, Array]:
    """Non-cached generation implementation (O(n²) but simpler)."""
    init_state = (padded_tokens, jnp.int32(initial_len), key, jnp.bool_(False))

    def cond_fn(state: tuple[Array, Array, Array, Array]) -> Array:
        _, cur_pos, _, done = state
        return (cur_pos < max_len) & ~done

    def body_fn(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        tokens, cur_pos, key, _ = state

        # Forward pass on full buffer - recompute everything each step
        tokens_bt = tokens[None, :]
        logits_btv = model(tokens_bt, key=key, inference=True)
        logits = logits_btv[0, cur_pos - 1, :]

        next_token, new_key = _sample_next_token(logits, temperature, top_p, key)
        new_tokens = tokens.at[cur_pos].set(next_token)
        done = jnp.bool_((eos_id >= 0) & (next_token == eos_id))

        return (new_tokens, cur_pos + 1, new_key, done)

    final_tokens, final_pos, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_tokens, final_pos


def llm_generate_stream(
    model: LLM,
    tokens: list[int],
    eos_id: int | None,
    max_new_tokens: int = 256,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    key: Array | None = None,
) -> Iterator[int]:
    """Streaming token generation that yields tokens one at a time.

    This is a non-JIT generator function that uses KV caching for efficient
    generation. Each token is yielded as soon as it's generated.

    Args:
        model: The LLM model.
        tokens: Initial token sequence as a list of integers.
        eos_id: End-of-sequence token ID (None to disable).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (>0).
        top_p: Top-p (nucleus) sampling probability.
        key: Optional PRNG key for sampling (defaults to key(0)).

    Yields:
        Generated tokens one at a time (does not include input tokens).
    """
    if key is None:
        key = jax.random.key(0)

    initial_len = len(tokens)
    max_len = initial_len + max_new_tokens

    # Initialize KV cache using model's dtype
    model_dtype = model.embed.weight.dtype
    caches = model.init_cache(max_len, dtype=model_dtype)

    # Prefill: process the initial prompt to fill the cache
    for pos, token in enumerate(tokens):
        token_arr = jnp.array(token, dtype=jnp.int32)
        logits_v, caches = model.forward_with_cache(token_arr, caches, pos)

    # logits_v now contains logits for predicting the next token after the prompt
    # Sample first token
    next_token, key = _sample_next_token(logits_v, temperature, top_p, key)
    next_token_int = int(next_token)

    # Check EOS
    if eos_id is not None and next_token_int == eos_id:
        return

    yield next_token_int

    # Continue generating
    cur_pos = initial_len
    for _ in range(max_new_tokens - 1):
        # Forward with cache
        logits_v, caches = model.forward_with_cache(next_token, caches, cur_pos)
        cur_pos += 1

        # Sample next token
        next_token, key = _sample_next_token(logits_v, temperature, top_p, key)
        next_token_int = int(next_token)

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
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (print all at once).")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode for Qwen3 models.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    # Parse dtype
    dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}
    dtype = dtype_map[args.dtype]

    # Loads the model repository.
    logger.info("Loading weights from %s...", args.repo.value)
    model = build_pretrained_model(args.repo, dtype=dtype)

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
        print()  # Final newline


if __name__ == "__main__":
    main()
