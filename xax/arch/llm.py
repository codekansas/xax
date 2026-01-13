"""Lightweight JAX LLM reference implementations."""

import functools
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import Array
from omegaconf import MISSING
from pydantic import AliasChoices, BaseModel, ConfigDict, Field as PydanticField

from xax.arch.attention import RotaryEmbedding
from xax.core.conf import field
from xax.nn.lora import LoRALinear

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


# --------------------------------------------------------------------------- #
# Core Modules                                                                 #
# --------------------------------------------------------------------------- #


def _linear(x_btd: Array, linear: eqx.nn.Linear) -> Array:
    """Apply linear layer with einsum for better performance.

    Also supports LoRALinear layers which have weight_oi/bias_o instead of weight/bias.
    """
    # Handle LoRALinear (weight_oi) vs eqx.nn.Linear (weight)
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


@dataclass(frozen=True)
class LLMConfig:
    """Model hyperparameters for decoder-only LLMs."""

    vocab_size: int = field(MISSING, help="Vocabulary size for token embeddings")
    embed_dim: int = field(MISSING, help="Model embedding dimension")
    q_heads: int = field(MISSING, help="Number of query attention heads")
    kv_heads: int = field(MISSING, help="Number of key/value attention heads (for GQA)")
    head_dim: int = field(MISSING, help="Dimension per attention head")
    num_layers: int = field(MISSING, help="Number of transformer layers")
    max_tsz: int = field(MISSING, help="Maximum sequence length")
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
    use_remat: bool = field(False, help="Recompute activations during backward to save memory")

    def with_vocab(self, vocab_size: int) -> "LLMConfig":
        return replace(self, vocab_size=vocab_size)

    @property
    def uses_yarn(self) -> bool:
        return self.rope_factor > 1.0 and self.rope_original_max_position_embeddings is not None


class RMSNorm(eqx.Module):
    """RMSNorm over the last dimension (no bias), matching LLaMA/Qwen style."""

    weight: Array
    eps: float = 1e-6

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.weight = jnp.ones((dim,), dtype=jnp.float32)
        self.eps = eps

    def __call__(self, x_btd: Array) -> Array:
        norm = jnp.sqrt(jnp.mean(jnp.square(x_btd), axis=-1, keepdims=True) + self.eps)
        return (x_btd / norm) * self.weight


class MultiHeadAttention(eqx.Module):
    """Grouped-query attention with rotary embeddings."""

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

        q_bthd = _linear(x_btd, self.q_proj).reshape(bsz, tsz, self.q_heads, self.head_dim)
        k_bthd = _linear(x_btd, self.k_proj).reshape(bsz, tsz, self.kv_heads, self.head_dim)
        v_bthd = _linear(x_btd, self.v_proj).reshape(bsz, tsz, self.kv_heads, self.head_dim)

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

        # Check if cuDNN flash attention can be used for better performance.
        # Requirements:
        # - dtype: fp16, bf16, or fp8
        # - head_dim: <= 128 and multiple of 8
        # - sequence length: multiple of 64 (checked dynamically)
        # - sliding window: only left window (r_window=0) with causal mask
        # - no custom bias (attention sinks disable cuDNN)
        dtype = q_bthd.dtype
        can_use_cudnn = (
            dtype in (jnp.float16, jnp.bfloat16)
            and self.head_dim <= 128
            and self.head_dim % 8 == 0
            and bias is None
            and tsz % 64 == 0
        )
        implementation = "cudnn" if can_use_cudnn else None

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
        return _linear(ctx_btd, self.o_proj)


class FeedForward(eqx.Module):
    """SwiGLU feed-forward layer (LLaMA/Qwen style)."""

    gate: eqx.nn.Linear
    up: eqx.nn.Linear
    down: eqx.nn.Linear

    def __call__(self, x_btd: Array) -> Array:
        chex.assert_rank(x_btd, {2, 3})
        gated_btd = jax.nn.silu(_linear(x_btd, self.gate)) * _linear(x_btd, self.up)
        return _linear(gated_btd, self.down)


class TransformerBlock(eqx.Module):
    """Single transformer layer with pre-norm."""

    attn: MultiHeadAttention
    attn_norm: RMSNorm
    mlp_norm: RMSNorm
    mlp: FeedForward

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


class LLM(eqx.Module):
    """Minimal decoder-only LLM."""

    embed: eqx.nn.Embedding
    blocks: tuple[TransformerBlock, ...]
    norm: RMSNorm
    lm_head: eqx.nn.Linear
    config: LLMConfig

    @classmethod
    def init(cls, config: LLMConfig, *, key: jax.Array) -> "LLM":
        """Initialize LLM with random weights."""
        k_emb, *block_keys, k_head = jax.random.split(key, config.num_layers + 2)
        embed = eqx.nn.Embedding(config.vocab_size, config.embed_dim, key=k_emb)
        blocks = []
        mlp_width = config.mlp_hidden_dim or (config.embed_dim * config.mlp_mult)
        attn_bias = config.attention_bias
        mlp_bias = config.mlp_bias

        for i in range(config.num_layers):
            k_attn, k_mlp = jax.random.split(block_keys[i], 2)
            q_dim = config.q_heads * config.head_dim
            kv_dim = config.kv_heads * config.head_dim

            # Determine sliding window for this layer
            layer_window: int | None = None
            if config.layer_attention_types is not None and i < len(config.layer_attention_types):
                if "sliding" in config.layer_attention_types[i]:
                    layer_window = config.sliding_window_size

            attn = MultiHeadAttention(
                q_proj=eqx.nn.Linear(config.embed_dim, q_dim, use_bias=attn_bias, key=k_attn),
                k_proj=eqx.nn.Linear(config.embed_dim, kv_dim, use_bias=attn_bias, key=k_attn),
                v_proj=eqx.nn.Linear(config.embed_dim, kv_dim, use_bias=attn_bias, key=k_attn),
                o_proj=eqx.nn.Linear(q_dim, config.embed_dim, use_bias=attn_bias, key=k_attn),
                rotary=RotaryEmbedding(
                    config.head_dim,
                    base=config.rope_theta,
                    style="concatenated",
                    factor=config.rope_factor,
                    original_max_position_embeddings=config.rope_original_max_position_embeddings,
                    beta_slow=config.rope_beta_slow,
                    beta_fast=config.rope_beta_fast,
                ),
                q_norm=None,
                k_norm=None,
                q_heads=config.q_heads,
                kv_heads=config.kv_heads,
                head_dim=config.head_dim,
                dropout_rate=config.dropout_rate,
                sliding_window_size=layer_window,
            )

            mlp = FeedForward(
                gate=eqx.nn.Linear(config.embed_dim, mlp_width, use_bias=mlp_bias, key=k_mlp),
                up=eqx.nn.Linear(config.embed_dim, mlp_width, use_bias=mlp_bias, key=k_mlp),
                down=eqx.nn.Linear(mlp_width, config.embed_dim, use_bias=mlp_bias, key=k_mlp),
            )

            blocks.append(
                TransformerBlock(
                    attn=attn,
                    attn_norm=RMSNorm(config.embed_dim, eps=config.rms_eps),
                    mlp_norm=RMSNorm(config.embed_dim, eps=config.rms_eps),
                    mlp=mlp,
                )
            )

        norm = RMSNorm(config.embed_dim, eps=config.rms_eps)
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
        logits_btv = _linear(x_btd, self.lm_head)
        return logits_btv


def build_qwen3_model(config: LLMConfig, *, key: jax.Array | None = None) -> LLM:
    """Creates a Qwen3-style model.

    Args:
        config: LLMConfig specifying model architecture.
        key: Optional PRNG key for initialization.

    Returns:
        Initialized LLM model.
    """
    key = jax.random.key(0) if key is None else key
    return LLM.init(config, key=key)


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


def hf_config_to_llm_config(
    cfg: dict[str, object],
    *,
    use_remat: bool = False,
    max_tsz: int = 32768,
) -> LLMConfig:
    """Derive an LLMConfig from a HuggingFace config dict.

    Args:
        cfg: HuggingFace config dictionary (from load_hf_config).
        use_remat: Whether to use gradient checkpointing (remat).
        max_tsz: Maximum sequence length (not always in HF config).

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

    return LLMConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        max_tsz=max_tsz,
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
        use_remat=use_remat,
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

    config_dict, state = _fetch_state_dict(repo_id, revision)

    # Validate that model config matches the HF config
    hf_parsed = HFConfig.model_validate(config_dict)
    if hf_parsed.hidden_size is not None and hf_parsed.hidden_size != model.config.embed_dim:
        raise ValueError(f"HF hidden_size {hf_parsed.hidden_size} != model.embed_dim {model.config.embed_dim}")

    mesh = jax.sharding.get_mesh()
    if mesh is None:
        raise ValueError("Mesh is not set. Please set the mesh via jax.set_mesh() before loading weights.")

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

        block = eqx.tree_at(lambda b: b.attn.q_proj, block, map_linear(block.attn.q_proj, q_w, q_b))
        block = eqx.tree_at(lambda b: b.attn.k_proj, block, map_linear(block.attn.k_proj, k_w, k_b))
        block = eqx.tree_at(lambda b: b.attn.v_proj, block, map_linear(block.attn.v_proj, v_w, v_b))
        block = eqx.tree_at(lambda b: b.attn.o_proj, block, map_linear(block.attn.o_proj, o_w, o_b))

        # QK-Norm (Qwen3 style) - only create if weights are present
        q_norm_w = state.get(f"{pfx_alt}.q_norm.weight")
        k_norm_w = state.get(f"{pfx_alt}.k_norm.weight")
        if q_norm_w is not None:
            q_norm = RMSNorm(q_norm_w.shape[-1], eps=model.config.rms_eps)
            q_norm = eqx.tree_at(lambda n: n.weight, q_norm, to_dtype(q_norm_w))
            block = eqx.tree_at(lambda b: b.attn.q_norm, block, q_norm, is_leaf=lambda x: x is None)
        if k_norm_w is not None:
            k_norm = RMSNorm(k_norm_w.shape[-1], eps=model.config.rms_eps)
            k_norm = eqx.tree_at(lambda n: n.weight, k_norm, to_dtype(k_norm_w))
            block = eqx.tree_at(lambda b: b.attn.k_norm, block, k_norm, is_leaf=lambda x: x is None)

        # Layer norms
        pre_gamma = _get_weight(state, f"{blk_pfx}.input_layernorm.weight", f"{blk_pfx_alt}.input_layernorm.weight")
        post_gamma = _get_weight(
            state, f"{blk_pfx}.post_attention_layernorm.weight", f"{blk_pfx_alt}.post_attention_layernorm.weight"
        )
        block = eqx.tree_at(lambda b: b.attn_norm.weight, block, to_dtype(pre_gamma))
        block = eqx.tree_at(lambda b: b.mlp_norm.weight, block, to_dtype(post_gamma))

        # MLP projections
        gate_w = _get_weight(state, f"{mlp_pfx}.gate_proj.weight", f"{mlp_pfx_alt}.gate_proj.weight")
        up_w = _get_weight(state, f"{mlp_pfx}.up_proj.weight", f"{mlp_pfx_alt}.up_proj.weight")
        down_w = _get_weight(state, f"{mlp_pfx}.down_proj.weight", f"{mlp_pfx_alt}.down_proj.weight")
        gate_b = _get_bias(state, f"{mlp_pfx}.gate_proj.bias", f"{mlp_pfx_alt}.gate_proj.bias")
        up_b = _get_bias(state, f"{mlp_pfx}.up_proj.bias", f"{mlp_pfx_alt}.up_proj.bias")
        down_b = _get_bias(state, f"{mlp_pfx}.down_proj.bias", f"{mlp_pfx_alt}.down_proj.bias")

        block = eqx.tree_at(lambda b: b.mlp.gate, block, map_linear(block.mlp.gate, gate_w, gate_b))
        block = eqx.tree_at(lambda b: b.mlp.up, block, map_linear(block.mlp.up, up_w, up_b))
        block = eqx.tree_at(lambda b: b.mlp.down, block, map_linear(block.mlp.down, down_w, down_b))

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
    tokens_arr, final_len = llm_generate_jit(
        model,
        jnp.array(tokens, dtype=jnp.int32),
        eos_id if eos_id is not None else -1,
        max_new_tokens,
        temperature,
        top_p,
        jax.random.key(0),
    )
    # Trim to actual generated length (excluding padding after EOS)
    return tokens_arr[: int(final_len)].tolist()


def llm_generate_jit(
    model: LLM,
    tokens_t: Array,
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    key: Array,
) -> tuple[Array, Array]:
    """JIT-compilable sampling-based decoding.

    Args:
        model: The LLM model.
        tokens_t: Initial token sequence, shape (seq_len,).
        eos_id: End-of-sequence token ID (-1 to disable).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (>0).
        top_p: Top-p (nucleus) sampling probability.
        key: PRNG key for sampling.

    Returns:
        Tuple of (tokens, final_length):
        - tokens: Generated token sequence including input tokens, shape (seq_len + max_new_tokens,).
        - final_length: Number of valid tokens (excluding padding after EOS).
    """
    # Pad tokens to max possible length
    initial_len = tokens_t.shape[0]
    max_len = initial_len + max_new_tokens
    padded_tokens = jnp.zeros(max_len, dtype=jnp.int32)
    padded_tokens = padded_tokens.at[:initial_len].set(tokens_t)

    # State: (tokens, current_position, key, done)
    # current_position is where we're generating the next token
    init_state = (padded_tokens, jnp.int32(initial_len), key, jnp.bool_(False))

    def cond_fn(state: tuple[Array, Array, Array, Array]) -> Array:
        _, cur_pos, _, done = state
        return (cur_pos < max_len) & ~done

    def body_fn(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        tokens, cur_pos, key, _ = state

        # Forward pass on full buffer - model handles padding implicitly
        # We pass all tokens and read logits at position (cur_pos - 1)
        tokens_bt = tokens[None, :]  # Shape: (1, max_len)
        logits_btv = model(tokens_bt, key=key, inference=True)

        # Get logits at the last valid position (cur_pos - 1)
        # Use dynamic indexing which is JIT-compatible
        logits = logits_btv[0, cur_pos - 1, :].astype(jnp.float32)

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

        # Update tokens at current position
        new_tokens = tokens.at[cur_pos].set(next_token)

        # Check for EOS (use -1 to disable)
        done = jnp.bool_((eos_id >= 0) & (next_token == eos_id))

        return (new_tokens, cur_pos + 1, key, done)

    final_tokens, final_pos, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # Return tokens and the final valid length
    return final_tokens, final_pos


def main() -> None:
    """Run LLM inference from command line."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install transformers to run the LLM demo: `pip install transformers`") from e

    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run lightweight LLM with HuggingFace weights.")
    parser.add_argument("--repo", type=str, required=True, help="HF repo id, e.g., Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--chat-template", action="store_true", help="Apply chat template for chat models.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    # Parse dtype
    dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}
    dtype = dtype_map[args.dtype]

    # Load config and build model
    cfg_dict = load_hf_config(args.repo, revision=args.revision)
    config = hf_config_to_llm_config(cfg_dict)
    model = build_qwen3_model(config)

    # Load weights
    logger.info("Loading weights from %s...", args.repo)
    model = load_hf_weights_into_llm(model, args.repo, revision=args.revision, dtype=dtype)
    logger.info("Weights loaded successfully!")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)

    # Prepare input tokens
    if args.chat_template or ("chat" in args.repo.lower()):
        tokens = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt},
            ],
            add_generation_prompt=True,
        )
    else:
        tokens = tokenizer.encode(args.prompt, return_tensors="np", add_special_tokens=False)[0].tolist()

    # Generate
    output_tokens = llm_generate(
        model,
        tokens=tokens,
        eos_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Print result
    text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
