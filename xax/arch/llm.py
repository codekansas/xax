"""Lightweight JAX LLM reference implementations (Qwen3, GPT-OSS, DeepSeek-R1).

These models are adapted from the `jax-ml/jax-llm-examples` reference
implementations but simplified to match the `xax` style and dependency set.

Key differences:

* Uses Equinox modules and standard JAX primitives (no Pallas kernels).
* Keeps grouped-query attention (separate q_heads/kv_heads) and rotary
  embeddings to stay architecturally faithful while remaining lightweight.
* Provides small default configs for CPU tests; full-size configs can be added
  by users when loading real checkpoints.
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from pydantic import AliasChoices, BaseModel, ConfigDict, Field as PydanticField
from safetensors.numpy import load_file as safe_load

from xax.arch.attention import RotaryEmbedding

try:
    from huggingface_hub import snapshot_download

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Please install huggingface_hub to access pre-trained LLM weights: `pip install huggingface-hub`"
    ) from e

logger = logging.getLogger(__name__)


def _linear(x_btd: Array, linear: eqx.nn.Linear) -> Array:
    y_bto = jnp.einsum("...d,od->...o", x_btd, linear.weight)
    if linear.bias is not None:
        y_bto = y_bto + linear.bias
    return y_bto


@dataclass(frozen=True)
class LLMConfig:
    """Model hyperparameters."""

    vocab_size: int
    embed_dim: int
    q_heads: int
    kv_heads: int
    head_dim: int
    num_layers: int
    max_tsz: int
    rope_theta: float = 10_000.0
    dropout_rate: float = 0.0
    mlp_mult: int = 4
    mlp_hidden_dim: int | None = None
    rms_eps: float = 1e-6
    attention_bias: bool = False  # Whether attention projections have bias
    mlp_bias: bool = False  # Whether MLP projections have bias

    def with_vocab(self, vocab_size: int) -> "LLMConfig":
        return replace(self, vocab_size=vocab_size)


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
            # Apply RMSNorm per head: (bsz, tsz, heads, head_dim) -> normalize over head_dim
            q_bthd = self.q_norm(q_bthd)
        if self.k_norm is not None:
            k_bthd = self.k_norm(k_bthd)

        positions_flat = positions_bt.reshape(-1)
        q_flat = q_bthd.reshape(-1, self.q_heads, self.head_dim)
        k_flat = k_bthd.reshape(-1, self.kv_heads, self.head_dim)

        q_flat = self.rotary.apply_rotary_embeddings(q_flat, positions=positions_flat)
        k_flat = self.rotary.apply_rotary_embeddings(k_flat, positions=positions_flat)

        q_bthd = q_flat.reshape(bsz, tsz, self.q_heads, self.head_dim)
        k_bthd = k_flat.reshape(bsz, tsz, self.kv_heads, self.head_dim)

        # Broadcast kv heads to match q heads (grouped-query attention).
        repeat_factor = self.q_heads // self.kv_heads
        k_bthd = jnp.repeat(k_bthd, repeat_factor, axis=2)
        v_bthd = jnp.repeat(v_bthd, repeat_factor, axis=2)

        attn_logits_bhtt = jnp.einsum("bthd,bThd->bhtT", q_bthd, k_bthd) / jnp.sqrt(self.head_dim)
        causal_mask_tt = jnp.tril(jnp.ones((tsz, tsz), dtype=bool))
        attn_logits_bhtt = jnp.where(causal_mask_tt[None, None, :, :], attn_logits_bhtt, -1e9)
        attn_weights_bhtt = jax.nn.softmax(attn_logits_bhtt, axis=-1)
        if self.dropout_rate > 0.0 and not inference:
            assert key is not None, "Dropout requires PRNG key when not in inference mode."
            attn_weights_bhtt = eqx.nn.Dropout(self.dropout_rate)(attn_weights_bhtt, key=key)
        ctx_bthd = jnp.einsum("bhtT,bThd->bthd", attn_weights_bhtt, v_bthd)
        ctx_btd = ctx_bthd.reshape(bsz, tsz, self.q_heads * self.head_dim)
        return _linear(ctx_btd, self.o_proj)


class FeedForward(eqx.Module):
    gate: eqx.nn.Linear
    up: eqx.nn.Linear
    down: eqx.nn.Linear

    def __call__(self, x_btd: Array) -> Array:
        chex.assert_rank(x_btd, {2, 3})  # (tsz, dim) or (bsz, tsz, dim)
        gated_btd = jax.nn.silu(_linear(x_btd, self.gate)) * _linear(x_btd, self.up)
        return _linear(gated_btd, self.down)


class TransformerBlock(eqx.Module):
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
        attn_key, mlp_key = (None, None)
        if key is not None:
            attn_key, mlp_key = jax.random.split(key, 2)
        y_btd = x_btd + self.attn(self.attn_norm(x_btd), positions_bt, key=attn_key, inference=inference)
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
            attn = MultiHeadAttention(
                q_proj=eqx.nn.Linear(config.embed_dim, q_dim, use_bias=attn_bias, key=k_attn),
                k_proj=eqx.nn.Linear(config.embed_dim, kv_dim, use_bias=attn_bias, key=k_attn),
                v_proj=eqx.nn.Linear(config.embed_dim, kv_dim, use_bias=attn_bias, key=k_attn),
                o_proj=eqx.nn.Linear(q_dim, config.embed_dim, use_bias=attn_bias, key=k_attn),
                rotary=RotaryEmbedding(config.head_dim, base=config.rope_theta, style="concatenated"),
                q_norm=None,  # QK-Norm only created if weights are present
                k_norm=None,  # QK-Norm only created if weights are present
                q_heads=config.q_heads,
                kv_heads=config.kv_heads,
                head_dim=config.head_dim,
                dropout_rate=config.dropout_rate,
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

    def __call__(
        self,
        tokens_bt: Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        """Returns logits shaped (bsz, tsz, vocab_size)."""
        chex.assert_rank(tokens_bt, 2)
        bsz, tsz = tokens_bt.shape
        positions_bt = jnp.broadcast_to(jnp.arange(tsz)[None, :], (bsz, tsz))

        key_seq = None
        if key is not None:
            key_seq = jax.random.split(key, len(self.blocks))

        x_btd = jnp.take(self.embed.weight, tokens_bt, axis=0)
        for i, block in enumerate(self.blocks):
            block_key = None if key_seq is None else key_seq[i]
            x_btd = block(x_btd, positions_bt, key=block_key, inference=inference)
        x_btd = self.norm(x_btd)
        logits_btv = _linear(x_btd, self.lm_head)
        return logits_btv


# Small configs for quick unit tests / demos.
QWEN3_SMALL = LLMConfig(
    vocab_size=32000,
    embed_dim=256,
    q_heads=8,
    kv_heads=4,
    head_dim=32,
    num_layers=4,
    max_tsz=512,
)

GPT_OSS_SMALL = LLMConfig(
    vocab_size=32000,
    embed_dim=384,
    q_heads=12,
    kv_heads=4,
    head_dim=32,
    num_layers=6,
    max_tsz=512,
)

DEEPSEEK_R1_SMALL = LLMConfig(
    vocab_size=32000,
    embed_dim=512,
    q_heads=16,
    kv_heads=8,
    head_dim=32,
    num_layers=8,
    max_tsz=512,
)


def build_qwen3_model(config: LLMConfig = QWEN3_SMALL, *, key: jax.Array | None = None) -> LLM:
    """Creates a Qwen3-style model."""
    key = jax.random.key(0) if key is None else key
    return LLM.init(config, key=key)


def build_gpt_oss_model(config: LLMConfig = GPT_OSS_SMALL, *, key: jax.Array | None = None) -> LLM:
    """Creates a GPT-OSS-style model."""
    key = jax.random.key(1) if key is None else key
    return LLM.init(config, key=key)


def build_deepseek_r1_model(config: LLMConfig = DEEPSEEK_R1_SMALL, *, key: jax.Array | None = None) -> LLM:
    """Creates a DeepSeek-R1-style model."""
    key = jax.random.key(2) if key is None else key
    return LLM.init(config, key=key)


def tie_embedding_and_head(model: LLM) -> LLM:
    """Returns a model with tied embedding and lm_head weights."""
    return eqx.tree_at(lambda mm: mm.lm_head.weight, model, model.embed.weight)


# --------------------------------------------------------------------------- #
# HuggingFace download/convert helpers                                        #
# --------------------------------------------------------------------------- #


def download_repo(repo_id: str, revision: str | None = None, cache_dir: str | None = None) -> Path:
    """Downloads a repo snapshot from the Huggingface Hub and returns the local path."""
    return Path(snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir))


def load_hf_config(repo_id: str, revision: str | None = None) -> dict[str, object]:
    """Loads HF config.json as dict."""
    path = download_repo(repo_id, revision=revision)
    with open(path / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_state_dict(repo_id: str, revision: str | None = None) -> tuple[dict[str, object], dict[str, jnp.ndarray]]:
    snapshot_path = download_repo(repo_id, revision=revision)
    safes = [os.path.join(snapshot_path, f) for f in os.listdir(snapshot_path) if f.endswith(".safetensors")]
    if not safes:
        raise FileNotFoundError("No .safetensors files found in snapshot.")
    state: dict[str, jnp.ndarray] = {}
    for sf in safes:
        loaded_states: dict[str, np.ndarray] = safe_load(sf)
        state.update({k: jnp.asarray(v) for k, v in loaded_states.items()})
    config = json.loads(open(f"{snapshot_path}/config.json", "r", encoding="utf-8").read())
    return config, state


def _maybe_get(key_sub: str, state: dict[str, jnp.ndarray]) -> jnp.ndarray:
    matches: list[str] = [k for k in state if key_sub in k]
    if not matches:
        raise KeyError(f"Missing parameter containing '{key_sub}'")
    if len(matches) > 1:
        matches.sort(key=len)
    return state[matches[0]]


def _get_weight(state: dict[str, jnp.ndarray], *keys: str) -> jnp.ndarray:
    """Get weight from state, trying each key in order."""
    for key in keys:
        if any(key in k for k in state):
            return _maybe_get(key, state)
    raise KeyError(f"Missing parameter for keys: {keys}")


def _get_bias(state: dict[str, jnp.ndarray], *keys: str) -> jnp.ndarray | None:
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


def hf_config_to_llm_config(cfg: dict[str, object], base: LLMConfig | None = None) -> LLMConfig:
    """Derive an LLMConfig from an HF config dict."""
    base = base or QWEN3_SMALL
    parsed = HFConfig.model_validate(cfg)
    embed_dim = parsed.hidden_size or base.embed_dim
    q_heads = parsed.num_attention_heads or base.q_heads
    kv_heads = parsed.num_key_value_heads or max(1, q_heads // 2)
    head_dim = embed_dim // q_heads
    num_layers = parsed.num_hidden_layers or base.num_layers
    vocab_size = parsed.vocab_size or base.vocab_size
    mlp_hidden_dim = parsed.intermediate_size or (embed_dim * base.mlp_mult)
    mlp_mult = max(1, mlp_hidden_dim // embed_dim)
    rope_theta = parsed.rope_theta or base.rope_theta
    rms_eps = parsed.rms_norm_eps or base.rms_eps
    # Default to False for biases if not specified (most LLaMA-style models don't have biases)
    attention_bias = parsed.attention_bias if parsed.attention_bias is not None else base.attention_bias
    mlp_bias = parsed.mlp_bias if parsed.mlp_bias is not None else base.mlp_bias
    return LLMConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        max_tsz=base.max_tsz,
        rope_theta=rope_theta,
        dropout_rate=base.dropout_rate,
        mlp_mult=mlp_mult,
        mlp_hidden_dim=mlp_hidden_dim,
        rms_eps=rms_eps,
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
    )


def load_hf_weights_into_llm(model: LLM, repo_id: str, revision: str | None = None) -> LLM:
    """Attempts to map a HF causal LM checkpoint into the lightweight LLM structure.

    This assumes the HF model uses familiar naming:
    - embeddings: `embed_tokens.weight` or `wte.weight`
    - per-layer projections: `layers.{i}.self_attn.{q,k,v,o}_proj.weight/bias`
      or GPT-style `h.{i}.attn.c_attn`/`c_proj`
    - MLP: `gate_proj`, `up_proj`, `down_proj` or GPT `mlp.c_fc`/`c_proj`
    - Norms: `input_layernorm` and `post_attention_layernorm` (or GPT `ln_1`, `ln_2`)
    - Final norm: `norm` or `ln_f`

    Shapes must match the model config; otherwise an error is raised.
    """
    config_dict, state = _fetch_state_dict(repo_id, revision)
    derived_cfg = hf_config_to_llm_config(config_dict, base=model.config)
    if derived_cfg.embed_dim != model.config.embed_dim:
        raise ValueError(f"HF hidden_size {derived_cfg.embed_dim} != model.embed_dim {model.config.embed_dim}")

    def to_f32(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(arr, dtype=jnp.float32)

    def map_linear(eq_lin: eqx.nn.Linear, w: jnp.ndarray, b: jnp.ndarray | None) -> eqx.nn.Linear:
        w_mapped = w if w.shape == eq_lin.weight.shape else w.T
        chex.assert_shape(w_mapped, eq_lin.weight.shape)
        eq_lin = eqx.tree_at(lambda lin: lin.weight, eq_lin, to_f32(w_mapped))
        if b is not None:
            # If weights have bias, we need to add it even if model didn't have one
            if eq_lin.bias is not None:
                chex.assert_shape(b, eq_lin.bias.shape)
                eq_lin = eqx.tree_at(lambda lin: lin.bias, eq_lin, to_f32(b))
            else:
                # Create new Linear with bias
                eq_lin = eqx.tree_at(lambda lin: lin.bias, eq_lin, to_f32(b), is_leaf=lambda x: x is None)
        return eq_lin

    # Embedding + LM head
    embed_w = _get_weight(state, "embed_tokens.weight", "wte.weight")
    model = eqx.tree_at(lambda m: m.embed.weight, model, to_f32(embed_w))

    # Check if model has separate lm_head weights (not tied to embeddings)
    has_separate_lm_head = any("lm_head.weight" in k for k in state)
    if has_separate_lm_head:
        lm_head_w = _get_weight(state, "lm_head.weight")
        model = eqx.tree_at(lambda m: m.lm_head.weight, model, to_f32(lm_head_w))
    elif model.lm_head.weight.shape == embed_w.shape:
        # Tie embedding and lm_head only if no separate lm_head exists
        model = tie_embedding_and_head(model)

    # Per-layer mappings
    for idx, block in enumerate(model.blocks):
        pfx = f"layers.{idx}.self_attn"
        pfx_alt = f"model.layers.{idx}.self_attn"
        mlp_pfx = f"layers.{idx}.mlp"

        # Attention projections (q, k, v, o)
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
            q_norm = RMSNorm(q_norm_w.shape[-1], eps=derived_cfg.rms_eps)
            q_norm = eqx.tree_at(lambda n: n.weight, q_norm, to_f32(q_norm_w))
            block = eqx.tree_at(lambda b: b.attn.q_norm, block, q_norm)
        if k_norm_w is not None:
            k_norm = RMSNorm(k_norm_w.shape[-1], eps=derived_cfg.rms_eps)
            k_norm = eqx.tree_at(lambda n: n.weight, k_norm, to_f32(k_norm_w))
            block = eqx.tree_at(lambda b: b.attn.k_norm, block, k_norm)

        # Layer norms (at block level, not under self_attn)
        blk_pfx = f"layers.{idx}"
        blk_pfx_alt = f"model.layers.{idx}"
        pre_gamma = _get_weight(state, f"{blk_pfx}.input_layernorm.weight", f"{blk_pfx_alt}.input_layernorm.weight")
        post_gamma = _get_weight(
            state, f"{blk_pfx}.post_attention_layernorm.weight", f"{blk_pfx_alt}.post_attention_layernorm.weight"
        )
        block = eqx.tree_at(lambda b: b.attn_norm.weight, block, to_f32(pre_gamma))
        block = eqx.tree_at(lambda b: b.mlp_norm.weight, block, to_f32(post_gamma))

        # MLP projections (gate, up, down)
        mlp_pfx_alt = f"model.layers.{idx}.mlp"
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
        model = eqx.tree_at(lambda m: m.norm.weight, model, to_f32(final_gamma))

    return model


def llm_generate(
    model: LLM,
    tokens: list[int],
    eos_id: int,
    max_new_tokens: int = 20,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[int]:
    """Sampling-based decoding for quick sanity checks."""
    tokens_bt = jnp.array(tokens, dtype=jnp.int32)[None, :]
    key = jax.random.key(0)
    for _ in range(max_new_tokens):
        logits_btv = model(tokens_bt, key=key, inference=True)
        logits = logits_btv[:, -1, :]
        if temperature > 0:
            logits = logits / temperature
        # top-p nucleus sampling
        sort_idx = jnp.argsort(logits, axis=-1)[:, ::-1]
        sort_logits = jnp.take_along_axis(logits, sort_idx, axis=-1)
        sort_probs = jax.nn.softmax(sort_logits, axis=-1)
        cum_probs = jnp.cumsum(sort_probs, axis=-1)
        mask = cum_probs > top_p
        mask = mask.at[:, 0].set(False)
        masked_logits = jnp.where(mask, -jnp.inf, sort_logits)
        key, subkey = jax.random.split(key)
        sampled_idx = jax.random.categorical(subkey, masked_logits, axis=-1)
        next_token = jnp.take_along_axis(sort_idx, sampled_idx[:, None], axis=-1)
        tokens_bt = jnp.concatenate([tokens_bt, next_token], axis=1)
        if eos_id is not None and int(next_token[0, 0]) == eos_id:
            break
    return jnp.array(tokens_bt[0]).tolist()


def quantize_state_dict(state: dict[str, jnp.ndarray], num_bits: int = 8) -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    """Symmetric per-tensor quantization: returns mapping to (quantized_int, scale)."""
    if num_bits not in (8, 4):
        raise ValueError("num_bits must be 4 or 8")
    max_q = (2 ** (num_bits - 1)) - 1
    quantized: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}
    for k, v in state.items():
        if jnp.issubdtype(v.dtype, jnp.integer):
            quantized[k] = (v, jnp.array(1.0, dtype=jnp.float32))
            continue
        scale = jnp.max(jnp.abs(v)).astype(jnp.float32) / max_q
        scale = jnp.where(scale == 0, jnp.array(1e-8, dtype=jnp.float32), scale)
        q = jnp.clip(jnp.round(v / scale), -max_q, max_q).astype(jnp.int8)
        quantized[k] = (q, scale)
    return quantized


def dequantize_state_dict(qstate: dict[str, tuple[jnp.ndarray, jnp.ndarray]]) -> dict[str, jnp.ndarray]:
    """Inverse of quantize_state_dict."""
    return {k: q.astype(jnp.float32) * s for k, (q, s) in qstate.items()}


def main() -> None:
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install transformers to run the LLM demo: `pip install transformers`") from e

    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run lightweight LLM with optional HF weights.")
    parser.add_argument("--model", choices=["qwen3", "gpt_oss", "deepseek_r1"], default="qwen3")
    parser.add_argument("--repo", type=str, required=True, help="HF repo id, e.g., Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--chat-template", action="store_true", help="Apply simple ChatML template for chat models.")
    args = parser.parse_args()

    cfg_dict = load_hf_config(args.repo, revision=args.revision)
    base = {"qwen3": QWEN3_SMALL, "gpt_oss": GPT_OSS_SMALL, "deepseek_r1": DEEPSEEK_R1_SMALL}[args.model]
    config = hf_config_to_llm_config(cfg_dict, base=base)
    builders = {
        "qwen3": build_qwen3_model,
        "gpt_oss": build_gpt_oss_model,
        "deepseek_r1": build_deepseek_r1_model,
    }
    model = builders[args.model](config)
    tokenizer = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    try:
        model = load_hf_weights_into_llm(model, args.repo, revision=args.revision)
        logger.info("Loaded HF weights")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Falling back to randomly initialized weights; reason: {exc}", stacklevel=2)

    if args.chat_template or ("chat" in args.repo.lower()):
        tokens = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt},
            ],
            add_generation_prompt=True,
        )

    else:
        tokens = tokenizer.encode(
            args.prompt,
            return_tensors="np",
            add_special_tokens=False,
        )[0].tolist()

    output_tokens = llm_generate(
        model,
        tokens=tokens,
        eos_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Shows the text to the user.
    text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    # python -m xax.arch.llm
    main()
