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

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, replace
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from xax.arch.attention import RotaryEmbedding

try:  # Optional HF hub only
    from huggingface_hub import snapshot_download

    _HF_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _HF_AVAILABLE = False

try:  # Optional tokenizer (avoids transformers dependency)
    from tokenizers import Tokenizer as HFTokenizer  # type: ignore[tokenized-import]

    _TOKENIZERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _TOKENIZERS_AVAILABLE = False

try:  # Optional safetensors for weight loading
    from safetensors.numpy import load_file as safe_load  # type: ignore[safetensors-import]

    _SAFETENSORS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _SAFETENSORS_AVAILABLE = False

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

    def with_vocab(self, vocab_size: int) -> "LLMConfig":
        return replace(self, vocab_size=vocab_size)


class MultiHeadAttention(eqx.Module):
    """Grouped-query attention with rotary embeddings."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    rotary: RotaryEmbedding
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
        bsz, tsz, _ = x_btd.shape
        q_bthd = _linear(x_btd, self.q_proj).reshape(bsz, tsz, self.q_heads, self.head_dim)
        k_bthd = _linear(x_btd, self.k_proj).reshape(bsz, tsz, self.kv_heads, self.head_dim)
        v_bthd = _linear(x_btd, self.v_proj).reshape(bsz, tsz, self.kv_heads, self.head_dim)

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
        gated_btd = jax.nn.silu(_linear(x_btd, self.gate)) * _linear(x_btd, self.up)
        return _linear(gated_btd, self.down)


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


class TransformerBlock(eqx.Module):
    attn: MultiHeadAttention
    attn_norm: "RMSNorm"
    mlp_norm: "RMSNorm"
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
        for i in range(config.num_layers):
            k_attn, k_mlp = jax.random.split(block_keys[i], 2)
            attn = MultiHeadAttention(
                q_proj=eqx.nn.Linear(config.embed_dim, config.q_heads * config.head_dim, key=k_attn),
                k_proj=eqx.nn.Linear(config.embed_dim, config.kv_heads * config.head_dim, key=k_attn),
                v_proj=eqx.nn.Linear(config.embed_dim, config.kv_heads * config.head_dim, key=k_attn),
                o_proj=eqx.nn.Linear(config.q_heads * config.head_dim, config.embed_dim, key=k_attn),
                rotary=RotaryEmbedding(head_dim=config.head_dim, base=config.rope_theta),
                q_heads=config.q_heads,
                kv_heads=config.kv_heads,
                head_dim=config.head_dim,
                dropout_rate=config.dropout_rate,
            )
            mlp = FeedForward(
                gate=eqx.nn.Linear(config.embed_dim, mlp_width, key=k_mlp),
                up=eqx.nn.Linear(config.embed_dim, mlp_width, key=k_mlp),
                down=eqx.nn.Linear(mlp_width, config.embed_dim, key=k_mlp),
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
        lm_head = eqx.nn.Linear(config.embed_dim, config.vocab_size, key=k_head)
        return cls(embed=embed, blocks=tuple(blocks), norm=norm, lm_head=lm_head, config=config)

    def __call__(
        self,
        tokens_bt: Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> Array:
        """Returns logits shaped (bsz, tsz, vocab_size)."""
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
QWEN3_SMALL = LLMConfig(vocab_size=32000, embed_dim=256, q_heads=8, kv_heads=4, head_dim=32, num_layers=4, max_tsz=512)
GPT_OSS_SMALL = LLMConfig(
    vocab_size=32000, embed_dim=384, q_heads=12, kv_heads=4, head_dim=32, num_layers=6, max_tsz=512
)
DEEPSEEK_R1_SMALL = LLMConfig(
    vocab_size=32000, embed_dim=512, q_heads=16, kv_heads=8, head_dim=32, num_layers=8, max_tsz=512
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


def _assert_hf_available() -> None:
    if not _HF_AVAILABLE:
        raise ImportError("huggingface_hub and transformers are required for weight download/conversion.")


def download_repo(repo_id: str, revision: str | None = None, cache_dir: str | None = None) -> str:
    """Downloads a repo snapshot from the Huggingface Hub and returns the local path."""
    _assert_hf_available()
    return snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)


def load_hf_config(repo_id: str, revision: str | None = None) -> dict[str, object]:
    """Loads HF config.json as dict."""
    _assert_hf_available()
    path = download_repo(repo_id, revision=revision)
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(repo_id: str, revision: str | None = None) -> HFTokenizer:
    """Loads a tokenizer from HF without requiring transformers if possible."""
    _assert_hf_available()
    snapshot_path = download_repo(repo_id, revision=revision)
    tok_path = f"{snapshot_path}/tokenizer.json"
    if _TOKENIZERS_AVAILABLE and jax.device_count() >= 0:  # cheap guard to keep lint happy
        tokenizer = HFTokenizer.from_file(tok_path)

        class _Wrapper:
            def __init__(self, tok: HFTokenizer) -> None:
                self.tok = tok
                self.pad_token_id = tok.token_to_id("<pad>") or tok.token_to_id("<s>") or tok.token_to_id("<unk>")
                self.eos_token_id = tok.token_to_id("</s>") or self.pad_token_id
                self.bos_token_id = tok.token_to_id("<s>")

            def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, list[int]]:
                ids = self.tok.encode(text).ids
                bos = self.tok.token_to_id("<s>")
                if bos is not None and (len(ids) == 0 or ids[0] != bos):
                    ids = [bos] + ids
                return {"input_ids": ids}

            def decode(self, ids: list[int] | Array, skip_special_tokens: bool = True) -> str:
                return self.tok.decode(ids)

        return _Wrapper(tokenizer)

    warnings.warn("tokenizers not available; install `tokenizers` for tokenizer support.", stacklevel=2)
    raise ImportError("Tokenizer loading requires `tokenizers` package.")


def _apply_chat_template(
    tokenizer: HFTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Render chat template matching tokenizer_config.json for Qwen chat."""
    rendered = ""
    if messages and messages[0]["role"] != "system":
        rendered += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

    for msg in messages:
        rendered += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

    if add_generation_prompt:
        rendered += "<|im_start|>assistant\n"
        if enable_thinking and tokenizer.token_to_id("<|think|>") is not None:
            rendered += "<|think|>"

    return rendered


def _fetch_state_dict(repo_id: str, revision: str | None = None) -> tuple[dict[str, object], dict[str, jnp.ndarray]]:
    _assert_hf_available()
    if not _SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required to load HF weights.")
    snapshot_path = download_repo(repo_id, revision=revision)
    safes = [os.path.join(snapshot_path, f) for f in os.listdir(snapshot_path) if f.endswith(".safetensors")]
    if not safes:
        raise FileNotFoundError("No .safetensors files found in snapshot.")
    state: dict[str, jnp.ndarray] = {}
    for sf in safes:
        state.update(safe_load(sf))
    config = json.loads(open(f"{snapshot_path}/config.json", "r", encoding="utf-8").read())
    return config, state


def _maybe_get(key_sub: str, state: dict[str, jnp.ndarray]) -> jnp.ndarray:
    matches = [k for k in state if key_sub in k]
    if not matches:
        raise KeyError(f"Missing parameter containing '{key_sub}'")
    if len(matches) > 1:
        # choose shortest match to reduce ambiguity
        matches = sorted(matches, key=len)
    return state[matches[0]]


def hf_config_to_llm_config(cfg: dict[str, object], base: LLMConfig | None = None) -> LLMConfig:
    """Derive an LLMConfig from an HF config dict."""
    base = base or QWEN3_SMALL
    embed_dim = int(cfg.get("hidden_size", base.embed_dim))
    q_heads = int(cfg.get("num_attention_heads", base.q_heads))
    kv_heads = int(cfg.get("num_key_value_heads", max(1, q_heads // 2)))
    head_dim = embed_dim // q_heads
    num_layers = int(cfg.get("num_hidden_layers", base.num_layers))
    vocab_size = int(cfg.get("vocab_size", base.vocab_size))
    mlp_hidden_dim = int(cfg.get("intermediate_size", embed_dim * base.mlp_mult))
    mlp_mult = max(1, mlp_hidden_dim // embed_dim)
    rope_theta = float(cfg.get("rope_theta", base.rope_theta))
    rms_eps = float(cfg.get("rms_norm_eps", base.rms_eps))
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

    def map_linear(eq_lin: eqx.nn.Linear, w: jnp.ndarray, b: jnp.ndarray | None) -> eqx.nn.Linear:
        if w.shape == eq_lin.weight.shape:
            w_mapped = w
        elif w.shape[::-1] == eq_lin.weight.shape:
            w_mapped = w.T
        else:
            raise ValueError(f"Shape mismatch for linear: expected {eq_lin.weight.shape}, got {w.shape}")
        eq_lin = eqx.tree_at(lambda lin: lin.weight, eq_lin, jnp.asarray(w_mapped, dtype=jnp.float32))
        if eq_lin.bias is not None and b is not None:
            if b.shape != eq_lin.bias.shape:
                raise ValueError(f"Shape mismatch for bias: expected {eq_lin.bias.shape}, got {b.shape}")
            eq_lin = eqx.tree_at(lambda lin: lin.bias, eq_lin, jnp.asarray(b, dtype=jnp.float32))
        return eq_lin

    # Embedding + LM head
    embed_w = (
        _maybe_get("embed_tokens.weight", state)
        if any("embed_tokens.weight" in k for k in state)
        else _maybe_get("wte.weight", state)
    )
    model = eqx.tree_at(lambda m: m.embed.weight, model, jnp.asarray(embed_w, dtype=jnp.float32))
    lm_head_w = _maybe_get("lm_head.weight", state) if any("lm_head.weight" in k for k in state) else embed_w
    if model.lm_head.weight.shape == embed_w.shape:
        model = eqx.tree_at(lambda m: m.lm_head.weight, model, jnp.asarray(lm_head_w, dtype=jnp.float32))
        # Tie if shapes match to stay faithful to HF tying
        model = tie_embedding_and_head(model)

    # Per-layer mappings
    for idx, block in enumerate(model.blocks):
        # Attention projections
        def kname(layer_idx: int, suffix: str) -> str:
            return f"layers.{layer_idx}.self_attn.{suffix}"

        q_w = _maybe_get(
            kname(idx, "q_proj.weight")
            if any(kname(idx, "q_proj.weight") in k for k in state)
            else f"model.layers.{idx}.self_attn.q_proj.weight",
            state,
        )
        k_w = _maybe_get(
            kname(idx, "k_proj.weight")
            if any(kname(idx, "k_proj.weight") in k for k in state)
            else f"model.layers.{idx}.self_attn.k_proj.weight",
            state,
        )
        v_w = _maybe_get(
            kname(idx, "v_proj.weight")
            if any(kname(idx, "v_proj.weight") in k for k in state)
            else f"model.layers.{idx}.self_attn.v_proj.weight",
            state,
        )
        o_w = _maybe_get(
            kname(idx, "o_proj.weight")
            if any(kname(idx, "o_proj.weight") in k for k in state)
            else f"model.layers.{idx}.self_attn.o_proj.weight",
            state,
        )

        q_b = state.get(kname(idx, "q_proj.bias"))
        k_b = state.get(kname(idx, "k_proj.bias"))
        v_b = state.get(kname(idx, "v_proj.bias"))
        o_b = state.get(kname(idx, "o_proj.bias"))

        block = eqx.tree_at(lambda b: b.attn.q_proj, block, map_linear(block.attn.q_proj, q_w, q_b))
        block = eqx.tree_at(lambda b: b.attn.k_proj, block, map_linear(block.attn.k_proj, k_w, k_b))
        block = eqx.tree_at(lambda b: b.attn.v_proj, block, map_linear(block.attn.v_proj, v_w, v_b))
        block = eqx.tree_at(lambda b: b.attn.o_proj, block, map_linear(block.attn.o_proj, o_w, o_b))

        # Norms
        pre_gamma = (
            _maybe_get(kname(idx, "input_layernorm.weight"), state)
            if any(kname(idx, "input_layernorm.weight") in k for k in state)
            else _maybe_get(f"layers.{idx}.input_layernorm.weight", state)
        )
        post_gamma = (
            _maybe_get(kname(idx, "post_attention_layernorm.weight"), state)
            if any(kname(idx, "post_attention_layernorm.weight") in k for k in state)
            else _maybe_get(f"layers.{idx}.post_attention_layernorm.weight", state)
        )
        block = eqx.tree_at(lambda b: b.attn_norm.weight, block, jnp.asarray(pre_gamma, dtype=jnp.float32))
        block = eqx.tree_at(lambda b: b.mlp_norm.weight, block, jnp.asarray(post_gamma, dtype=jnp.float32))

        # MLP
        gate_w = _maybe_get(f"layers.{idx}.mlp.gate_proj.weight", state)
        up_w = _maybe_get(f"layers.{idx}.mlp.up_proj.weight", state)
        down_w = _maybe_get(f"layers.{idx}.mlp.down_proj.weight", state)
        gate_b = state.get(f"layers.{idx}.mlp.gate_proj.bias")
        up_b = state.get(f"layers.{idx}.mlp.up_proj.bias")
        down_b = state.get(f"layers.{idx}.mlp.down_proj.bias")

        block = eqx.tree_at(lambda b: b.mlp.gate, block, map_linear(block.mlp.gate, gate_w, gate_b))
        block = eqx.tree_at(lambda b: b.mlp.up, block, map_linear(block.mlp.up, up_w, up_b))
        block = eqx.tree_at(lambda b: b.mlp.down, block, map_linear(block.mlp.down, down_w, down_b))

        model = eqx.tree_at(lambda mod, layer_idx=idx: mod.blocks[layer_idx], model, block)

    # Final norm
    final_gamma = (
        _maybe_get("norm.weight", state) if any("norm.weight" in k for k in state) else state.get("ln_f.weight")
    )
    if final_gamma is not None:
        model = eqx.tree_at(lambda m: m.norm.weight, model, jnp.asarray(final_gamma, dtype=jnp.float32))

    return model


def generate(
    model: LLM,
    tokenizer: HFTokenizer | Callable[..., dict[str, Array | list[int]]],
    prompt: str,
    max_new_tokens: int = 20,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    chat_template: bool = False,
) -> str:
    """Sampling-based decoding for quick sanity checks."""
    if isinstance(tokenizer, HFTokenizer):
        if chat_template:
            formatted_prompt = _apply_chat_template(
                tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True
            )
        else:
            formatted_prompt = prompt
        tokens = tokenizer(formatted_prompt).input_ids
    else:
        formatted_prompt = prompt
        tokens = tokenizer(formatted_prompt, return_tensors="np")["input_ids"]

    tokens_bt = jnp.array(tokens, dtype=jnp.int32)
    if tokens_bt.ndim == 1:
        tokens_bt = tokens_bt[None, :]
    key = jax.random.key(0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
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
    return tokenizer.decode(jnp.array(tokens_bt[0]).tolist(), skip_special_tokens=True)


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
    tokenizer = load_tokenizer(args.repo, revision=args.revision)
    try:
        model = load_hf_weights_into_llm(model, args.repo, revision=args.revision)
        logger.info("Loaded HF weights")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Falling back to randomly initialized weights; reason: {exc}", stacklevel=2)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chat_template=args.chat_template or ("chat" in args.repo.lower()),
    )
    print(text)


if __name__ == "__main__":
    # python -m xax.arch.llm
    main()
