#!/usr/bin/env -S uv run --no-project --script
"""Two-stage TTS architecture for LJSpeech using Mimi codec tokens.

This example keeps the core idea of fine-tuning a pretrained decoder-only LLM on
discrete speech codec tokens, but uses a compact codec vocabulary and a
coarse-to-fine factorization:

Stage 1 (semantic/coarse):
  - Fine-tune a pretrained LLM (Qwen3) to predict Mimi Q0 (semantic) tokens from text.
  - Loss is computed only on the audio segment (prefix-LM style).

Stage 2 (acoustic/fine):
  - Predict Mimi Q1-Q7 residual codebooks using a lightweight non-causal
    transformer over time plus **depth-wise factorization** across codebooks
    (VALL-E-style), conditioned on:
      - frame-aligned hidden states from the semantic LLM
      - the Q0 token stream

Sequence format (stage 1):
  [TEXT_START] [TEXT] [TEXT_END] [AUDIO_START] [Q0_CODES] [AUDIO_END]
"""

import functools
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from scipy.signal import resample_poly
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.whisper.tokenization_whisper_fast import WhisperTokenizerFast

import xax
from xax.arch.attention import RMSNorm, TransformerStack, apply_linear
from xax.arch.llm import chunked_cross_entropy_acc, chunked_cross_entropy_loss

logger = logging.getLogger(__name__)

# Codec constants.
NUM_QUANTIZERS = 8  # Q0 (semantic) + Q1-Q7 (residual)
CODEBOOK_SIZE = xax.MIMI_CODEBOOK_SIZE  # 2048

# Stage-2 padding token (not a valid codec index).
AUDIO_PAD_TOKEN_ID = CODEBOOK_SIZE

# Special token string markers (for text tokenizer).
TEXT_START_TOKEN = "<|text_start|>"
TEXT_END_TOKEN = "<|text_end|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

# Audio Q0 token strings: <|audio_q0_0|> ... <|audio_q0_2047|>
Q0_TOKEN_FMT = "<|audio_q0_{idx}|>"

# LoRA targets.
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up")


class Batch(TypedDict):
    codes: Array  # (bsz, seq_len) text + Q0 tokens
    audio_codes: Array  # (bsz, max_frames, 8) full codec codes (padded)


class SemanticTTSModel(eqx.Module):
    """Stage 1: LLM for predicting Q0 (semantic) codec tokens."""

    llm: xax.LLM

    def generate_tokens(
        self,
        prompt_tokens_s: Array,
        *,
        max_new_tokens: int,
        audio_end_id: int,
        temperature: float,
        top_p: float,
        key: PRNGKeyArray,
        allowed_token_range: tuple[int, int],
        min_new_tokens_before_eos: int,
    ) -> tuple[Array, Array]:
        return xax.llm_generate_jit(
            self.llm,
            prompt_tokens_s,
            eos_id=audio_end_id,
            max_new_tokens=max_new_tokens,
            context_tn=None,
            temperature=temperature,
            top_p=top_p,
            key=key,
            allowed_token_range=allowed_token_range,
            min_new_tokens_before_eos=min_new_tokens_before_eos,
        )


RESIDUAL_HEAD_DIM = 64
RESIDUAL_NUM_HEADS = 4
RESIDUAL_NUM_LAYERS = 4
RESIDUAL_MLP_DIM = 512


class ResidualModel(eqx.Module):
    """Stage 2: Non-causal transformer predicting Q1-Q7 with depth conditioning.

    Conditioned on frame-aligned semantic hidden states and the Q0 stream.
    """

    hidden_proj: eqx.nn.Linear
    q0_embed: eqx.nn.Embedding
    residual_embed: eqx.nn.Embedding
    stack: TransformerStack
    norm: RMSNorm
    out_proj_l: tuple[eqx.nn.Linear, ...]
    layer_embedding_ld: Array

    @staticmethod
    def build(
        llm_embed_dim: int,
        *,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
        key: PRNGKeyArray,
    ) -> "ResidualModel":
        embed_dim = head_dim * num_heads

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        hidden_proj = eqx.nn.Linear(llm_embed_dim, embed_dim, key=k1)
        q0_embed = eqx.nn.Embedding(CODEBOOK_SIZE, embed_dim, key=k2)
        residual_embed = eqx.nn.Embedding(CODEBOOK_SIZE, embed_dim, key=k3)
        stack = TransformerStack.build(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=mlp_dim,
            num_layers=num_layers,
            key=k4,
            causal=False,
            use_rotary_embeddings=True,
        )
        norm = RMSNorm.build(embed_dim, eps=1e-6)
        out_proj_keys = jax.random.split(k5, NUM_QUANTIZERS - 1)
        out_proj_l = tuple(
            eqx.nn.Linear(embed_dim, CODEBOOK_SIZE, use_bias=False, key=out_proj_keys[layer_idx])
            for layer_idx in range(NUM_QUANTIZERS - 1)
        )
        layer_embedding_ld = jax.random.normal(k6, (NUM_QUANTIZERS - 1, embed_dim)) * 0.02
        return ResidualModel(
            hidden_proj=hidden_proj,
            q0_embed=q0_embed,
            residual_embed=residual_embed,
            stack=stack,
            norm=norm,
            out_proj_l=out_proj_l,
            layer_embedding_ld=layer_embedding_ld,
        )

    def _forward_hidden(
        self,
        q0_codes_t: Array,
        semantic_hidden_td: Array,
        frame_mask_t: Array,
    ) -> Array:
        q0_codes_t = jnp.where(frame_mask_t, q0_codes_t, 0)
        q0_codes_t = jnp.clip(q0_codes_t, 0, CODEBOOK_SIZE - 1)
        sem_td = jax.vmap(self.hidden_proj)(semantic_hidden_td)
        q0_td = jax.vmap(self.q0_embed)(q0_codes_t)
        x_td = sem_td + q0_td
        x_td = jnp.where(frame_mask_t[:, None], x_td, 0)

        # Bidirectional attention with padding mask (prevents padded frames from
        # influencing valid frames).
        attn_mask_tt = frame_mask_t[:, None] & frame_mask_t[None, :]
        attn_mask_11tt = attn_mask_tt[None, None, :, :]
        y_td, _ = self.stack.forward(x_td, mask=attn_mask_11tt)
        return jax.vmap(self.norm)(y_td)

    def compute_loss(
        self,
        audio_codes_ft: Array,
        semantic_hidden_td: Array,
    ) -> tuple[Array, Array]:
        """Compute per-codebook CE loss/accuracy for Q1-Q7.

        Args:
            audio_codes_ft: Codec codes, shape (8, max_frames).
            semantic_hidden_td: Frame-aligned semantic hidden states,
                shape (max_frames, llm_embed_dim). Must be zero for padded frames.

        Returns:
            Tuple of (loss_l, acc_l) each shaped (7,).
        """
        frame_mask_t = audio_codes_ft[0] != AUDIO_PAD_TOKEN_ID
        denom = jnp.maximum(frame_mask_t.astype(jnp.float32).sum(), 1.0)
        q0_codes_t = audio_codes_ft[0]

        hidden_td = self._forward_hidden(
            q0_codes_t=q0_codes_t,
            semantic_hidden_td=semantic_hidden_td,
            frame_mask_t=frame_mask_t,
        )

        # Depth-wise conditioning: predict Q1..Q7 sequentially, each conditioned
        # on previous codebooks (VALL-E-style factorization). This stays
        # non-autoregressive over time.
        prev_sum_td = jnp.zeros_like(hidden_td)

        losses = []
        accs = []
        for layer_idx in range(1, NUM_QUANTIZERS):
            targets_t = jnp.where(frame_mask_t, audio_codes_ft[layer_idx], 0)
            targets_t = jnp.clip(targets_t, 0, CODEBOOK_SIZE - 1)

            layer_hidden_td = hidden_td + self.layer_embedding_ld[layer_idx - 1] + prev_sum_td
            logits_tv = apply_linear(layer_hidden_td.astype(jnp.float32), self.out_proj_l[layer_idx - 1])

            loss_t = optax.softmax_cross_entropy_with_integer_labels(logits_tv, targets_t)
            loss = jnp.where(frame_mask_t, loss_t, 0.0).sum() / denom

            pred_t = jnp.argmax(logits_tv, axis=-1).astype(jnp.int32)
            correct_t = (pred_t == targets_t) & frame_mask_t
            acc = correct_t.astype(jnp.float32).sum() / denom

            losses.append(loss)
            accs.append(acc)

            # Teacher forcing for depth conditioning.
            tgt_embed_td = jax.vmap(self.residual_embed)(targets_t)
            tgt_embed_td = jnp.where(frame_mask_t[:, None], tgt_embed_td, 0)
            prev_sum_td = prev_sum_td + tgt_embed_td + self.layer_embedding_ld[layer_idx - 1]

        return jnp.stack(losses), jnp.stack(accs)

    def generate_codes(
        self,
        q0_codes_t: Array,
        semantic_hidden_td: Array,
        *,
        num_frames: Array,
        max_frames: int,
        temperature: float,
        top_p: float,
        key: PRNGKeyArray,
    ) -> Array:
        """Generate full codec codes (Q0..Q7) for a padded frame axis."""
        frame_mask_t = jnp.arange(max_frames) < num_frames

        hidden_td = self._forward_hidden(
            q0_codes_t=q0_codes_t,
            semantic_hidden_td=semantic_hidden_td,
            frame_mask_t=frame_mask_t,
        )

        def sample_from_logits(logits_tv: Array, *, key: PRNGKeyArray) -> Array:
            if temperature <= 0:
                return jnp.argmax(logits_tv, axis=-1).astype(jnp.int32)

            scaled = logits_tv / temperature
            sort_idx_tv = jnp.argsort(scaled, axis=-1, descending=True)
            sort_logits_tv = jnp.take_along_axis(scaled, sort_idx_tv, axis=-1)
            sort_probs_tv = jax.nn.softmax(sort_logits_tv, axis=-1)
            cum_probs_tv = jnp.cumsum(sort_probs_tv, axis=-1)
            nucleus_mask_tv = cum_probs_tv > top_p
            nucleus_mask_tv = nucleus_mask_tv.at[:, 0].set(False)
            masked_logits_tv = jnp.where(nucleus_mask_tv, -jnp.inf, sort_logits_tv)
            sampled_idx_t = jax.random.categorical(key, masked_logits_tv, axis=-1)
            return jnp.take_along_axis(sort_idx_tv, sampled_idx_t[:, None], axis=-1)[:, 0].astype(jnp.int32)

        codes = [jnp.where(frame_mask_t, q0_codes_t, 0)]
        prev_sum_td = jnp.zeros_like(hidden_td)
        for layer_idx in range(1, NUM_QUANTIZERS):
            layer_hidden_td = hidden_td + self.layer_embedding_ld[layer_idx - 1] + prev_sum_td
            logits_tv = apply_linear(layer_hidden_td.astype(jnp.float32), self.out_proj_l[layer_idx - 1])
            key, subkey = jax.random.split(key)
            pred_t = sample_from_logits(logits_tv, key=subkey)
            pred_t = jnp.where(frame_mask_t, pred_t, 0)
            pred_t = jnp.clip(pred_t, 0, CODEBOOK_SIZE - 1)
            codes.append(pred_t)

            pred_embed_td = jax.vmap(self.residual_embed)(pred_t)
            pred_embed_td = jnp.where(frame_mask_t[:, None], pred_embed_td, 0)
            prev_sum_td = prev_sum_td + pred_embed_td + self.layer_embedding_ld[layer_idx - 1]

        return jnp.stack(codes, axis=0)


class FullTTSModel(eqx.Module):
    semantic: SemanticTTSModel
    residual: ResidualModel
    mimi: xax.MimiModel | None
    whisper_transcriber: xax.WhisperTranscriber | None

    @staticmethod
    def build(
        llm: xax.LLM,
        *,
        enable_heavy_eval: bool,
        whisper_repo_id: str,
        residual_head_dim: int,
        residual_num_heads: int,
        residual_num_layers: int,
        residual_mlp_dim: int,
        key: PRNGKeyArray,
    ) -> "FullTTSModel":
        mimi = xax.build_pretrained_mimi() if enable_heavy_eval else None
        semantic = SemanticTTSModel(llm)

        residual = ResidualModel.build(
            llm_embed_dim=llm.config.embed_dim,
            head_dim=residual_head_dim,
            num_heads=residual_num_heads,
            num_layers=residual_num_layers,
            mlp_dim=residual_mlp_dim,
            key=key,
        )

        if enable_heavy_eval:
            whisper_cfg = xax.load_whisper_config(whisper_repo_id)
            whisper_model = xax.build_pretrained_whisper(repo_id=whisper_repo_id)
            whisper_transcriber: xax.WhisperTranscriber | None = xax.WhisperTranscriber(
                model=whisper_model,
                eos_token_id=whisper_cfg.eos_token_id,
            )
        else:
            whisper_transcriber = None

        return FullTTSModel(
            semantic=semantic,
            residual=residual,
            mimi=mimi,
            whisper_transcriber=whisper_transcriber,
        )


@dataclass
class Config(xax.SupervisedConfig):
    # Model settings.
    llm_repo: xax.LLMRepo = xax.field(xax.LLMRepo.QWEN3_600M, help="Pretrained model")
    residual_head_dim: int = xax.field(RESIDUAL_HEAD_DIM, help="Residual model attention head dimension")
    residual_num_heads: int = xax.field(RESIDUAL_NUM_HEADS, help="Residual model number of attention heads")
    residual_num_layers: int = xax.field(RESIDUAL_NUM_LAYERS, help="Residual model number of layers")
    residual_mlp_dim: int = xax.field(RESIDUAL_MLP_DIM, help="Residual model feed-forward hidden dim")
    semantic_loss_weight: float = xax.field(1.0, help="Weight for stage-1 semantic loss in total loss")
    acoustic_loss_weight: float = xax.field(1.0, help="Weight for stage-2 acoustic loss in total loss")
    text_loss_weight: float = xax.field(
        0.0,
        help="Optional LM loss on text prefix (usually 0 for prefix-LM training)",
    )

    # LoRA settings.
    use_lora: bool = xax.field(True, help="Whether to apply LoRA to the semantic LLM")
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(16.0, help="LoRA alpha parameter")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")
    lora_targets: tuple[str, ...] | None = xax.field(DEFAULT_LORA_TARGETS, help="Layer suffixes for LoRA")

    # Training settings.
    learning_rate: float = xax.field(3e-4, help="Peak learning rate")
    semantic_learning_rate: float | None = xax.field(
        None,
        help=(
            "Optional peak learning rate for stage-1 trainable params (LoRA + extra embeddings). "
            "Defaults to learning_rate."
        ),
    )
    residual_learning_rate: float | None = xax.field(
        None,
        help="Optional peak learning rate for stage-2 residual model params. Defaults to learning_rate.",
    )
    warmup_steps: int = xax.field(100, help="Number of warmup steps")
    length_percentile: float = xax.field(0.95, help="Percentile to use for padding lengths")
    q0_corruption_prob: float = xax.field(
        0.0,
        help=(
            "Stage-2 training only: with this probability, replace a Q0 conditioning token with a random code. "
            "This makes the residual model more robust to stage-1 generation errors."
        ),
    )

    # Eval settings.
    enable_heavy_eval: bool = xax.field(
        False,
        help="If true, generate audio + run Whisper ASR in heavy logs (slow, extra memory).",
    )
    whisper_repo_id: str = xax.field(
        "openai/whisper-large-v3-turbo",
        help="Whisper repo for ASR-based evaluation (only used when enable_heavy_eval is true).",
    )
    eval_prompt: str = xax.field("Hello, world! I'm a TTS model.", help="Prompt to use for evaluation")
    eval_prompt_in_domain: str = xax.field(
        "Master these, and do not let them master you.",
        help="Secondary (in-domain) prompt to sanity-check inference",
    )
    semantic_gen_temperature: float = xax.field(0.8, help="Sampling temperature for semantic generation")
    semantic_gen_top_p: float = xax.field(0.9, help="Top-p for semantic generation")
    semantic_gen_min_new_tokens: int = xax.field(
        96,
        help="Minimum Q0 tokens to generate before allowing EOS",
    )
    residual_gen_temperature: float = xax.field(0.0, help="Sampling temperature for residual generation (0=argmax)")
    residual_gen_top_p: float = xax.field(0.95, help="Top-p for residual generation")


class LJSpeechTTS(xax.SupervisedTask[Config]):
    tokenizer: Qwen2TokenizerFast
    whisper_tokenizer: WhisperTokenizerFast | None

    text_start_id: int
    text_end_id: int
    audio_start_id: int
    audio_end_id: int
    first_q0_id: int

    base_vocab_size: int
    extra_vocab_size: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_repo.value)
        self.base_vocab_size = len(self.tokenizer)

        # Add special tokens for sequence boundaries.
        self.tokenizer.add_special_tokens({"additional_special_tokens": [TEXT_START_TOKEN, TEXT_END_TOKEN]})
        self.tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_START_TOKEN, AUDIO_END_TOKEN]})
        self.text_start_id = self.tokenizer.convert_tokens_to_ids(TEXT_START_TOKEN)
        self.text_end_id = self.tokenizer.convert_tokens_to_ids(TEXT_END_TOKEN)
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids(AUDIO_START_TOKEN)
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids(AUDIO_END_TOKEN)

        # Add one token per Mimi Q0 code.
        q0_tokens = [Q0_TOKEN_FMT.format(idx=idx) for idx in range(CODEBOOK_SIZE)]
        self.tokenizer.add_tokens(q0_tokens)
        self.first_q0_id = self.tokenizer.convert_tokens_to_ids(q0_tokens[0])
        self.extra_vocab_size = len(self.tokenizer) - self.base_vocab_size

        if self.first_q0_id < 0:
            raise ValueError("Failed to add Q0 codec tokens to the tokenizer.")

        # Whisper tokenizer is only needed for heavy evaluation.
        if self.config.enable_heavy_eval:
            logger.info("Loading Whisper tokenizer for ASR evaluation")
            path = xax.download_whisper_repo(self.config.whisper_repo_id)
            self.whisper_tokenizer = WhisperTokenizerFast.from_pretrained(str(path))
        else:
            self.whisper_tokenizer = None

        # Precompute eval prompts.
        def make_prompt_tokens(prompt: str) -> Array:
            text_tokens = np.asarray(self.tokenizer.encode(prompt), dtype=np.int32)
            ids = np.concatenate(
                [
                    [self.text_start_id],
                    text_tokens,
                    [self.text_end_id, self.audio_start_id],
                ]
            )
            return jnp.asarray(ids, dtype=jnp.int32)

        self.eval_prompt_tokens = make_prompt_tokens(self.config.eval_prompt)
        self.eval_prompt_in_domain_tokens = make_prompt_tokens(self.config.eval_prompt_in_domain)

    @functools.cached_property
    def _padding_lengths(self) -> tuple[int, int]:
        path = self.dataset_cache_dir / "maximum_lengths.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data["max_seq_len"]), int(data["max_audio_frames"])

    @property
    def max_seq_length(self) -> int:
        return self._padding_lengths[0]

    @property
    def max_audio_frames(self) -> int:
        return self._padding_lengths[1]

    @override
    def get_model(self, params: xax.InitParams) -> FullTTSModel:
        key = params.key

        key, llm_key = jax.random.split(key)
        llm = xax.build_pretrained_llm(
            self.config.llm_repo,
            extra_tokens=self.extra_vocab_size,
            tied_extra_embed=True,
            key=llm_key,
        )

        if self.config.use_lora:
            key, lora_key = jax.random.split(key)
            llm = xax.loraize_by_path(
                llm,
                rank=self.config.lora_rank,
                include_suffixes=list(self.config.lora_targets) if self.config.lora_targets else None,
                alpha=self.config.lora_alpha,
                dropout_rate=self.config.lora_dropout,
                key=lora_key,
            )

        key, mimi_key = jax.random.split(key)
        return FullTTSModel.build(
            llm=llm,
            enable_heavy_eval=self.config.enable_heavy_eval,
            whisper_repo_id=self.config.whisper_repo_id,
            residual_head_dim=self.config.residual_head_dim,
            residual_num_heads=self.config.residual_num_heads,
            residual_num_layers=self.config.residual_num_layers,
            residual_mlp_dim=self.config.residual_mlp_dim,
            key=mimi_key,
        )

    @override
    def get_trainable_filter_spec(self, model: FullTTSModel) -> FullTTSModel:
        # Stage 1: LoRA + extra token embeddings/heads.
        llm_spec = xax.lora_filter_spec(model.semantic.llm)
        extra_embed_spec = jax.tree.map(lambda _: True, llm_spec.extra_embed)
        llm_spec = eqx.tree_at(lambda m: m.extra_embed, llm_spec, extra_embed_spec)
        semantic_spec = SemanticTTSModel(llm=llm_spec)

        # Stage 2: Train all residual params.
        residual_spec = jax.tree.map(eqx.is_inexact_array, model.residual)

        # Mimi + Whisper: frozen (or absent).
        mimi_spec = None if model.mimi is None else jax.tree.map(lambda _: False, model.mimi)
        whisper_spec = (
            None
            if model.whisper_transcriber is None
            else jax.tree.map(lambda _: False, model.whisper_transcriber)
        )

        return FullTTSModel(
            semantic=semantic_spec,
            residual=residual_spec,
            mimi=mimi_spec,
            whisper_transcriber=whisper_spec,
        )

    @override
    def get_optimizer(self) -> xax.Optimizer:
        def make_schedule(peak_lr: float) -> optax.Schedule:
            warmup_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=peak_lr,
                transition_steps=self.config.warmup_steps,
            )
            constant_schedule = optax.constant_schedule(value=peak_lr)
            return optax.join_schedules([warmup_schedule, constant_schedule], [self.config.warmup_steps])

        semantic_lr = self.config.learning_rate
        if self.config.semantic_learning_rate is not None:
            semantic_lr = self.config.semantic_learning_rate
        residual_lr = self.config.learning_rate
        if self.config.residual_learning_rate is not None:
            residual_lr = self.config.residual_learning_rate

        if semantic_lr == residual_lr:
            return optax.adamw(learning_rate=make_schedule(semantic_lr), weight_decay=0.001)

        def label_fn(params: FullTTSModel) -> FullTTSModel:
            mimi_labels = None if params.mimi is None else jax.tree.map(lambda _: "semantic", params.mimi)
            whisper_labels = (
                None
                if params.whisper_transcriber is None
                else jax.tree.map(lambda _: "semantic", params.whisper_transcriber)
            )
            return FullTTSModel(
                semantic=jax.tree.map(lambda _: "semantic", params.semantic),
                residual=jax.tree.map(lambda _: "residual", params.residual),
                mimi=mimi_labels,
                whisper_transcriber=whisper_labels,
            )

        transforms = {
            "semantic": optax.adamw(learning_rate=make_schedule(semantic_lr), weight_decay=0.001),
            "residual": optax.adamw(learning_rate=make_schedule(residual_lr), weight_decay=0.001),
        }
        return optax.multi_transform(transforms, label_fn)

    def _get_extra_head(self, llm: xax.LLM) -> eqx.nn.Embedding | eqx.nn.Linear:
        extra_head = llm.extra_embed if llm.tied_extra_embed else llm.extra_lm_head
        if extra_head is None:
            raise ValueError("Stage-1 model must be built with extra_tokens for audio Q0 tokens.")
        return extra_head

    def _audio_lm_head(self, llm: xax.LLM) -> Callable[[Array], Array]:
        base_vocab_size = llm.config.vocab_size
        extra_head = self._get_extra_head(llm)

        q0_min_extra_id = self.first_q0_id - base_vocab_size
        q0_max_extra_id = q0_min_extra_id + CODEBOOK_SIZE
        audio_end_extra_id = self.audio_end_id - base_vocab_size

        def lm_head(hidden_td: Array) -> Array:
            logits = apply_linear(hidden_td, extra_head)
            token_ids_v = jnp.arange(logits.shape[-1], dtype=jnp.int32)
            allowed_v = (token_ids_v >= q0_min_extra_id) & (token_ids_v < q0_max_extra_id)
            allowed_v = allowed_v | (token_ids_v == audio_end_extra_id)
            min_logit = jnp.asarray(jnp.finfo(logits.dtype).min, dtype=logits.dtype)
            return jnp.where(allowed_v[None, :], logits, min_logit)

        return lm_head

    @override
    def compute_loss(
        self,
        model: FullTTSModel,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        codes_bs = batch["codes"]
        audio_codes_btf = batch["audio_codes"]

        # Stage 1: semantic Q0 prediction (prefix-LM style, loss on audio segment only).
        tokens_bs, targets_bs = codes_bs[:, :-1], codes_bs[:, 1:]
        llm = model.semantic.llm
        base_vocab_size = llm.config.vocab_size

        q0_min_id = self.first_q0_id
        q0_max_id = self.first_q0_id + CODEBOOK_SIZE
        semantic_mask_bs = ((targets_bs >= q0_min_id) & (targets_bs < q0_max_id)) | (targets_bs == self.audio_end_id)

        # Optional text prefix loss (kept off by default).
        text_end_mask_bs = codes_bs == self.text_end_id
        has_text_end_b = jnp.any(text_end_mask_bs, axis=1)
        text_end_idx_b = jnp.argmax(text_end_mask_bs, axis=1)
        target_pos_bs = jnp.broadcast_to(jnp.arange(targets_bs.shape[1])[None, :], targets_bs.shape)
        text_mask_bs = (target_pos_bs < text_end_idx_b[:, None]) & has_text_end_b[:, None]
        text_mask_bs = text_mask_bs & (targets_bs < base_vocab_size)

        audio_lm_head = self._audio_lm_head(llm)

        def compute_stage1_terms(
            sample_tokens_s: Array,
            sample_targets_s: Array,
            sample_semantic_mask_s: Array,
            sample_text_mask_s: Array,
        ) -> tuple[Array, Array, Array, Array, Array]:
            hidden_sd = llm.forward_hidden(sample_tokens_s)

            targets_extra_s = jnp.where(sample_semantic_mask_s, sample_targets_s - base_vocab_size, 0)
            targets_extra_s = jnp.clip(targets_extra_s, 0, self.extra_vocab_size - 1)
            semantic_loss = chunked_cross_entropy_loss(
                hidden_sd,
                targets_extra_s,
                audio_lm_head,
                mask_t=sample_semantic_mask_s,
                chunk_size=128,
            )
            semantic_acc = chunked_cross_entropy_acc(
                hidden_sd,
                targets_extra_s,
                audio_lm_head,
                mask_t=sample_semantic_mask_s,
                chunk_size=128,
            )

            if self.config.text_loss_weight <= 0:
                text_loss = jnp.array(0.0, dtype=jnp.float32)
                text_acc = jnp.array(0.0, dtype=jnp.float32)
            else:
                # Text prefix loss uses the base vocab only.
                def text_head(hidden_td: Array) -> Array:
                    return apply_linear(hidden_td, llm.lm_head)

                targets_text_s = jnp.where(sample_text_mask_s, sample_targets_s, 0)
                targets_text_s = jnp.clip(targets_text_s, 0, base_vocab_size - 1)
                text_loss = chunked_cross_entropy_loss(
                    hidden_sd,
                    targets_text_s,
                    text_head,
                    mask_t=sample_text_mask_s,
                    chunk_size=128,
                )
                text_acc = chunked_cross_entropy_acc(
                    hidden_sd,
                    targets_text_s,
                    text_head,
                    mask_t=sample_text_mask_s,
                    chunk_size=128,
                )

            return semantic_loss, semantic_acc, text_loss, text_acc, hidden_sd

        stage1_loss_b, stage1_acc_b, text_loss_b, text_acc_b, hidden_bsd = jax.vmap(
            compute_stage1_terms,
            in_axes=(0, 0, 0, 0),
        )(tokens_bs, targets_bs, semantic_mask_bs, text_mask_bs)

        stage1_loss = stage1_loss_b.mean()
        stage1_acc = stage1_acc_b.mean()
        text_loss = text_loss_b.mean()
        text_acc = text_acc_b.mean()

        # Stage 2: residual prediction loss.
        stage2_losses_bl, stage2_accs_bl = self._compute_stage2_losses_by_codebook(
            model,
            codes_bs=codes_bs,
            audio_codes_btf=audio_codes_btf,
            hidden_bsd=hidden_bsd,
            key=key,
        )
        stage2_loss_b = stage2_losses_bl.mean(axis=1)
        stage2_acc_b = stage2_accs_bl.mean(axis=1)
        stage2_loss = stage2_loss_b.mean()
        stage2_acc = stage2_acc_b.mean()
        stage2_losses_l = stage2_losses_bl.mean(axis=0)
        stage2_accs_l = stage2_accs_bl.mean(axis=0)

        weighted_stage1_loss = self.config.semantic_loss_weight * stage1_loss
        weighted_stage2_loss = self.config.acoustic_loss_weight * stage2_loss
        weighted_text_loss = self.config.text_loss_weight * text_loss
        total_loss = weighted_stage1_loss + weighted_stage2_loss + weighted_text_loss

        metrics: dict[str, xax.Metric] = {
            "loss": xax.Scalar(total_loss),
            "stage1_loss": xax.Scalar(stage1_loss),
            "stage2_loss": xax.Scalar(stage2_loss),
            "text_loss": xax.Scalar(text_loss),
            "stage1_accuracy": xax.Scalar(stage1_acc),
            "stage2_accuracy": xax.Scalar(stage2_acc),
            "text_accuracy": xax.Scalar(text_acc),
            "weighted_stage1_loss": xax.Scalar(weighted_stage1_loss),
            "weighted_stage2_loss": xax.Scalar(weighted_stage2_loss),
            "weighted_text_loss": xax.Scalar(weighted_text_loss),
        }
        for quantizer_idx in range(1, NUM_QUANTIZERS):
            # Q1..Q7 only.
            layer_idx = quantizer_idx - 1
            metrics[f"stage2_loss_q{quantizer_idx}"] = xax.Scalar(stage2_losses_l[layer_idx])
            metrics[f"stage2_accuracy_q{quantizer_idx}"] = xax.Scalar(stage2_accs_l[layer_idx])

        if heavy and self.config.enable_heavy_eval:
            if model.mimi is None or model.whisper_transcriber is None:
                raise ValueError("enable_heavy_eval was set, but the model was built without Mimi/Whisper modules.")
            audio_t, gt_audio_t, gt_text_t, gen_len, num_frames, invalid_count = self._generate_audio(model, batch, key)
            metrics["generated_audio"] = xax.Audio(audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["real_audio"] = xax.Audio(gt_audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["eval_prompt"] = xax.Tokens(self.eval_prompt_tokens, tokenizer="llm")
            metrics["generated_q0_length"] = xax.Scalar(gen_len.astype(jnp.float32))
            metrics["generated_num_frames"] = xax.Scalar(num_frames.astype(jnp.float32))
            metrics["invalid_q0_token_count"] = xax.Scalar(invalid_count.astype(jnp.float32))

            whisper_audio_t = self._resample_audio_for_whisper(
                audio_t,
                whisper_sample_rate=model.whisper_transcriber.sample_rate,
            )
            transcript_tokens, _, _ = model.whisper_transcriber.transcribe(whisper_audio_t, max_tokens=64)
            metrics["transcript"] = xax.Tokens(transcript_tokens, tokenizer="whisper")

            whisper_gt_audio_t = self._resample_audio_for_whisper(
                gt_audio_t,
                whisper_sample_rate=model.whisper_transcriber.sample_rate,
            )
            gt_transcript_tokens, _, _ = model.whisper_transcriber.transcribe(whisper_gt_audio_t, max_tokens=64)
            metrics["gt_transcript"] = xax.Tokens(gt_transcript_tokens, tokenizer="whisper")
            metrics["gt_text"] = xax.Tokens(gt_text_t, tokenizer="llm")

            # In-domain prompt sanity check.
            key, indomain_key = jax.random.split(key)
            audio_in_domain_t, q0_len_in_domain, num_frames_in_domain, invalid_in_domain = (
                self._generate_audio_from_prompt(
                    model,
                    prompt_tokens_s=self.eval_prompt_in_domain_tokens,
                    key=indomain_key,
                )
            )
            metrics["generated_audio_in_domain"] = xax.Audio(audio_in_domain_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["eval_prompt_in_domain"] = xax.Tokens(self.eval_prompt_in_domain_tokens, tokenizer="llm")
            metrics["generated_q0_length_in_domain"] = xax.Scalar(q0_len_in_domain.astype(jnp.float32))
            metrics["generated_num_frames_in_domain"] = xax.Scalar(num_frames_in_domain.astype(jnp.float32))
            metrics["invalid_q0_token_count_in_domain"] = xax.Scalar(invalid_in_domain.astype(jnp.float32))

            whisper_audio_in_domain_t = self._resample_audio_for_whisper(
                audio_in_domain_t,
                whisper_sample_rate=model.whisper_transcriber.sample_rate,
            )
            transcript_in_domain_tokens, _, _ = model.whisper_transcriber.transcribe(
                whisper_audio_in_domain_t,
                max_tokens=64,
            )
            metrics["transcript_in_domain"] = xax.Tokens(transcript_in_domain_tokens, tokenizer="whisper")

        return total_loss, metrics

    def _compute_stage2_losses_by_codebook(
        self,
        model: FullTTSModel,
        *,
        codes_bs: Array,
        audio_codes_btf: Array,
        hidden_bsd: Array,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Returns per-codebook stage2 losses/accuracies.

        Returns:
            Tuple of (losses_bl, accs_bl) each shaped (bsz, 7) for Q1..Q7.
        """
        bsz = codes_bs.shape[0]
        max_frames = audio_codes_btf.shape[1]
        max_seq_len = codes_bs.shape[1]
        keys_b = jax.random.split(key, bsz)

        def compute_sample(
            codes_s: Array,
            hidden_sd: Array,
            audio_codes_tf: Array,
            key: PRNGKeyArray,
        ) -> tuple[Array, Array]:
            audio_start_mask = codes_s == self.audio_start_id
            audio_start_idx = jnp.where(audio_start_mask.any(), jnp.argmax(audio_start_mask), max_seq_len)
            audio_end_mask = codes_s == self.audio_end_id
            audio_end_idx = jnp.where(audio_end_mask.any(), jnp.argmax(audio_end_mask), max_seq_len)
            q0_start = audio_start_idx + 1
            q0_len = jnp.maximum(audio_end_idx - q0_start, 0)

            q0_hidden_sd = jnp.roll(hidden_sd, -q0_start, axis=0)[:max_frames]
            frame_mask_t = jnp.arange(max_frames) < jnp.minimum(q0_len, max_frames)
            semantic_hidden_td = jnp.where(frame_mask_t[:, None], q0_hidden_sd, 0)

            audio_codes_ft = audio_codes_tf.T
            if self.config.q0_corruption_prob > 0:
                q0_codes_t = audio_codes_ft[0]
                frame_mask_t = q0_codes_t != AUDIO_PAD_TOKEN_ID
                mask_key, codes_key = jax.random.split(key)
                corrupt_mask_t = (jax.random.uniform(mask_key, (max_frames,)) < self.config.q0_corruption_prob) & (
                    frame_mask_t
                )
                random_codes_t = jax.random.randint(codes_key, (max_frames,), 0, CODEBOOK_SIZE)
                corrupted_q0_t = jnp.where(corrupt_mask_t, random_codes_t, q0_codes_t)
                audio_codes_ft = audio_codes_ft.at[0].set(corrupted_q0_t)

            losses_l, accs_l = model.residual.compute_loss(
                audio_codes_ft=audio_codes_ft,
                semantic_hidden_td=semantic_hidden_td,
            )
            return losses_l, accs_l

        stage2_losses_bl, stage2_accs_bl = jax.vmap(compute_sample, in_axes=(0, 0, 0, 0))(
            codes_bs,
            hidden_bsd,
            audio_codes_btf,
            keys_b,
        )
        return stage2_losses_bl, stage2_accs_bl

    def _generate_audio(
        self,
        model: FullTTSModel,
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        if model.mimi is None:
            raise ValueError("Mimi model is required for audio generation.")
        audio_gen_t, gen_len, num_frames, invalid_count = self._generate_audio_from_prompt(
            model,
            prompt_tokens_s=self.eval_prompt_tokens,
            key=key,
        )

        # Ground-truth audio from the first sample.
        gt_codes_tf = batch["audio_codes"][0]  # (T, 8)
        gt_codes_ft = gt_codes_tf.T
        gt_codes_ft = jnp.where((gt_codes_ft >= 0) & (gt_codes_ft < CODEBOOK_SIZE), gt_codes_ft, 0)
        audio_gt = model.mimi.decode(gt_codes_ft)

        gt_ids = batch["codes"][0]
        gt_text_ids = jax.lax.dynamic_slice(gt_ids, (1,), (gt_ids.shape[0] - 1,))

        return audio_gen_t, audio_gt[0], gt_text_ids, gen_len, num_frames, invalid_count

    def _resample_audio_for_whisper(self, audio_t: Array, *, whisper_sample_rate: int) -> Array:
        """Resample Mimi (24 kHz) audio to Whisper sample rate (usually 16 kHz).

        Whisper models expect audio at a fixed sample rate; passing Mimi's
        24 kHz audio directly will produce meaningless transcripts.
        """
        audio_t = audio_t.astype(jnp.float32)
        if xax.MIMI_SAMPLE_RATE == whisper_sample_rate:
            return audio_t
        in_len = int(audio_t.shape[0])
        out_len = max(1, int(round(in_len * whisper_sample_rate / xax.MIMI_SAMPLE_RATE)))
        return jax.image.resize(audio_t, (out_len,), method="linear")

    def _generate_audio_from_prompt(
        self,
        model: FullTTSModel,
        *,
        prompt_tokens_s: Array,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array, Array, Array]:
        if model.mimi is None:
            raise ValueError("Mimi model is required for audio generation.")
        k1, k2 = jax.random.split(key)
        max_frames = self.max_audio_frames

        q0_min_id = self.first_q0_id
        q0_max_id = self.first_q0_id + CODEBOOK_SIZE

        # Ensure we never generate more frames than Stage 2 can decode.
        max_new_tokens = min(
            self.max_seq_length - int(prompt_tokens_s.shape[0]),
            max_frames + 1,  # +1 to allow the EOS token.
        )

        gen_tokens_s, gen_pos = model.semantic.generate_tokens(
            prompt_tokens_s=prompt_tokens_s,
            max_new_tokens=max_new_tokens,
            audio_end_id=self.audio_end_id,
            temperature=self.config.semantic_gen_temperature,
            top_p=self.config.semantic_gen_top_p,
            key=k1,
            allowed_token_range=(q0_min_id, q0_max_id),
            min_new_tokens_before_eos=self.config.semantic_gen_min_new_tokens,
        )

        prompt_len = int(prompt_tokens_s.shape[0])
        gen_only_s = gen_tokens_s[prompt_len:]
        gen_len = jnp.maximum(gen_pos - prompt_len, 0)

        last_idx = jnp.clip(gen_len - 1, 0)
        has_eos = (gen_len > 0) & (gen_only_s[last_idx] == self.audio_end_id)
        gen_len = jnp.where(has_eos, gen_len - 1, gen_len)

        # Stop early on invalid tokens to keep decoding robust.
        valid_mask_s = (gen_only_s >= q0_min_id) & (gen_only_s < q0_max_id)
        prefix_mask_s = jnp.arange(gen_only_s.shape[0]) < gen_len
        invalid_prefix_s = prefix_mask_s & ~valid_mask_s
        invalid_count = jnp.sum(invalid_prefix_s.astype(jnp.int32))
        first_invalid_idx = jnp.argmax(invalid_prefix_s.astype(jnp.int32))
        gen_len = jnp.where(invalid_prefix_s.any(), first_invalid_idx, gen_len)

        num_frames = jnp.minimum(gen_len, jnp.asarray(max_frames, dtype=jnp.int32))
        frame_mask_t = jnp.arange(max_frames) < num_frames

        q0_tokens_s = jnp.where(valid_mask_s, gen_only_s, q0_min_id)
        q0_codes_s = q0_tokens_s - q0_min_id
        q0_codes_t = jnp.where(frame_mask_t, q0_codes_s[:max_frames], 0)
        q0_codes_t = jnp.clip(q0_codes_t, 0, CODEBOOK_SIZE - 1)

        # Frame-aligned semantic hidden states from the generated Q0 stream.
        hidden_sd = model.semantic.llm.forward_hidden(gen_tokens_s)
        q0_hidden_td = hidden_sd[prompt_len : prompt_len + max_frames]
        semantic_hidden_td = jnp.where(frame_mask_t[:, None], q0_hidden_td, 0)

        all_codes_ft = model.residual.generate_codes(
            q0_codes_t=q0_codes_t,
            semantic_hidden_td=semantic_hidden_td,
            num_frames=num_frames,
            max_frames=max_frames,
            temperature=self.config.residual_gen_temperature,
            top_p=self.config.residual_gen_top_p,
            key=k2,
        )
        all_codes_ft = jnp.where(frame_mask_t[None, :], all_codes_ft, 0)
        all_codes_ft = jnp.clip(all_codes_ft, 0, CODEBOOK_SIZE - 1)
        audio_gen = model.mimi.decode(all_codes_ft)

        return audio_gen[0], gen_len, num_frames, invalid_count

    @override
    def decode_tokens(self, tokens: np.ndarray, token_type: str) -> str:
        token_list: list[int] = tokens.tolist()
        match token_type:
            case "whisper":
                if self.whisper_tokenizer is None:
                    return ""
                transcript_tokens = [t for t in token_list[4:]]
                return self.whisper_tokenizer.decode(transcript_tokens, skip_special_tokens=True)
            case "llm":
                transcript_tokens = [t for t in token_list if t < self.first_q0_id]
                return self.tokenizer.decode(transcript_tokens, skip_special_tokens=True)
            case _:
                raise ValueError(f"Invalid token type: {token_type}")

    @override
    def get_dataset(self) -> Dataset:
        return cast(Dataset, self.load_dataset("train"))

    @xax.dataset_fn("train", dependencies=["unpadded"])
    def train_dataset(self) -> Dataset:
        ds = cast(Dataset, self.load_dataset("unpadded"))

        code_lengths = np.array([len(c) for c in ds["codes"]])
        audio_lengths = np.array([len(c) for c in ds["audio_codes"]])
        max_seq_len = int(np.percentile(code_lengths, self.config.length_percentile * 100))
        max_audio_frames = int(np.percentile(audio_lengths, self.config.length_percentile * 100))

        self.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_cache_dir / "maximum_lengths.json").write_text(
            json.dumps({"max_seq_len": max_seq_len, "max_audio_frames": max_audio_frames}),
            encoding="utf-8",
        )

        logger.info(
            "Padding to %.0fth percentile: max_seq_len=%d, max_audio_frames=%d",
            self.config.length_percentile * 100,
            max_seq_len,
            max_audio_frames,
        )

        pre_len = len(ds)
        ds = ds.filter(
            lambda ex: len(ex["codes"]) <= max_seq_len and len(ex["audio_codes"]) <= max_audio_frames,
            desc="Filtering by length",
        )
        logger.info("Filtered %d examples to %d", pre_len, len(ds))

        def pad_sample(example: dict) -> dict:
            codes_raw = np.asarray(example["codes"], dtype=np.int32)
            audio_codes_raw = np.asarray(example["audio_codes"], dtype=np.int32)

            seq_len = min(len(codes_raw), max_seq_len)
            codes = np.full(max_seq_len, self.tokenizer.pad_token_id, dtype=np.int32)
            codes[:seq_len] = codes_raw[:seq_len]

            num_frames = min(len(audio_codes_raw), max_audio_frames)
            audio_codes = np.full((max_audio_frames, NUM_QUANTIZERS), AUDIO_PAD_TOKEN_ID, dtype=np.int32)
            audio_codes[:num_frames] = audio_codes_raw[:num_frames]

            return {"codes": codes, "audio_codes": audio_codes}

        ds = cast(Dataset, ds.map(pad_sample, desc="Padding"))
        return ds

    @xax.dataset_fn("unpadded", dependencies=["tokenized_v2"])
    def unpadded_dataset(self) -> Dataset:
        ds = cast(Dataset, self.load_dataset("tokenized_v2"))

        def prepare_sample(example: dict) -> dict:
            text_tokens = np.asarray(example["text_tokens"], dtype=np.int32)
            # Text segment with explicit boundaries.
            text_with_special = np.concatenate([[self.text_start_id], text_tokens, [self.text_end_id]])

            audio_codes_tc = np.asarray(example["audio_codes"], dtype=np.int32)  # (T, 8)
            q0_codes_t = audio_codes_tc[:, 0]
            q0_tokens = q0_codes_t + self.first_q0_id
            audio_with_special = np.concatenate([[self.audio_start_id], q0_tokens, [self.audio_end_id]])

            codes = np.concatenate([text_with_special, audio_with_special]).astype(np.int32)
            return {"codes": codes, "audio_codes": audio_codes_tc.astype(np.int32)}

        result = cast(Dataset, ds.map(prepare_sample, desc="Preparing sequences"))
        cols_to_keep = ["codes", "audio_codes"]
        cols_to_remove = [c for c in result.column_names if c not in cols_to_keep]
        if cols_to_remove:
            result = result.remove_columns(cols_to_remove)
        return cast(Dataset, result)

    @xax.dataset_fn("tokenized_v2", use_hash=False)
    def tokenized_v2_dataset(self) -> Dataset:
        columns = ["text_tokens", "audio_codes"]

        logger.info("Loading LJSpeech dataset...")
        raw_ds = load_dataset("keithito/lj_speech", split="train")

        def resample_audio(example: dict) -> dict:
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            if sr != xax.MIMI_SAMPLE_RATE:
                # Avoid librosa/numba in this repo environment; SciPy's polyphase
                # resampler is stable and deterministic.
                gcd = math.gcd(sr, xax.MIMI_SAMPLE_RATE)
                up = xax.MIMI_SAMPLE_RATE // gcd
                down = sr // gcd
                audio = resample_poly(audio, up=up, down=down).astype(np.float32)
            else:
                audio = audio.astype(np.float32)
            max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
            audio = audio / max_val
            return {"resampled_audio": audio, "audio_length": len(audio)}

        logger.info("Stage 1: Resampling audio on CPU...")
        # Note: datasets' multiprocessing defaults to `fork`, which is a common
        # deadlock source when used after importing JAX (multithreaded). Keep
        # this single-process unless we explicitly switch to spawn.
        resampled_ds = cast(Dataset, raw_ds.map(resample_audio, desc="Resampling audio"))

        logger.info("Stage 2: Encoding audio with Mimi on GPU (batched)...")
        mimi = xax.build_pretrained_mimi(dtype=jnp.bfloat16)
        max_audio_len = max(resampled_ds["audio_length"])
        hop_length = int(round(mimi.config.sampling_rate / mimi.config.frame_rate))

        @jax.jit
        def batch_encode(audio_bct: Array) -> Array:
            return jax.vmap(lambda audio_ct: mimi.encode(audio_ct, num_quantizers=NUM_QUANTIZERS))(audio_bct)

        def encode_and_tokenize_batch(examples: dict[str, list]) -> dict[str, list]:
            outputs: dict[str, list] = {column: [] for column in columns}
            audio_list = examples["resampled_audio"]
            bsz = len(audio_list)

            audio_batch_np = np.zeros((bsz, max_audio_len), dtype=np.float32)
            for idx, audio in enumerate(audio_list):
                audio_batch_np[idx, : len(audio)] = audio
            audio_batch = jnp.asarray(audio_batch_np, dtype=jnp.bfloat16)

            codes_batch_bct = batch_encode(audio_batch[:, None, :])
            codes_batch_np = np.asarray(codes_batch_bct)

            for idx in range(bsz):
                audio_codes_ct = codes_batch_np[idx]
                orig_audio_len = examples["audio_length"][idx]

                actual_frames = audio_codes_ct.shape[1]
                # Mimi uses a 12.5 Hz frame rate by default (hop_length=1920 at 24 kHz).
                # Empirically, `mimi.encode`'s time axis matches:
                #   frames = floor((n_samples + hop_length / 2) / hop_length)
                estimated_frames = max(1, int((orig_audio_len + hop_length // 2) // hop_length))
                valid_frames = min(estimated_frames, actual_frames)

                audio_codes_tc = audio_codes_ct[:, :valid_frames].T.astype(np.int32)

                text = examples["normalized_text"][idx]
                text_tokens = np.asarray(self.tokenizer.encode(text), dtype=np.int32)

                outputs["text_tokens"].append(text_tokens)
                outputs["audio_codes"].append(audio_codes_tc)

            return outputs

        ds = cast(
            Dataset,
            resampled_ds.map(
                encode_and_tokenize_batch,
                batched=True,
                batch_size=32,
                remove_columns=resampled_ds.column_names,
                desc="Encoding with Mimi",
            ),
        )

        logger.info("Dataset preprocessing complete. %d samples", len(ds))
        logger.info("Columns: text_tokens (T,), audio_codes (T, %d)", NUM_QUANTIZERS)
        return ds


if __name__ == "__main__":
    LJSpeechTTS.launch(
        Config(
            batch_size=8,
            max_grad_norm=2.0,
            gradient_accumulation_steps=1,
            log_heavy_every_n_seconds=300,
            step_kind="second",
        ),
    )
