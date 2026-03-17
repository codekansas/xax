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
import hashlib
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from jaxtyping import Array, PRNGKeyArray
from scipy.signal import resample_poly
from transformers import AutoConfig, AutoTokenizer
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
    """Stage 1 semantic model.

    By default this is the original autoregressive Q0 decoder (a pretrained LLM
    over text + Q0 tokens). Optionally it can be augmented with a lightweight
    non-autoregressive text-to-frame upsampler that predicts all Q0 frames in
    parallel from text-prefix hidden states.
    """

    llm: xax.LLM
    future_q0_heads: tuple[eqx.nn.Linear, ...] | None = None
    non_ar_in_proj: eqx.nn.Linear | None = None
    non_ar_stack: TransformerStack | None = None
    non_ar_norm: RMSNorm | None = None
    non_ar_out_proj: eqx.nn.Linear | None = None
    non_ar_to_llm_proj: eqx.nn.Linear | None = None

    @property
    def use_non_ar(self) -> bool:
        return self.non_ar_in_proj is not None

    @staticmethod
    def build(
        llm: xax.LLM,
        *,
        future_prediction_steps: int,
        use_non_ar: bool,
        non_ar_hidden_dim: int,
        non_ar_num_heads: int,
        non_ar_num_layers: int,
        non_ar_mlp_dim: int,
        key: PRNGKeyArray,
    ) -> "SemanticTTSModel":
        if not use_non_ar and future_prediction_steps <= 0:
            return SemanticTTSModel(llm=llm)

        keys = jax.random.split(key, 4 + max(future_prediction_steps, 0))
        future_q0_heads = None
        if future_prediction_steps > 0:
            future_q0_heads = tuple(
                eqx.nn.Linear(llm.config.embed_dim, CODEBOOK_SIZE, use_bias=False, key=keys[idx])
                for idx in range(future_prediction_steps)
            )

        if not use_non_ar:
            return SemanticTTSModel(llm=llm, future_q0_heads=future_q0_heads)

        k1, k2, k3, k4 = keys[-4:]
        non_ar_in_proj = eqx.nn.Linear(llm.config.embed_dim, non_ar_hidden_dim, key=k1)
        non_ar_stack = TransformerStack.build(
            embed_dim=non_ar_hidden_dim,
            num_heads=non_ar_num_heads,
            ff_dim=non_ar_mlp_dim,
            num_layers=non_ar_num_layers,
            key=k2,
            causal=False,
            use_rotary_embeddings=True,
        )
        non_ar_norm = RMSNorm.build(non_ar_hidden_dim, eps=1e-6)
        non_ar_out_proj = eqx.nn.Linear(non_ar_hidden_dim, CODEBOOK_SIZE, use_bias=False, key=k3)
        non_ar_to_llm_proj = eqx.nn.Linear(non_ar_hidden_dim, llm.config.embed_dim, use_bias=False, key=k4)
        return SemanticTTSModel(
            llm=llm,
            future_q0_heads=future_q0_heads,
            non_ar_in_proj=non_ar_in_proj,
            non_ar_stack=non_ar_stack,
            non_ar_norm=non_ar_norm,
            non_ar_out_proj=non_ar_out_proj,
            non_ar_to_llm_proj=non_ar_to_llm_proj,
        )

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
        if self.use_non_ar:
            raise ValueError("generate_tokens is only valid for the autoregressive semantic decoder.")
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
        semantic_future_prediction_steps: int,
        semantic_non_ar: bool,
        semantic_non_ar_hidden_dim: int,
        semantic_non_ar_num_heads: int,
        semantic_non_ar_num_layers: int,
        semantic_non_ar_mlp_dim: int,
        whisper_repo_id: str,
        residual_head_dim: int,
        residual_num_heads: int,
        residual_num_layers: int,
        residual_mlp_dim: int,
        key: PRNGKeyArray,
    ) -> "FullTTSModel":
        k1, k2 = jax.random.split(key)
        mimi = xax.build_pretrained_mimi() if enable_heavy_eval else None
        semantic = SemanticTTSModel.build(
            llm,
            future_prediction_steps=semantic_future_prediction_steps,
            use_non_ar=semantic_non_ar,
            non_ar_hidden_dim=semantic_non_ar_hidden_dim,
            non_ar_num_heads=semantic_non_ar_num_heads,
            non_ar_num_layers=semantic_non_ar_num_layers,
            non_ar_mlp_dim=semantic_non_ar_mlp_dim,
            key=k1,
        )

        residual = ResidualModel.build(
            llm_embed_dim=llm.config.embed_dim,
            head_dim=residual_head_dim,
            num_heads=residual_num_heads,
            num_layers=residual_num_layers,
            mlp_dim=residual_mlp_dim,
            key=k2,
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
    semantic_non_ar: bool = xax.field(
        False,
        help=(
            "If true, replace autoregressive stage-1 Q0 decoding with a lightweight non-autoregressive "
            "text-to-frame semantic predictor built on top of Qwen text-prefix hidden states."
        ),
    )
    semantic_non_ar_hidden_dim: int = xax.field(512, help="Internal hidden dim for the non-AR stage-1 upsampler")
    semantic_non_ar_num_heads: int = xax.field(8, help="Attention heads for the non-AR stage-1 upsampler")
    semantic_non_ar_num_layers: int = xax.field(2, help="Transformer layers for the non-AR stage-1 upsampler")
    semantic_non_ar_mlp_dim: int = xax.field(1024, help="Feed-forward hidden dim for the non-AR stage-1 upsampler")
    semantic_future_prediction_steps: int = xax.field(
        0,
        help=(
            "Number of extra future-Q0 heads for Stage 1. Each head predicts an additional future semantic token "
            "from the same hidden state (multi-token prediction style)."
        ),
    )
    semantic_future_prediction_weight: float = xax.field(
        0.0,
        help="Weight for the auxiliary Stage-1 future-Q0 prediction loss.",
    )
    semantic_loss_weight: float = xax.field(1.0, help="Weight for stage-1 semantic loss in total loss")
    acoustic_loss_weight: float = xax.field(1.0, help="Weight for stage-2 acoustic loss in total loss")
    semantic_eos_weight: float = xax.field(
        1.0,
        help=(
            "Relative weighting for the stage-1 AUDIO_END token within the semantic loss. "
            "Increasing this can make EOS emission more reliable during generation."
        ),
    )
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
    text_source: str = xax.field(
        "normalized",
        help=(
            "Which LJSpeech text field to use for conditioning. "
            "`normalized` matches keithito/lj_speech normalized_text. "
            "`raw` uses the original text with punctuation/casing. "
            "`both` doubles the dataset by including both variants. "
            "`both_plus_normalized` keeps both variants but oversamples normalized text 2:1 over raw. "
            "`both_weighted` uses the repeat counts below for a generic normalized/raw mixture."
        ),
    )
    text_source_weighted_normalized_repeats: int = xax.field(
        1,
        help="Only used when text_source=both_weighted. Number of normalized-text copies per utterance.",
    )
    text_source_weighted_raw_repeats: int = xax.field(
        1,
        help="Only used when text_source=both_weighted. Number of raw-text copies per utterance.",
    )
    q0_corruption_prob: float = xax.field(
        0.0,
        help=(
            "Stage-2 training only: with this probability, replace a Q0 conditioning token with a random code. "
            "This makes the residual model more robust to stage-1 generation errors."
        ),
    )
    semantic_q0_corruption_prob: float = xax.field(
        0.0,
        help=(
            "Stage-1 training only: with this probability, corrupt an input Q0 token in the teacher-forced "
            "context with a random Q0 token. This is a simple denoising objective that can reduce exposure "
            "bias and make EOS emission more reliable under free-running generation."
        ),
    )
    semantic_self_condition_prob: float = xax.field(
        0.0,
        help=(
            "Stage-1 training only: probability of replacing a random suffix of teacher-forced Q0 inputs with "
            "the model's own greedy predictions from a first pass. This lightweight scheduled-sampling variant "
            "directly targets exposure bias in semantic generation."
        ),
    )
    semantic_self_condition_use_inference_policy: bool = xax.field(
        False,
        help=(
            "When semantic_self_condition_prob > 0 and fixed blockwise Stage-1 decode is active, replace the "
            "random Q0 suffix with a free-running blockwise decode that matches the inference policy instead of "
            "cheap teacher-forced one-step predictions. Early keep counts are snapped to grouped-step boundaries "
            "so the grouped->exact transition stays aligned with inference."
        ),
    )
    semantic_self_condition_grouped_prefix_only: bool = xax.field(
        False,
        help=(
            "Only used when semantic_self_condition_use_inference_policy=true. Limit the expensive free-running "
            "blockwise decode to the remaining grouped-prefix window, then fill the later suffix with the usual "
            "cheap one-step predictions. This is a lower-cost hybrid train/infer-match variant for the early "
            "grouped, late exact regime."
        ),
    )
    semantic_self_condition_inference_policy_prob: float = xax.field(
        1.0,
        help=(
            "Only used when semantic_self_condition_use_inference_policy=true. Conditional probability that a "
            "self-conditioned example uses the expensive inference-matched blockwise decode instead of the usual "
            "cheap one-step prediction path. Values below 1.0 create a mixed scheduled-sampling regime that keeps "
            "most examples cheap while occasionally injecting a true grouped-prefix free-running replacement, "
            "including the targeted early-prefix variant."
        ),
    )
    semantic_self_condition_match_early_grouped_prefix: bool = xax.field(
        False,
        help=(
            "Only used when semantic_self_condition_use_inference_policy=true. Instead of replacing a random Q0 "
            "suffix, replace only the first grouped-prefix window after AUDIO_START with a short free-running "
            "blockwise decode from the pure text prompt, then leave the later Q0 inputs teacher-forced. This is a "
            "targeted, lower-cost train/infer-match path for the early-grouped, late-exact regime."
        ),
    )
    semantic_self_condition_match_grouped_future_slots_only: bool = xax.field(
        False,
        help=(
            "Only used when semantic_self_condition_match_early_grouped_prefix=true. Restrict the early-prefix "
            "replacement to the grouped positions that are speculative under the current blockwise policy instead of "
            "replacing the whole early grouped window. For exact-last grouped decoding this means only the middle "
            "future-token slots are replaced."
        ),
    )
    detach_semantic_hidden_for_stage2: bool = xax.field(
        False,
        help=(
            "If true, stop gradients from the stage-2 residual loss from flowing into the semantic LLM. "
            "This can preserve the LLM's text-conditioning behavior and sometimes improves OOD stability, "
            "at the cost of weaker joint co-adaptation."
        ),
    )
    residual_semantic_hidden_dropout_prob: float = xax.field(
        0.0,
        help=(
            "Stage-2 training only: dropout probability for frame-aligned semantic hidden states. "
            "This prevents the residual model from over-relying on semantic hidden states that may be "
            "distribution-shifted at inference (generated Q0 stream), improving robustness."
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
    semantic_gen_temperature: float = xax.field(
        0.0,
        help=(
            "Sampling temperature for semantic generation (0=greedy). "
            "For evaluation, greedy decoding is usually more stable than sampling."
        ),
    )
    semantic_gen_top_p: float = xax.field(1.0, help="Top-p for semantic generation")
    semantic_gen_min_new_tokens: int = xax.field(
        0,
        help="Minimum Q0 tokens to generate before allowing EOS",
    )
    semantic_block_decode_size: int = xax.field(
        1,
        help=(
            "Optional blockwise stage-1 generation size for autoregressive decoding. "
            "1 keeps standard token-by-token decoding; values >1 use future-Q0 heads to append small token blocks."
        ),
    )
    semantic_block_decode_max_grouped_tokens: int = xax.field(
        0,
        help=(
            "Optional limit on how many new semantic tokens to decode with grouped blockwise inference before "
            "falling back to one-at-a-time decoding. <=0 keeps grouped decoding active for the whole utterance."
        ),
    )
    semantic_block_decode_schedule: str | None = xax.field(
        None,
        help=(
            "Optional comma-separated per-step block decode schedule, e.g. '3,3,1,2,2,1'. "
            "Each entry is the number of semantic tokens to emit on that decode step; 1 means exact token-by-token. "
            "When set, this overrides semantic_block_decode_max_grouped_tokens and allows richer position-dependent "
            "early grouped-prefix transition policies."
        ),
    )
    semantic_block_decode_exact_last_token: bool = xax.field(
        False,
        help=(
            "If true, grouped block decoding ends each grouped step on an exact token: emit the first token exactly, "
            "use future heads only for the middle speculative tokens, then decode the final token of the grouped step "
            "with the exact next-token head after feeding the speculative prefix."
        ),
    )
    semantic_eval_compare_exact_last_candidate: bool = xax.field(
        False,
        help=(
            "Heavy eval only: when blockwise decoding is active, also generate an alternate candidate with the "
            "exact-last-token grouped transition flag toggled and choose the lower semantic NLL candidate under the "
            "teacher-forced stage-1 model. This is a cheap reranking heuristic for noisy grouped-prefix branches."
        ),
    )
    semantic_eval_compare_schedule_candidate: str | None = xax.field(
        None,
        help=(
            "Heavy eval only: optional alternate block decode schedule to compare against the configured grouped "
            "decode policy, scored with teacher-forced stage-1 semantic NLL. Example: '3,3,3,1,2'."
        ),
    )
    semantic_length_heuristic_frames_per_text_token: float | None = xax.field(
        None,
        help=(
            "Optional inference-only cap: max_frames ~= frames_per_text_token * num_text_tokens. "
            "This can reduce drift/repetition when EOS isn't reliable."
        ),
    )
    semantic_length_heuristic_factor: float = xax.field(
        1.25,
        help="Multiplier applied to semantic_length_heuristic_frames_per_text_token before capping max_frames.",
    )
    semantic_length_heuristic_min_frames: int = xax.field(
        1,
        help="Lower bound on the heuristic max_frames cap (only used when the heuristic is enabled).",
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
        model_vocab_size = int(AutoConfig.from_pretrained(config.llm_repo.value).vocab_size)
        tokenizer_vocab_size = len(self.tokenizer)
        self.base_vocab_size = model_vocab_size

        # Some Qwen tokenizers expose fewer live token ids than the model's padded
        # embedding matrix. Fill that gap first so all newly added TTS/audio tokens
        # start at ids >= model_vocab_size and therefore use the dedicated extra
        # embedding / LM-head path instead of accidentally landing inside the frozen
        # base-vocab range.
        if tokenizer_vocab_size < model_vocab_size:
            gap_tokens = [f"<|reserved_gap_{idx}|>" for idx in range(model_vocab_size - tokenizer_vocab_size)]
            self.tokenizer.add_tokens(gap_tokens)

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

    def _get_padding_lengths(self) -> tuple[int, int]:
        if hasattr(self, "_padding_lengths_cache"):
            return self._padding_lengths_cache

        path = self.dataset_cache_dir / "maximum_lengths.json"
        file_max_seq_len: int | None = None
        file_max_audio_frames: int | None = None
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                file_max_seq_len = int(data["max_seq_len"])
                file_max_audio_frames = int(data["max_audio_frames"])
            except Exception:
                logger.exception("Failed to read %s; will infer lengths from cached dataset", path)

        if "train" not in getattr(self, "dataset_functions", {}):
            fallback_lengths = self._infer_padding_lengths_from_tokenized_cache()
            if fallback_lengths is not None:
                self._padding_lengths_cache = fallback_lengths
                return self._padding_lengths_cache

        # Prefer the cached train dataset's shapes as the source of truth.
        try:
            train_ds = cast(Dataset, self.load_dataset("train"))
            ex0 = train_ds[0]
            ds_max_seq_len = int(np.asarray(ex0["codes"], dtype=np.int32).shape[0])
            ds_max_audio_frames = int(np.asarray(ex0["audio_codes"], dtype=np.int32).shape[0])
            if (file_max_seq_len != ds_max_seq_len) or (file_max_audio_frames != ds_max_audio_frames):
                self.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"max_seq_len": ds_max_seq_len, "max_audio_frames": ds_max_audio_frames}),
                    encoding="utf-8",
                )
            self._padding_lengths_cache = (ds_max_seq_len, ds_max_audio_frames)
            return self._padding_lengths_cache
        except Exception:
            if file_max_seq_len is None or file_max_audio_frames is None:
                raise
            self._padding_lengths_cache = (file_max_seq_len, file_max_audio_frames)
            return self._padding_lengths_cache

    def _infer_padding_lengths_from_tokenized_cache(self) -> tuple[int, int] | None:
        tokenized_root = self.dataset_cache_dir / "tokenized"
        candidate_names = ("v4", "v3", "default")
        text_source = self.config.text_source.strip().lower()

        for candidate_name in candidate_names:
            candidate_path = tokenized_root / candidate_name
            if not candidate_path.exists():
                continue

            logger.info("Inferring padding lengths from tokenized cache at %s", candidate_path)
            ds = cast(Dataset, load_from_disk(str(candidate_path)))
            audio_lengths = np.asarray([len(codes_t) for codes_t in ds["audio_codes"]], dtype=np.int32)
            max_audio_frames = int(np.percentile(audio_lengths, self.config.length_percentile * 100))

            columns = set(ds.column_names)
            if {"text_norm", "text_raw"}.issubset(columns):
                text_norm_lengths = np.asarray(
                    [len(self.tokenizer.encode(text)) for text in cast(list[str], ds["text_norm"])],
                    dtype=np.int32,
                )
                text_raw_lengths = np.asarray(
                    [len(self.tokenizer.encode(text)) for text in cast(list[str], ds["text_raw"])],
                    dtype=np.int32,
                )
                if text_source == "normalized":
                    code_lengths = text_norm_lengths + audio_lengths + 4
                elif text_source == "raw":
                    code_lengths = text_raw_lengths + audio_lengths + 4
                elif text_source == "both":
                    code_lengths = np.concatenate(
                        [
                            text_norm_lengths + audio_lengths + 4,
                            text_raw_lengths + audio_lengths + 4,
                        ]
                    )
                elif text_source == "both_plus_normalized":
                    code_lengths = np.concatenate(
                        [
                            text_norm_lengths + audio_lengths + 4,
                            text_norm_lengths + audio_lengths + 4,
                            text_raw_lengths + audio_lengths + 4,
                        ]
                    )
                elif text_source == "both_weighted":
                    norm_repeats = max(0, int(self.config.text_source_weighted_normalized_repeats))
                    raw_repeats = max(0, int(self.config.text_source_weighted_raw_repeats))
                    code_length_parts = [text_norm_lengths + audio_lengths + 4] * norm_repeats
                    code_length_parts.extend([text_raw_lengths + audio_lengths + 4] * raw_repeats)
                    if not code_length_parts:
                        raise ValueError("text_source=both_weighted requires at least one normalized or raw repeat.")
                    code_lengths = np.concatenate(code_length_parts)
                else:
                    raise ValueError(
                        "Invalid text_source: "
                        f"{self.config.text_source!r} (expected normalized, raw, both, both_plus_normalized, or both_weighted)"
                    )
            elif "text_tokens" in columns:
                text_lengths = np.asarray([len(tokens_s) for tokens_s in ds["text_tokens"]], dtype=np.int32)
                code_lengths = text_lengths + audio_lengths + 4
            else:
                continue

            max_seq_len = int(np.percentile(code_lengths, self.config.length_percentile * 100))
            self.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.dataset_cache_dir / "maximum_lengths.json"
            path.write_text(
                json.dumps({"max_seq_len": max_seq_len, "max_audio_frames": max_audio_frames}),
                encoding="utf-8",
            )
            return max_seq_len, max_audio_frames

        return None

    @property
    def max_seq_length(self) -> int:
        return self._get_padding_lengths()[0]

    @property
    def max_audio_frames(self) -> int:
        return self._get_padding_lengths()[1]

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
            semantic_future_prediction_steps=self.config.semantic_future_prediction_steps,
            semantic_non_ar=self.config.semantic_non_ar,
            semantic_non_ar_hidden_dim=self.config.semantic_non_ar_hidden_dim,
            semantic_non_ar_num_heads=self.config.semantic_non_ar_num_heads,
            semantic_non_ar_num_layers=self.config.semantic_non_ar_num_layers,
            semantic_non_ar_mlp_dim=self.config.semantic_non_ar_mlp_dim,
            whisper_repo_id=self.config.whisper_repo_id,
            residual_head_dim=self.config.residual_head_dim,
            residual_num_heads=self.config.residual_num_heads,
            residual_num_layers=self.config.residual_num_layers,
            residual_mlp_dim=self.config.residual_mlp_dim,
            key=mimi_key,
        )

    @override
    def _get_dataset_hash(self, name: str) -> str:
        """Extend dataset hashing with config-dependent fields.

        XAX dataset caching hashes dataset functions (and their dependencies)
        but does not include runtime config values. This task's `train` (and
        optionally `unpadded`) datasets depend on fields like `length_percentile`
        and `text_source`, so we incorporate a small hash of those fields into
        the cache key to prevent stale re-use across experiments.
        """
        base_hash = super()._get_dataset_hash(name)
        if name not in {"train", "unpadded"}:
            return base_hash

        payload = {
            "length_percentile": float(self.config.length_percentile),
            "llm_repo": str(self.config.llm_repo),
            "text_source": str(self.config.text_source),
            "text_source_weighted_normalized_repeats": int(self.config.text_source_weighted_normalized_repeats),
            "text_source_weighted_raw_repeats": int(self.config.text_source_weighted_raw_repeats),
            "base_vocab_size": int(self.base_vocab_size),
            "first_q0_id": int(self.first_q0_id),
        }
        cfg_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        return f"{base_hash}_{cfg_hash}"

    @override
    def get_trainable_filter_spec(self, model: FullTTSModel) -> FullTTSModel:
        # Stage 1: LoRA + extra token embeddings/heads + optional non-AR semantic upsampler.
        llm_spec = xax.lora_filter_spec(model.semantic.llm)
        extra_embed_spec = jax.tree.map(lambda _: True, llm_spec.extra_embed)
        llm_spec = eqx.tree_at(lambda m: m.extra_embed, llm_spec, extra_embed_spec)
        semantic_spec = SemanticTTSModel(
            llm=llm_spec,
            future_q0_heads=(
                None
                if model.semantic.future_q0_heads is None
                else tuple(jax.tree.map(eqx.is_inexact_array, head) for head in model.semantic.future_q0_heads)
            ),
            non_ar_in_proj=(
                None if model.semantic.non_ar_in_proj is None else jax.tree.map(eqx.is_inexact_array, model.semantic.non_ar_in_proj)
            ),
            non_ar_stack=(
                None if model.semantic.non_ar_stack is None else jax.tree.map(eqx.is_inexact_array, model.semantic.non_ar_stack)
            ),
            non_ar_norm=(
                None if model.semantic.non_ar_norm is None else jax.tree.map(eqx.is_inexact_array, model.semantic.non_ar_norm)
            ),
            non_ar_out_proj=(
                None if model.semantic.non_ar_out_proj is None else jax.tree.map(eqx.is_inexact_array, model.semantic.non_ar_out_proj)
            ),
            non_ar_to_llm_proj=(
                None
                if model.semantic.non_ar_to_llm_proj is None
                else jax.tree.map(eqx.is_inexact_array, model.semantic.non_ar_to_llm_proj)
            ),
        )

        # Stage 2: Train all residual params.
        residual_spec = jax.tree.map(eqx.is_inexact_array, model.residual)

        # Mimi + Whisper: frozen (or absent).
        mimi_spec = None if model.mimi is None else jax.tree.map(lambda _: False, model.mimi)
        whisper_spec = (
            None if model.whisper_transcriber is None else jax.tree.map(lambda _: False, model.whisper_transcriber)
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

    def _generate_semantic_tokens_blockwise_greedy(
        self,
        model: FullTTSModel,
        *,
        prompt_tokens_s: Array,
        max_new_tokens: int | Array,
        prompt_len: int | Array | None = None,
        initial_generated_count: int | Array = 0,
        exact_last_token_override: bool | None = None,
        block_decode_schedule_override: str | None = None,
    ) -> tuple[Array, Array]:
        llm = model.semantic.llm
        future_heads = model.semantic.future_q0_heads
        if future_heads is None or len(future_heads) == 0:
            raise ValueError("Blockwise semantic generation requires Stage-1 future-Q0 heads.")
        if prompt_len is None and max_new_tokens <= 0:
            return prompt_tokens_s, jnp.asarray(prompt_tokens_s.shape[0], dtype=jnp.int32)

        exact_last_token = (
            bool(self.config.semantic_block_decode_exact_last_token)
            if exact_last_token_override is None
            else bool(exact_last_token_override)
        )
        max_supported_block_decode_size = len(future_heads) + (2 if exact_last_token else 1)
        fixed_block_decode_size = max(1, min(int(self.config.semantic_block_decode_size), max_supported_block_decode_size))
        raw_block_decode_schedule = (
            self.config.semantic_block_decode_schedule
            if block_decode_schedule_override is None
            else block_decode_schedule_override
        )
        schedule_sizes: list[int] = []
        if raw_block_decode_schedule is not None and raw_block_decode_schedule.strip():
            try:
                schedule_sizes = [int(part.strip()) for part in raw_block_decode_schedule.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid semantic_block_decode_schedule: {raw_block_decode_schedule!r}"
                ) from exc
            if not schedule_sizes or any(size <= 0 for size in schedule_sizes):
                raise ValueError(
                    "semantic_block_decode_schedule must contain one or more positive integers, e.g. '3,3,1,2,2,1'."
                )
            schedule_sizes = [max(1, min(size, max_supported_block_decode_size)) for size in schedule_sizes]
        if not schedule_sizes and fixed_block_decode_size <= 1:
            end_pos = prompt_tokens_s.shape[0] if prompt_len is None else prompt_len
            return prompt_tokens_s, jnp.asarray(end_pos, dtype=jnp.int32)
        if self.config.semantic_gen_temperature != 0.0 or self.config.semantic_gen_top_p < 1.0:
            raise ValueError("Blockwise semantic generation currently only supports greedy decoding.")

        max_block_decode_size = max([fixed_block_decode_size, *schedule_sizes]) if schedule_sizes else fixed_block_decode_size
        schedule_len = len(schedule_sizes)
        schedule_sizes_t = jnp.asarray(schedule_sizes if schedule_sizes else [1], dtype=jnp.int32)

        audio_lm_head = self._audio_lm_head(llm)
        base_vocab_size = llm.config.vocab_size
        q0_min_id = self.first_q0_id
        audio_end_extra_id = self.audio_end_id - base_vocab_size
        future_head_count = max(0, max_block_decode_size - (2 if exact_last_token else 1))
        used_future_heads = future_heads[:future_head_count]
        max_grouped_tokens = jnp.asarray(int(self.config.semantic_block_decode_max_grouped_tokens), dtype=jnp.int32)
        fixed_block_decode_size_t = jnp.asarray(fixed_block_decode_size, dtype=jnp.int32)
        one_t = jnp.asarray(1, dtype=jnp.int32)
        initial_generated_count_t = jnp.asarray(initial_generated_count, dtype=jnp.int32)

        if prompt_len is None:
            initial_len = int(prompt_tokens_s.shape[0])
            max_len = initial_len + int(max_new_tokens)
            stop_pos = jnp.asarray(max_len, dtype=jnp.int32)
            cur_pos_init = jnp.asarray(initial_len, dtype=jnp.int32)
            padded_tokens = jnp.zeros((max_len,), dtype=jnp.int32)
            padded_tokens = padded_tokens.at[:initial_len].set(prompt_tokens_s)

            caches = llm.init_cache(max_len, dtype=llm.embed.weight.dtype)
            prompt_hidden_sd, caches = llm.forward_hidden(prompt_tokens_s, caches=caches)
            current_hidden_d = prompt_hidden_sd[-1]
        else:
            max_len = int(prompt_tokens_s.shape[0])
            prompt_len_t = jnp.asarray(prompt_len, dtype=jnp.int32)
            max_new_tokens_t = jnp.asarray(max_new_tokens, dtype=jnp.int32)
            stop_pos = jnp.minimum(prompt_len_t + max_new_tokens_t, jnp.asarray(max_len, dtype=jnp.int32))
            cur_pos_init = prompt_len_t
            positions_s = jnp.arange(max_len, dtype=jnp.int32)
            padded_tokens = jnp.where(positions_s < prompt_len_t, prompt_tokens_s, 0)

            caches = llm.init_cache(max_len, dtype=llm.embed.weight.dtype)
            current_hidden_d = jnp.zeros((llm.config.embed_dim,), dtype=llm.embed.weight.dtype)

            def prime_prefix(
                carry: tuple[list[xax.TransformerBlockCache], Array],
                idx: Array,
            ) -> tuple[tuple[list[xax.TransformerBlockCache], Array], None]:
                inner_caches, inner_hidden_d = carry
                token_1 = jax.lax.dynamic_slice(prompt_tokens_s, (idx,), (1,))

                def do_feed(
                    inner_carry: tuple[list[xax.TransformerBlockCache], Array],
                ) -> tuple[list[xax.TransformerBlockCache], Array]:
                    token_hidden_1d, next_caches = llm.forward_hidden(token_1, caches=inner_carry[0])
                    return next_caches, token_hidden_1d[0]

                next_caches, next_hidden_d = jax.lax.cond(
                    idx < prompt_len_t,
                    do_feed,
                    lambda inner_carry: inner_carry,
                    (inner_caches, inner_hidden_d),
                )
                return (next_caches, next_hidden_d), None

            (caches, current_hidden_d), _ = jax.lax.scan(
                prime_prefix,
                (caches, current_hidden_d),
                jnp.arange(max_len, dtype=jnp.int32),
            )

        min_new_tokens_before_eos = int(self.config.semantic_gen_min_new_tokens)
        min_logit_value = jnp.asarray(jnp.finfo(jnp.float32).min, dtype=jnp.float32)

        def cond_fn(state: tuple[Array, Array, Array, Array, list[xax.TransformerBlockCache], Array, Array]) -> Array:
            _, cur_pos, _, done, _, _, _ = state
            return (cur_pos < stop_pos) & ~done

        def body_fn(
            state: tuple[Array, Array, Array, Array, list[xax.TransformerBlockCache], Array, Array],
        ) -> tuple[Array, Array, Array, Array, list[xax.TransformerBlockCache], Array, Array]:
            tokens_t, cur_pos, generated_count, done, caches, current_hidden_d, decode_step = state

            next_logits_v = audio_lm_head(current_hidden_d[None, :].astype(jnp.float32))[0]
            next_logits_v = jax.lax.cond(
                (self.audio_end_id >= 0) & (generated_count < min_new_tokens_before_eos),
                lambda logits_v: logits_v.at[audio_end_extra_id].set(min_logit_value),
                lambda logits_v: logits_v,
                next_logits_v,
            )
            first_extra_id = jnp.argmax(next_logits_v).astype(jnp.int32)
            first_token = first_extra_id + base_vocab_size
            first_is_eos = first_token == self.audio_end_id

            future_block_tokens = []
            for head in used_future_heads:
                future_logits_v = apply_linear(current_hidden_d[None, :].astype(jnp.float32), head)[0]
                future_code = jnp.argmax(future_logits_v).astype(jnp.int32)
                future_block_tokens.append(future_code + q0_min_id)
            if future_block_tokens:
                future_tokens_t = jnp.stack(future_block_tokens)
            else:
                future_tokens_t = jnp.zeros((0,), dtype=jnp.int32)

            if schedule_len > 0:
                schedule_idx = jnp.minimum(decode_step, jnp.asarray(schedule_len - 1, dtype=jnp.int32))
                active_block_decode_size = jax.lax.cond(
                    decode_step < schedule_len,
                    lambda idx: schedule_sizes_t[idx],
                    lambda _: one_t,
                    schedule_idx,
                )
            else:
                active_block_decode_size = jnp.where(
                    (max_grouped_tokens <= 0) | (generated_count < max_grouped_tokens),
                    fixed_block_decode_size_t,
                    one_t,
                )

            use_grouped = active_block_decode_size > 1
            remaining_after_first = jnp.maximum(stop_pos - (cur_pos + 1), 0)
            grouped_extra_tokens = jnp.minimum(jnp.maximum(active_block_decode_size - 1, 0), remaining_after_first)
            max_extra_tokens = jnp.where(use_grouped & ~first_is_eos, grouped_extra_tokens, 0)
            block_len = one_t + max_extra_tokens

            def feed_token(
                token: Array,
                carry: tuple[list[xax.TransformerBlockCache], Array],
            ) -> tuple[list[xax.TransformerBlockCache], Array]:
                inner_caches, _ = carry
                hidden_1d, next_caches = llm.forward_hidden(token[None], caches=inner_caches)
                return next_caches, hidden_1d[0]

            if exact_last_token:
                block_tokens_t = jnp.full((max_block_decode_size,), q0_min_id, dtype=jnp.int32)
                block_tokens_t = block_tokens_t.at[0].set(first_token)

                step_caches, step_hidden_d = feed_token(first_token, (caches, current_hidden_d))

                speculative_count = jnp.maximum(block_len - 2, 0)
                for idx in range(max_block_decode_size - 2):
                    speculative_token = jnp.where(
                        jnp.asarray(idx, dtype=jnp.int32) < speculative_count,
                        future_tokens_t[idx],
                        q0_min_id,
                    )
                    block_tokens_t = block_tokens_t.at[idx + 1].set(speculative_token)
                    step_caches, step_hidden_d = jax.lax.cond(
                        jnp.asarray(idx, dtype=jnp.int32) < speculative_count,
                        lambda carry: feed_token(speculative_token, carry),
                        lambda carry: carry,
                        (step_caches, step_hidden_d),
                    )

                def compute_final_exact_token(hidden_d: Array) -> tuple[Array, Array]:
                    final_logits_v = audio_lm_head(hidden_d[None, :].astype(jnp.float32))[0]
                    final_logits_v = jax.lax.cond(
                        (self.audio_end_id >= 0) & ((generated_count + block_len - 1) < min_new_tokens_before_eos),
                        lambda logits_v: logits_v.at[audio_end_extra_id].set(min_logit_value),
                        lambda logits_v: logits_v,
                        final_logits_v,
                    )
                    token = jnp.argmax(final_logits_v).astype(jnp.int32) + base_vocab_size
                    return token, token == self.audio_end_id

                final_token, final_is_eos = jax.lax.cond(
                    block_len > 1,
                    compute_final_exact_token,
                    lambda _: (jnp.asarray(q0_min_id, dtype=jnp.int32), jnp.asarray(False)),
                    step_hidden_d,
                )
                block_tokens_t = jax.lax.cond(
                    block_len > 1,
                    lambda bt: bt.at[block_len - 1].set(final_token),
                    lambda bt: bt,
                    block_tokens_t,
                )
                tokens_t = jax.lax.dynamic_update_slice(tokens_t, block_tokens_t, (cur_pos,))
                step_caches, step_hidden_d = jax.lax.cond(
                    block_len > 1,
                    lambda carry: feed_token(final_token, carry),
                    lambda carry: carry,
                    (step_caches, step_hidden_d),
                )

                return (
                    tokens_t,
                    cur_pos + block_len,
                    generated_count + block_len,
                    done | first_is_eos | final_is_eos,
                    step_caches,
                    step_hidden_d,
                    decode_step + 1,
                )

            block_tokens_t = jnp.full((max_block_decode_size,), q0_min_id, dtype=jnp.int32)
            block_tokens_t = block_tokens_t.at[0].set(first_token)
            for idx in range(max_block_decode_size - 1):
                block_tokens_t = block_tokens_t.at[idx + 1].set(
                    jnp.where(
                        (jnp.asarray(idx, dtype=jnp.int32) < max_extra_tokens),
                        future_tokens_t[idx],
                        q0_min_id,
                    )
                )
            tokens_t = jax.lax.dynamic_update_slice(tokens_t, block_tokens_t, (cur_pos,))

            def feed_one(
                carry: tuple[list[xax.TransformerBlockCache], Array],
                idx: Array,
            ) -> tuple[tuple[list[xax.TransformerBlockCache], Array], None]:
                caches, current_hidden_d = carry

                def do_feed(
                    inner_carry: tuple[list[xax.TransformerBlockCache], Array],
                ) -> tuple[list[xax.TransformerBlockCache], Array]:
                    inner_caches, _ = inner_carry
                    token_1 = jax.lax.dynamic_slice(block_tokens_t, (idx,), (1,))
                    hidden_1d, next_caches = llm.forward_hidden(token_1, caches=inner_caches)
                    return next_caches, hidden_1d[0]

                next_caches, next_hidden_d = jax.lax.cond(
                    idx < block_len,
                    do_feed,
                    lambda inner_carry: inner_carry,
                    (caches, current_hidden_d),
                )
                return (next_caches, next_hidden_d), None

            (caches, current_hidden_d), _ = jax.lax.scan(
                feed_one,
                (caches, current_hidden_d),
                jnp.arange(max_block_decode_size, dtype=jnp.int32),
            )

            return (
                tokens_t,
                cur_pos + block_len,
                generated_count + block_len,
                done | first_is_eos,
                caches,
                current_hidden_d,
                decode_step + 1,
            )

        init_state = (
            padded_tokens,
            cur_pos_init,
            initial_generated_count_t,
            jnp.asarray(False),
            caches,
            current_hidden_d,
            jnp.asarray(0, dtype=jnp.int32),
        )
        final_tokens, final_pos, _, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return final_tokens, final_pos

    def _semantic_packed_prefix_hidden(
        self,
        tokens_s: Array,
        prefix_hidden_sd: Array,
    ) -> tuple[Array, Array]:
        seq_len = int(tokens_s.shape[0])
        positions_s = jnp.arange(seq_len, dtype=jnp.int32)
        audio_start_mask_s = tokens_s == self.audio_start_id
        audio_start_idx = jnp.where(audio_start_mask_s.any(), jnp.argmax(audio_start_mask_s), seq_len - 1)
        prefix_mask_s = positions_s <= audio_start_idx

        sort_key_s = jnp.where(prefix_mask_s, positions_s, positions_s + seq_len)
        sort_idx_s = jnp.argsort(sort_key_s)
        packed_prefix_sd = prefix_hidden_sd[sort_idx_s]
        prefix_count = prefix_mask_s.astype(jnp.int32).sum()
        packed_valid_s = jnp.arange(seq_len, dtype=jnp.int32) < prefix_count
        packed_prefix_sd = jnp.where(packed_valid_s[:, None], packed_prefix_sd, 0)
        return packed_prefix_sd, packed_valid_s

    def _semantic_text_prefix_interpolation(
        self,
        tokens_s: Array,
        prefix_hidden_sd: Array,
        *,
        total_frames: int,
    ) -> Array:
        packed_prefix_sd, packed_valid_s = self._semantic_packed_prefix_hidden(tokens_s, prefix_hidden_sd)
        prefix_count = packed_valid_s.astype(jnp.int32).sum()
        safe_prefix_count = jnp.maximum(prefix_count, 1)
        frame_idx_t = (jnp.arange(total_frames, dtype=jnp.int32) * safe_prefix_count) // jnp.maximum(total_frames, 1)
        frame_idx_t = jnp.clip(frame_idx_t, 0, safe_prefix_count - 1)
        return packed_prefix_sd[frame_idx_t]

    def _semantic_inference_max_frames(self, prompt_len: int) -> int:
        max_frames = self.max_audio_frames
        if self.config.semantic_length_heuristic_frames_per_text_token is not None:
            text_token_count = max(0, prompt_len - 3)
            pred_frames = int(
                round(
                    text_token_count
                    * float(self.config.semantic_length_heuristic_frames_per_text_token)
                    * float(self.config.semantic_length_heuristic_factor)
                )
            )
            pred_frames = max(pred_frames, int(self.config.semantic_length_heuristic_min_frames))
            max_frames = min(max_frames, pred_frames)
        return max(1, max_frames)

    def _semantic_non_ar_forward(
        self,
        model: FullTTSModel,
        tokens_s: Array,
        *,
        total_frames: int,
    ) -> tuple[Array, Array, Array]:
        semantic = model.semantic
        if not semantic.use_non_ar:
            raise ValueError("Non-autoregressive semantic path is not enabled.")
        assert semantic.non_ar_in_proj is not None
        assert semantic.non_ar_stack is not None
        assert semantic.non_ar_norm is not None
        assert semantic.non_ar_out_proj is not None
        assert semantic.non_ar_to_llm_proj is not None

        prefix_hidden_sd = semantic.llm.forward_hidden(tokens_s)
        packed_prefix_sd, packed_valid_s = self._semantic_packed_prefix_hidden(tokens_s, prefix_hidden_sd)
        frame_seed_td = self._semantic_text_prefix_interpolation(tokens_s, prefix_hidden_sd, total_frames=total_frames)

        packed_prefix_hd = jax.vmap(semantic.non_ar_in_proj)(packed_prefix_sd)
        frame_seed_hd = jax.vmap(semantic.non_ar_in_proj)(frame_seed_td)
        combined_inputs_ud = jnp.concatenate([packed_prefix_hd, frame_seed_hd], axis=0)
        combined_valid_u = jnp.concatenate(
            [packed_valid_s, jnp.ones((total_frames,), dtype=jnp.bool_)],
            axis=0,
        )
        attn_mask_11uu = (combined_valid_u[:, None] & combined_valid_u[None, :])[None, None, :, :]
        combined_hidden_ud, _ = semantic.non_ar_stack.forward(combined_inputs_ud, mask=attn_mask_11uu)
        combined_hidden_ud = jax.vmap(semantic.non_ar_norm)(combined_hidden_ud)
        frame_hidden_td = combined_hidden_ud[packed_prefix_hd.shape[0] :]
        q0_logits_tv = jax.vmap(semantic.non_ar_out_proj)(frame_hidden_td.astype(jnp.float32))
        llm_hidden_td = jax.vmap(semantic.non_ar_to_llm_proj)(frame_hidden_td)
        return prefix_hidden_sd, q0_logits_tv, llm_hidden_td

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
        bsz = int(codes_bs.shape[0])

        # Stage 1: semantic Q0 prediction (prefix-LM style, loss on audio segment only).
        tokens_bs, targets_bs = codes_bs[:, :-1], codes_bs[:, 1:]
        llm = model.semantic.llm
        base_vocab_size = llm.config.vocab_size

        q0_min_id = self.first_q0_id
        q0_max_id = self.first_q0_id + CODEBOOK_SIZE
        q0_mask_bs = (targets_bs >= q0_min_id) & (targets_bs < q0_max_id)
        eos_mask_bs = targets_bs == self.audio_end_id

        # Optional text prefix loss (kept off by default).
        text_end_mask_bs = codes_bs == self.text_end_id
        has_text_end_b = jnp.any(text_end_mask_bs, axis=1)
        text_end_idx_b = jnp.argmax(text_end_mask_bs, axis=1)
        target_pos_bs = jnp.broadcast_to(jnp.arange(targets_bs.shape[1])[None, :], targets_bs.shape)
        text_mask_bs = (target_pos_bs < text_end_idx_b[:, None]) & has_text_end_b[:, None]
        text_mask_bs = text_mask_bs & (targets_bs < base_vocab_size)

        audio_lm_head = self._audio_lm_head(llm)
        raw_block_decode_schedule = self.config.semantic_block_decode_schedule
        parsed_block_decode_schedule: list[int] = []
        if raw_block_decode_schedule is not None and raw_block_decode_schedule.strip():
            try:
                parsed_block_decode_schedule = [
                    int(part.strip()) for part in raw_block_decode_schedule.split(",") if part.strip()
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid semantic_block_decode_schedule: {raw_block_decode_schedule!r}"
                ) from exc
            if not parsed_block_decode_schedule or any(size <= 0 for size in parsed_block_decode_schedule):
                raise ValueError(
                    "semantic_block_decode_schedule must contain one or more positive integers, e.g. '3,3,1,2,2,1'."
                )
            future_head_count = len(model.semantic.future_q0_heads) if model.semantic.future_q0_heads is not None else 0
            max_supported_block_decode_size = future_head_count + (2 if self.config.semantic_block_decode_exact_last_token else 1)
            parsed_block_decode_schedule = [
                max(1, min(size, max_supported_block_decode_size)) for size in parsed_block_decode_schedule
            ]
        has_block_decode_schedule = bool(parsed_block_decode_schedule)
        schedule_sizes_t = jnp.asarray(parsed_block_decode_schedule if parsed_block_decode_schedule else [1], dtype=jnp.int32)
        schedule_cum_sizes_t = jnp.cumsum(schedule_sizes_t)
        schedule_len_t = jnp.asarray(len(parsed_block_decode_schedule), dtype=jnp.int32)
        schedule_total_tokens_t = jnp.asarray(sum(parsed_block_decode_schedule), dtype=jnp.int32)

        use_semantic_self_condition = self.config.semantic_self_condition_prob > 0 and not heavy
        use_blockwise_self_condition = (
            use_semantic_self_condition
            and self.config.semantic_self_condition_use_inference_policy
            and model.semantic.future_q0_heads is not None
            and len(model.semantic.future_q0_heads) > 0
            and not has_block_decode_schedule
            and self.config.semantic_block_decode_size > 1
        )
        use_blockwise_self_condition_prefix_only = (
            use_blockwise_self_condition and self.config.semantic_self_condition_grouped_prefix_only
        )
        use_blockwise_self_condition_early_prefix = (
            use_blockwise_self_condition and self.config.semantic_self_condition_match_early_grouped_prefix
        )
        use_schedule_self_condition_early_prefix = (
            use_semantic_self_condition
            and self.config.semantic_self_condition_use_inference_policy
            and model.semantic.future_q0_heads is not None
            and len(model.semantic.future_q0_heads) > 0
            and has_block_decode_schedule
            and self.config.semantic_self_condition_match_early_grouped_prefix
        )
        use_early_prefix_inference_policy_match = (
            use_blockwise_self_condition_early_prefix or use_schedule_self_condition_early_prefix
        )

        if model.semantic.use_non_ar:
            max_frames = int(audio_codes_btf.shape[1])

            def compute_stage1_terms_non_ar(
                sample_tokens_s: Array,
                sample_audio_codes_tf: Array,
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
                _, q0_logits_tv, frame_llm_hidden_td = self._semantic_non_ar_forward(
                    model,
                    sample_tokens_s,
                    total_frames=max_frames,
                )
                q0_targets_t = sample_audio_codes_tf[:, 0]
                frame_mask_t = q0_targets_t != AUDIO_PAD_TOKEN_ID
                q0_targets_t = jnp.where(frame_mask_t, q0_targets_t, 0)

                q0_loss_t = optax.softmax_cross_entropy_with_integer_labels(q0_logits_tv, q0_targets_t)
                denom = jnp.maximum(frame_mask_t.astype(jnp.float32).sum(), 1.0)
                q0_loss = jnp.where(frame_mask_t, q0_loss_t, 0.0).sum() / denom

                q0_pred_t = jnp.argmax(q0_logits_tv, axis=-1).astype(jnp.int32)
                q0_acc = ((q0_pred_t == q0_targets_t) & frame_mask_t).astype(jnp.float32).sum() / denom

                seq_len = int(sample_tokens_s.shape[0])
                positions_s = jnp.arange(seq_len, dtype=jnp.int32)
                audio_start_mask_s = sample_tokens_s == self.audio_start_id
                audio_start_idx = jnp.where(audio_start_mask_s.any(), jnp.argmax(audio_start_mask_s), seq_len - 1)
                q0_start = audio_start_idx + 1
                frame_pos_s = positions_s - q0_start
                frame_valid_s = (frame_pos_s >= 0) & (frame_pos_s < max_frames)
                frame_idx_s = jnp.clip(frame_pos_s, 0, max_frames - 1)
                hidden_sd = jnp.where(frame_valid_s[:, None], frame_llm_hidden_td[frame_idx_s], 0)

                zero = jnp.array(0.0, dtype=jnp.float32)
                return q0_loss, q0_acc, q0_loss, q0_acc, zero, zero, zero, zero, hidden_sd

            (
                stage1_loss_b,
                stage1_acc_b,
                stage1_q0_loss_b,
                stage1_q0_acc_b,
                stage1_eos_loss_b,
                stage1_eos_acc_b,
                text_loss_b,
                text_acc_b,
                hidden_bsd,
            ) = jax.vmap(compute_stage1_terms_non_ar, in_axes=(0, 0))(tokens_bs, audio_codes_btf)
        else:

            def compute_stage1_terms(
                sample_tokens_s: Array,
                sample_targets_s: Array,
                sample_q0_mask_s: Array,
                sample_eos_mask_s: Array,
                sample_text_mask_s: Array,
                sample_key: PRNGKeyArray,
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
                work_key = sample_key
                q0_input_mask_s = (sample_tokens_s >= q0_min_id) & (sample_tokens_s < q0_max_id)
                q0_input_count = q0_input_mask_s.astype(jnp.int32).sum()

                if self.config.semantic_q0_corruption_prob > 0:
                    work_key, mask_key, codes_key = jax.random.split(work_key, 3)
                    corrupt_mask_s = (
                        jax.random.uniform(mask_key, sample_tokens_s.shape) < self.config.semantic_q0_corruption_prob
                    ) & q0_input_mask_s
                    random_q0_s = jax.random.randint(codes_key, sample_tokens_s.shape, q0_min_id, q0_max_id)
                    sample_tokens_s = jnp.where(corrupt_mask_s, random_q0_s, sample_tokens_s)

                if use_semantic_self_condition:
                    work_key, activate_key, keep_key, policy_key = jax.random.split(work_key, 4)
                    do_self_condition = jax.random.uniform(activate_key, ()) < self.config.semantic_self_condition_prob

                    def self_condition(tokens_s: Array) -> Array:
                        safe_q0_input_count = jnp.maximum(q0_input_count, 1)
                        raw_keep_count = jax.random.randint(keep_key, (), 0, safe_q0_input_count + 1)
                        q0_rank_s = jnp.cumsum(q0_input_mask_s.astype(jnp.int32), axis=0)
                        use_inference_policy = (
                            jax.random.uniform(policy_key, ()) < self.config.semantic_self_condition_inference_policy_prob
                            if (use_blockwise_self_condition or use_schedule_self_condition_early_prefix)
                            else jnp.asarray(False)
                        )

                        first_pass_hidden_sd = llm.forward_hidden(tokens_s)
                        first_pass_logits_sv = audio_lm_head(first_pass_hidden_sd)
                        first_pass_pred_s = jnp.argmax(first_pass_logits_sv, axis=-1).astype(jnp.int32) + base_vocab_size
                        pred_input_s = jnp.concatenate([tokens_s[:1], first_pass_pred_s[:-1]], axis=0)
                        valid_pred_mask_s = (pred_input_s >= q0_min_id) & (pred_input_s < q0_max_id)

                        def cheap_replace(keep_count: Array) -> Array:
                            replace_suffix_s = q0_input_mask_s & (q0_rank_s > keep_count)
                            replace_mask_s = replace_suffix_s & valid_pred_mask_s
                            return jnp.where(replace_mask_s, pred_input_s, tokens_s)

                        if use_early_prefix_inference_policy_match:
                            seq_len = int(tokens_s.shape[0])
                            audio_start_mask_s = tokens_s == self.audio_start_id
                            audio_start_idx = jnp.where(
                                audio_start_mask_s.any(), jnp.argmax(audio_start_mask_s), seq_len - 1
                            )
                            prompt_len = audio_start_idx + 1
                            max_grouped_tokens_t = jnp.asarray(
                                int(self.config.semantic_block_decode_max_grouped_tokens),
                                dtype=jnp.int32,
                            )
                            grouped_new_tokens = (
                                jnp.minimum(q0_input_count, schedule_total_tokens_t)
                                if has_block_decode_schedule
                                else jax.lax.cond(
                                    max_grouped_tokens_t > 0,
                                    lambda limit: jnp.minimum(q0_input_count, limit),
                                    lambda _: q0_input_count,
                                    max_grouped_tokens_t,
                                )
                            )
                            replace_prefix_s = q0_input_mask_s & (q0_rank_s <= grouped_new_tokens)
                            step_size_s = jnp.ones_like(q0_rank_s, dtype=jnp.int32)
                            step_rank_s = q0_rank_s
                            step_source_rank_s = jnp.maximum(q0_rank_s - 1, 0)
                            if has_block_decode_schedule:
                                step_idx_s = jnp.sum(
                                    q0_rank_s[:, None] > schedule_cum_sizes_t[None, :],
                                    axis=1,
                                    dtype=jnp.int32,
                                )
                                clipped_step_idx_s = jnp.minimum(step_idx_s, jnp.maximum(schedule_len_t - 1, 0))
                                step_size_s = schedule_sizes_t[clipped_step_idx_s]
                                prev_cum_s = jnp.where(
                                    step_idx_s > 0,
                                    schedule_cum_sizes_t[jnp.maximum(clipped_step_idx_s - 1, 0)],
                                    0,
                                )
                                step_rank_s = q0_rank_s - prev_cum_s
                                step_source_rank_s = prev_cum_s
                            elif self.config.semantic_block_decode_size > 1:
                                block_size_t = jnp.asarray(max(1, int(self.config.semantic_block_decode_size)), dtype=jnp.int32)
                                step_size_s = jnp.where(q0_rank_s <= grouped_new_tokens, block_size_t, 1)
                                step_source_rank_s = ((jnp.maximum(q0_rank_s - 1, 0) // block_size_t) * block_size_t).astype(jnp.int32)
                                step_rank_s = q0_rank_s - step_source_rank_s

                            if self.config.semantic_self_condition_match_grouped_future_slots_only:
                                replace_prefix_s = replace_prefix_s & (step_size_s > 1)
                                if self.config.semantic_block_decode_exact_last_token:
                                    replace_prefix_s = replace_prefix_s & (step_rank_s > 1) & (step_rank_s < step_size_s)
                                else:
                                    replace_prefix_s = replace_prefix_s & (step_rank_s > 1)

                            def cheap_prefix_replace(_: Array) -> Array:
                                replace_mask_s = replace_prefix_s & valid_pred_mask_s
                                return jnp.where(replace_mask_s, pred_input_s, tokens_s)

                            def expensive_prefix_replace(_: Array) -> Array:
                                generated_tokens_s, _ = self._generate_semantic_tokens_blockwise_greedy(
                                    model,
                                    prompt_tokens_s=tokens_s,
                                    prompt_len=prompt_len,
                                    max_new_tokens=grouped_new_tokens,
                                    initial_generated_count=0,
                                )
                                grouped_valid_pred_mask_s = (
                                    (generated_tokens_s >= q0_min_id) & (generated_tokens_s < q0_max_id)
                                )
                                replace_mask_s = replace_prefix_s & grouped_valid_pred_mask_s
                                return jnp.where(replace_mask_s, generated_tokens_s, tokens_s)

                            return jax.lax.cond(
                                use_inference_policy,
                                expensive_prefix_replace,
                                cheap_prefix_replace,
                                jnp.asarray(0, dtype=jnp.int32),
                            )

                        if use_blockwise_self_condition:
                            block_size_t = jnp.asarray(max(1, int(self.config.semantic_block_decode_size)), dtype=jnp.int32)
                            max_grouped_tokens_t = jnp.asarray(
                                int(self.config.semantic_block_decode_max_grouped_tokens),
                                dtype=jnp.int32,
                            )
                            snapped_keep_count = jax.lax.cond(
                                max_grouped_tokens_t > 0,
                                lambda raw: jax.lax.cond(
                                    raw < max_grouped_tokens_t,
                                    lambda r: (r // block_size_t) * block_size_t,
                                    lambda r: r,
                                    raw,
                                ),
                                lambda raw: (raw // block_size_t) * block_size_t,
                                raw_keep_count,
                            )
                            snapped_keep_count = jnp.minimum(snapped_keep_count, q0_input_count)

                            seq_len = int(tokens_s.shape[0])
                            audio_start_mask_s = tokens_s == self.audio_start_id
                            audio_start_idx = jnp.where(audio_start_mask_s.any(), jnp.argmax(audio_start_mask_s), seq_len - 1)
                            prompt_len = audio_start_idx + 1 + snapped_keep_count

                            def expensive_replace(_: Array) -> Array:
                                if use_blockwise_self_condition_prefix_only:
                                    grouped_new_tokens = q0_input_count - snapped_keep_count
                                    grouped_new_tokens = jax.lax.cond(
                                        max_grouped_tokens_t > 0,
                                        lambda remaining: jnp.minimum(
                                            remaining,
                                            jnp.maximum(max_grouped_tokens_t - snapped_keep_count, 0),
                                        ),
                                        lambda remaining: remaining,
                                        grouped_new_tokens,
                                    )
                                    generated_tokens_s, _ = self._generate_semantic_tokens_blockwise_greedy(
                                        model,
                                        prompt_tokens_s=tokens_s,
                                        prompt_len=prompt_len,
                                        max_new_tokens=grouped_new_tokens,
                                        initial_generated_count=snapped_keep_count,
                                    )
                                    grouped_valid_pred_mask_s = (
                                        (generated_tokens_s >= q0_min_id) & (generated_tokens_s < q0_max_id)
                                    )
                                    hybrid_pred_input_s = jnp.where(grouped_valid_pred_mask_s, generated_tokens_s, pred_input_s)
                                    hybrid_valid_pred_mask_s = (
                                        (hybrid_pred_input_s >= q0_min_id) & (hybrid_pred_input_s < q0_max_id)
                                    )
                                    replace_suffix_s = q0_input_mask_s & (q0_rank_s > snapped_keep_count)
                                    replace_mask_s = replace_suffix_s & hybrid_valid_pred_mask_s
                                    return jnp.where(replace_mask_s, hybrid_pred_input_s, tokens_s)

                                generated_tokens_s, _ = self._generate_semantic_tokens_blockwise_greedy(
                                    model,
                                    prompt_tokens_s=tokens_s,
                                    prompt_len=prompt_len,
                                    max_new_tokens=q0_input_count - snapped_keep_count,
                                    initial_generated_count=snapped_keep_count,
                                )
                                expensive_valid_pred_mask_s = (
                                    (generated_tokens_s >= q0_min_id) & (generated_tokens_s < q0_max_id)
                                )
                                replace_suffix_s = q0_input_mask_s & (q0_rank_s > snapped_keep_count)
                                replace_mask_s = replace_suffix_s & expensive_valid_pred_mask_s
                                return jnp.where(replace_mask_s, generated_tokens_s, tokens_s)

                            return jax.lax.cond(
                                use_inference_policy,
                                expensive_replace,
                                lambda _: cheap_replace(raw_keep_count),
                                jnp.asarray(0, dtype=jnp.int32),
                            )

                        return cheap_replace(raw_keep_count)

                    sample_tokens_s = jax.lax.cond(do_self_condition, self_condition, lambda t: t, sample_tokens_s)

                hidden_sd = llm.forward_hidden(sample_tokens_s)

                sample_semantic_mask_s = sample_q0_mask_s | sample_eos_mask_s
                targets_extra_s = jnp.where(sample_semantic_mask_s, sample_targets_s - base_vocab_size, 0)
                targets_extra_s = jnp.clip(targets_extra_s, 0, self.extra_vocab_size - 1)
                q0_loss = chunked_cross_entropy_loss(
                    hidden_sd,
                    targets_extra_s,
                    audio_lm_head,
                    mask_t=sample_q0_mask_s,
                    chunk_size=128,
                )
                q0_acc = chunked_cross_entropy_acc(
                    hidden_sd,
                    targets_extra_s,
                    audio_lm_head,
                    mask_t=sample_q0_mask_s,
                    chunk_size=128,
                )

                eos_loss = chunked_cross_entropy_loss(
                    hidden_sd,
                    targets_extra_s,
                    audio_lm_head,
                    mask_t=sample_eos_mask_s,
                    chunk_size=128,
                )
                eos_acc = chunked_cross_entropy_acc(
                    hidden_sd,
                    targets_extra_s,
                    audio_lm_head,
                    mask_t=sample_eos_mask_s,
                    chunk_size=128,
                )

                # Weight EOS relative to Q0 positions. This uses a weighted average
                # on the total negative log-likelihood sum (and similarly for
                # accuracy), treating EOS as `semantic_eos_weight` positions.
                q0_count = jnp.maximum(sample_q0_mask_s.astype(jnp.float32).sum(), 1.0)
                eos_count = sample_eos_mask_s.astype(jnp.float32).sum()
                eos_weight = jnp.asarray(self.config.semantic_eos_weight, dtype=jnp.float32)
                denom = q0_count + eos_weight * eos_count
                semantic_loss = (q0_loss * q0_count + eos_loss * eos_weight * eos_count) / denom
                semantic_acc = (q0_acc * q0_count + eos_acc * eos_weight * eos_count) / denom

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

                return semantic_loss, semantic_acc, q0_loss, q0_acc, eos_loss, eos_acc, text_loss, text_acc, hidden_sd

            (
                stage1_loss_b,
                stage1_acc_b,
                stage1_q0_loss_b,
                stage1_q0_acc_b,
                stage1_eos_loss_b,
                stage1_eos_acc_b,
                text_loss_b,
                text_acc_b,
                hidden_bsd,
            ) = jax.vmap(
                compute_stage1_terms,
                in_axes=(0, 0, 0, 0, 0, 0),
            )(
                tokens_bs,
                targets_bs,
                q0_mask_bs,
                eos_mask_bs,
                text_mask_bs,
                jax.random.split(key, bsz),
            )

        stage1_loss = stage1_loss_b.mean()
        stage1_acc = stage1_acc_b.mean()
        stage1_q0_loss = stage1_q0_loss_b.mean()
        stage1_q0_acc = stage1_q0_acc_b.mean()
        stage1_eos_loss = stage1_eos_loss_b.mean()
        stage1_eos_acc = stage1_eos_acc_b.mean()
        text_loss = text_loss_b.mean()
        text_acc = text_acc_b.mean()

        stage1_future_q0_loss = jnp.array(0.0, dtype=jnp.float32)
        stage1_future_q0_acc = jnp.array(0.0, dtype=jnp.float32)
        if (
            (not model.semantic.use_non_ar)
            and self.config.semantic_future_prediction_weight > 0
            and model.semantic.future_q0_heads is not None
        ):
            future_loss_terms = []
            future_acc_terms = []
            for offset, head in enumerate(model.semantic.future_q0_heads, start=1):
                future_targets_bs = jnp.pad(targets_bs[:, offset:], ((0, 0), (0, offset)))
                future_mask_bs = jnp.pad(q0_mask_bs[:, offset:], ((0, 0), (0, offset)))
                future_targets_bs = jnp.where(future_mask_bs, future_targets_bs - q0_min_id, 0)
                future_targets_bs = jnp.clip(future_targets_bs, 0, CODEBOOK_SIZE - 1)

                def future_head(hidden_td: Array, *, head: eqx.nn.Linear = head) -> Array:
                    return apply_linear(hidden_td.astype(jnp.float32), head)

                future_loss_b = jax.vmap(
                    lambda hidden_sd, target_s, mask_s: chunked_cross_entropy_loss(
                        hidden_sd,
                        target_s,
                        future_head,
                        mask_t=mask_s,
                        chunk_size=128,
                    )
                )(hidden_bsd, future_targets_bs, future_mask_bs)
                future_acc_b = jax.vmap(
                    lambda hidden_sd, target_s, mask_s: chunked_cross_entropy_acc(
                        hidden_sd,
                        target_s,
                        future_head,
                        mask_t=mask_s,
                        chunk_size=128,
                    )
                )(hidden_bsd, future_targets_bs, future_mask_bs)
                future_loss_terms.append(future_loss_b.mean())
                future_acc_terms.append(future_acc_b.mean())

            stage1_future_q0_loss = jnp.mean(jnp.stack(future_loss_terms)).astype(jnp.float32)
            stage1_future_q0_acc = jnp.mean(jnp.stack(future_acc_terms)).astype(jnp.float32)
            stage1_loss = stage1_loss + self.config.semantic_future_prediction_weight * stage1_future_q0_loss

        # Stage 2: residual prediction loss.
        hidden_bsd_for_stage2 = hidden_bsd
        if self.config.detach_semantic_hidden_for_stage2:
            hidden_bsd_for_stage2 = jax.lax.stop_gradient(hidden_bsd_for_stage2)
        stage2_losses_bl, stage2_accs_bl = self._compute_stage2_losses_by_codebook(
            model,
            codes_bs=codes_bs,
            audio_codes_btf=audio_codes_btf,
            hidden_bsd=hidden_bsd_for_stage2,
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
            "stage1_q0_loss": xax.Scalar(stage1_q0_loss),
            "stage1_q0_accuracy": xax.Scalar(stage1_q0_acc),
            "stage1_future_q0_loss": xax.Scalar(stage1_future_q0_loss),
            "stage1_future_q0_accuracy": xax.Scalar(stage1_future_q0_acc),
            "stage1_eos_loss": xax.Scalar(stage1_eos_loss),
            "stage1_eos_accuracy": xax.Scalar(stage1_eos_acc),
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
            (
                audio_t,
                gt_audio_t,
                gt_text_t,
                gen_len,
                num_frames,
                invalid_count,
                has_eos,
                max_frames_cap,
            ) = self._generate_audio(model, batch, key)
            metrics["generated_audio"] = xax.Audio(audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["real_audio"] = xax.Audio(gt_audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["generated_audio_rms"] = xax.Scalar(
                jnp.sqrt(jnp.mean(audio_t.astype(jnp.float32) ** 2)).astype(jnp.float32)
            )
            metrics["real_audio_rms"] = xax.Scalar(
                jnp.sqrt(jnp.mean(gt_audio_t.astype(jnp.float32) ** 2)).astype(jnp.float32)
            )
            metrics["eval_prompt"] = xax.Tokens(self.eval_prompt_tokens, tokenizer="llm")
            metrics["generated_q0_length"] = xax.Scalar(gen_len.astype(jnp.float32))
            metrics["generated_num_frames"] = xax.Scalar(num_frames.astype(jnp.float32))
            metrics["invalid_q0_token_count"] = xax.Scalar(invalid_count.astype(jnp.float32))
            metrics["semantic_has_eos"] = xax.Scalar(has_eos.astype(jnp.float32))
            metrics["semantic_max_frames_cap"] = xax.Scalar(max_frames_cap.astype(jnp.float32))

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
            (
                audio_in_domain_t,
                q0_len_in_domain,
                num_frames_in_domain,
                invalid_in_domain,
                has_eos_in_domain,
                max_cap_in_domain,
            ) = self._generate_audio_from_prompt(
                model,
                prompt_tokens_s=self.eval_prompt_in_domain_tokens,
                key=indomain_key,
            )
            metrics["generated_audio_in_domain"] = xax.Audio(audio_in_domain_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["generated_audio_rms_in_domain"] = xax.Scalar(
                jnp.sqrt(jnp.mean(audio_in_domain_t.astype(jnp.float32) ** 2)).astype(jnp.float32)
            )
            metrics["eval_prompt_in_domain"] = xax.Tokens(self.eval_prompt_in_domain_tokens, tokenizer="llm")
            metrics["generated_q0_length_in_domain"] = xax.Scalar(q0_len_in_domain.astype(jnp.float32))
            metrics["generated_num_frames_in_domain"] = xax.Scalar(num_frames_in_domain.astype(jnp.float32))
            metrics["invalid_q0_token_count_in_domain"] = xax.Scalar(invalid_in_domain.astype(jnp.float32))
            metrics["semantic_has_eos_in_domain"] = xax.Scalar(has_eos_in_domain.astype(jnp.float32))
            metrics["semantic_max_frames_cap_in_domain"] = xax.Scalar(max_cap_in_domain.astype(jnp.float32))

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
            if self.config.residual_semantic_hidden_dropout_prob > 0:
                key, drop_key = jax.random.split(key)
                drop_prob = self.config.residual_semantic_hidden_dropout_prob
                keep_mask_t = jax.random.uniform(drop_key, (max_frames,)) >= drop_prob
                keep_mask_t = keep_mask_t & frame_mask_t
                semantic_hidden_td = jnp.where(keep_mask_t[:, None], semantic_hidden_td, 0)
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

    def _score_semantic_candidate_tokens(
        self,
        model: FullTTSModel,
        candidate_tokens_s: Array,
        *,
        candidate_pos: Array,
        prompt_len: int,
    ) -> Array:
        if candidate_tokens_s.shape[0] <= prompt_len:
            return jnp.asarray(0.0, dtype=jnp.float32)

        llm = model.semantic.llm
        audio_lm_head = self._audio_lm_head(llm)
        base_vocab_size = llm.config.vocab_size
        q0_max_id = self.first_q0_id + CODEBOOK_SIZE

        input_tokens_s = candidate_tokens_s[:-1]
        targets_s = candidate_tokens_s[1:]
        target_pos_s = jnp.arange(targets_s.shape[0], dtype=jnp.int32)
        generated_target_mask_s = (
            (target_pos_s >= jnp.asarray(prompt_len - 1, dtype=jnp.int32))
            & (target_pos_s < (candidate_pos - 1))
        )
        semantic_target_mask_s = (
            ((targets_s >= self.first_q0_id) & (targets_s < q0_max_id))
            | (targets_s == self.audio_end_id)
        )
        score_mask_s = generated_target_mask_s & semantic_target_mask_s
        targets_extra_s = jnp.where(score_mask_s, targets_s - base_vocab_size, 0)
        targets_extra_s = jnp.clip(targets_extra_s, 0, self.extra_vocab_size - 1)
        hidden_sd = llm.forward_hidden(input_tokens_s)
        return chunked_cross_entropy_loss(
            hidden_sd,
            targets_extra_s,
            audio_lm_head,
            mask_t=score_mask_s,
            chunk_size=128,
        )

    def _generate_audio(
        self,
        model: FullTTSModel,
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        if model.mimi is None:
            raise ValueError("Mimi model is required for audio generation.")
        audio_gen_t, gen_len, num_frames, invalid_count, has_eos, max_frames_cap = self._generate_audio_from_prompt(
            model,
            prompt_tokens_s=self.eval_prompt_tokens,
            key=key,
        )

        # Ground-truth audio from the first sample.
        gt_codes_tf = batch["audio_codes"][0]  # (T, 8)
        gt_num_frames = (gt_codes_tf[:, 0] != AUDIO_PAD_TOKEN_ID).sum()
        gt_codes_ft = gt_codes_tf.T
        gt_codes_ft = jnp.where((gt_codes_ft >= 0) & (gt_codes_ft < CODEBOOK_SIZE), gt_codes_ft, 0)
        audio_gt = model.mimi.decode(gt_codes_ft)
        audio_gt_t = audio_gt[0]
        hop_length = int(round(model.mimi.config.sampling_rate / model.mimi.config.frame_rate))
        audio_gt_t = self._mask_audio_after_num_frames(audio_gt_t, num_frames=gt_num_frames, hop_length=hop_length)

        gt_ids = batch["codes"][0]
        gt_text_ids = jax.lax.dynamic_slice(gt_ids, (1,), (gt_ids.shape[0] - 1,))

        return audio_gen_t, audio_gt_t, gt_text_ids, gen_len, num_frames, invalid_count, has_eos, max_frames_cap

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

    def _mask_audio_after_num_frames(self, audio_t: Array, *, num_frames: Array, hop_length: int) -> Array:
        """Zeros out audio samples after `num_frames` codec frames.

        This keeps tensor shapes static (JIT-friendly) while avoiding long tails
        of decoded garbage/silence from padded codec frames.
        """
        audio_t = audio_t.astype(jnp.float32)
        num_samples = jnp.maximum(num_frames.astype(jnp.int32), 0) * jnp.asarray(hop_length, dtype=jnp.int32)
        idx_s = jnp.arange(audio_t.shape[0], dtype=jnp.int32)
        return jnp.where(idx_s < num_samples, audio_t, 0.0)

    def _generate_audio_from_prompt(
        self,
        model: FullTTSModel,
        *,
        prompt_tokens_s: Array,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        if model.mimi is None:
            raise ValueError("Mimi model is required for audio generation.")
        k1, k2 = jax.random.split(key)
        prompt_len = int(prompt_tokens_s.shape[0])
        max_frames = self._semantic_inference_max_frames(prompt_len)

        q0_min_id = self.first_q0_id
        q0_max_id = self.first_q0_id + CODEBOOK_SIZE

        if model.semantic.use_non_ar:
            _, q0_logits_tv, semantic_hidden_td = self._semantic_non_ar_forward(
                model,
                prompt_tokens_s,
                total_frames=max_frames,
            )
            q0_codes_t = jnp.argmax(q0_logits_tv, axis=-1).astype(jnp.int32)
            q0_codes_t = jnp.clip(q0_codes_t, 0, CODEBOOK_SIZE - 1)
            num_frames = jnp.asarray(max_frames, dtype=jnp.int32)
            gen_len = num_frames
            invalid_count = jnp.asarray(0, dtype=jnp.int32)
            has_eos = jnp.asarray(False)
            frame_mask_t = jnp.arange(max_frames) < num_frames
        else:
            # Ensure we never generate more frames than Stage 2 can decode.
            max_new_tokens = min(
                self.max_seq_length - prompt_len,
                max_frames + 1,  # +1 to allow the EOS token.
            )

            use_blockwise_decode = (
                model.semantic.future_q0_heads is not None
                and (
                    self.config.semantic_block_decode_size > 1
                    or (
                        self.config.semantic_block_decode_schedule is not None
                        and self.config.semantic_block_decode_schedule.strip() != ""
                    )
                )
            )
            if use_blockwise_decode:
                gen_tokens_s, gen_pos = self._generate_semantic_tokens_blockwise_greedy(
                    model,
                    prompt_tokens_s=prompt_tokens_s,
                    max_new_tokens=max_new_tokens,
                )
                best_score = self._score_semantic_candidate_tokens(
                    model,
                    gen_tokens_s,
                    candidate_pos=gen_pos,
                    prompt_len=prompt_len,
                )

                if self.config.semantic_eval_compare_exact_last_candidate:
                    alt_gen_tokens_s, alt_gen_pos = self._generate_semantic_tokens_blockwise_greedy(
                        model,
                        prompt_tokens_s=prompt_tokens_s,
                        max_new_tokens=max_new_tokens,
                        exact_last_token_override=not self.config.semantic_block_decode_exact_last_token,
                    )
                    alt_score = self._score_semantic_candidate_tokens(
                        model,
                        alt_gen_tokens_s,
                        candidate_pos=alt_gen_pos,
                        prompt_len=prompt_len,
                    )
                    gen_tokens_s, gen_pos, best_score = jax.lax.cond(
                        alt_score < best_score,
                        lambda _: (alt_gen_tokens_s, alt_gen_pos, alt_score),
                        lambda _: (gen_tokens_s, gen_pos, best_score),
                        operand=None,
                    )

                if (
                    self.config.semantic_eval_compare_schedule_candidate is not None
                    and self.config.semantic_eval_compare_schedule_candidate.strip() != ""
                ):
                    schedule_gen_tokens_s, schedule_gen_pos = self._generate_semantic_tokens_blockwise_greedy(
                        model,
                        prompt_tokens_s=prompt_tokens_s,
                        max_new_tokens=max_new_tokens,
                        block_decode_schedule_override=self.config.semantic_eval_compare_schedule_candidate,
                    )
                    schedule_score = self._score_semantic_candidate_tokens(
                        model,
                        schedule_gen_tokens_s,
                        candidate_pos=schedule_gen_pos,
                        prompt_len=prompt_len,
                    )
                    gen_tokens_s, gen_pos, best_score = jax.lax.cond(
                        schedule_score < best_score,
                        lambda _: (schedule_gen_tokens_s, schedule_gen_pos, schedule_score),
                        lambda _: (gen_tokens_s, gen_pos, best_score),
                        operand=None,
                    )
            else:
                gen_tokens_s, gen_pos = model.semantic.generate_tokens(
                    prompt_tokens_s=prompt_tokens_s,
                    max_new_tokens=max_new_tokens,
                    audio_end_id=self.audio_end_id,
                    temperature=self.config.semantic_gen_temperature,
                    top_p=self.config.semantic_gen_top_p,
                    key=k1,
                    allowed_token_range=(q0_min_id, q0_max_id),
                    min_new_tokens_before_eos=min(
                        self.config.semantic_gen_min_new_tokens,
                        max(0, max_new_tokens - 1),
                    ),
                )

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
            # Avoid degenerate 0-frame generations (they can cause awkward masking
            # patterns downstream). This only affects pathological early-EOS cases.
            num_frames = jnp.maximum(num_frames, jnp.asarray(1, dtype=jnp.int32))
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

        audio_t = audio_gen[0]
        hop_length = int(round(model.mimi.config.sampling_rate / model.mimi.config.frame_rate))
        audio_t = self._mask_audio_after_num_frames(audio_t, num_frames=num_frames, hop_length=hop_length)

        max_frames_cap = jnp.asarray(max_frames, dtype=jnp.int32)
        return audio_t, gen_len, num_frames, invalid_count, has_eos, max_frames_cap

    def _normalize_asr_text(self, text: str) -> str:
        """Normalize text for ASR-based string metrics (WER/CER)."""
        text = text.lower()
        # Keep letters/numbers/apostrophes. Collapse all other runs to spaces.
        text = re.sub(r"[^a-z0-9']+", " ", text)
        return re.sub(r"\\s+", " ", text).strip()

    def _edit_distance(self, ref: list[str], hyp: list[str]) -> int:
        """Levenshtein edit distance (insert/delete/substitute)."""
        if not ref:
            return len(hyp)
        if not hyp:
            return len(ref)
        dp = list(range(len(hyp) + 1))
        for ref_idx, ref_tok in enumerate(ref, start=1):
            prev = dp[0]
            dp[0] = ref_idx
            for hyp_idx, hyp_tok in enumerate(hyp, start=1):
                cur = dp[hyp_idx]
                cost = 0 if ref_tok == hyp_tok else 1
                dp[hyp_idx] = min(
                    dp[hyp_idx] + 1,  # delete
                    dp[hyp_idx - 1] + 1,  # insert
                    prev + cost,  # substitute
                )
                prev = cur
        return dp[-1]

    def _wer(self, ref: str, hyp: str) -> float:
        ref_words = self._normalize_asr_text(ref).split()
        hyp_words = self._normalize_asr_text(hyp).split()
        denom = max(1, len(ref_words))
        return self._edit_distance(ref_words, hyp_words) / float(denom)

    def _cer(self, ref: str, hyp: str) -> float:
        ref_norm = self._normalize_asr_text(ref).replace(" ", "")
        hyp_norm = self._normalize_asr_text(hyp).replace(" ", "")
        denom = max(1, len(ref_norm))
        return self._edit_distance(list(ref_norm), list(hyp_norm)) / float(denom)

    @override
    def log_step(self, metrics: xax.FrozenDict[str, xax.Metric], state: xax.State, heavy: bool) -> None:
        metrics_d: dict[str, xax.Metric] = dict(metrics)

        # Compute WER/CER as host-side derived metrics from the text summaries.
        if heavy and self.config.enable_heavy_eval:
            try:
                prompt_metric = cast(xax.Tokens, metrics_d["eval_prompt"])
                transcript_metric = cast(xax.Tokens, metrics_d["transcript"])
                prompt_text = self.decode_tokens(np.asarray(jax.device_get(prompt_metric.value)), "llm")
                transcript_text = self.decode_tokens(np.asarray(jax.device_get(transcript_metric.value)), "whisper")
                ref_words = self._normalize_asr_text(prompt_text).split()
                hyp_words = self._normalize_asr_text(transcript_text).split()
                hyp_prefix = " ".join(hyp_words[: len(ref_words)])
                metrics_d["asr_wer"] = xax.Scalar(
                    jnp.asarray(self._wer(prompt_text, transcript_text), dtype=jnp.float32)
                )
                metrics_d["asr_cer"] = xax.Scalar(
                    jnp.asarray(self._cer(prompt_text, transcript_text), dtype=jnp.float32)
                )
                metrics_d["asr_wer_prefix"] = xax.Scalar(
                    jnp.asarray(self._wer(prompt_text, hyp_prefix), dtype=jnp.float32)
                )
                metrics_d["asr_ref_num_words"] = xax.Scalar(jnp.asarray(float(len(ref_words)), dtype=jnp.float32))
                metrics_d["asr_hyp_num_words"] = xax.Scalar(jnp.asarray(float(len(hyp_words)), dtype=jnp.float32))

                prompt_in_metric = cast(xax.Tokens, metrics_d["eval_prompt_in_domain"])
                transcript_in_metric = cast(xax.Tokens, metrics_d["transcript_in_domain"])
                prompt_in_text = self.decode_tokens(np.asarray(jax.device_get(prompt_in_metric.value)), "llm")
                transcript_in_text = self.decode_tokens(
                    np.asarray(jax.device_get(transcript_in_metric.value)),
                    "whisper",
                )
                ref_in_words = self._normalize_asr_text(prompt_in_text).split()
                hyp_in_words = self._normalize_asr_text(transcript_in_text).split()
                hyp_in_prefix = " ".join(hyp_in_words[: len(ref_in_words)])
                metrics_d["asr_wer_in_domain"] = xax.Scalar(
                    jnp.asarray(self._wer(prompt_in_text, transcript_in_text), dtype=jnp.float32)
                )
                metrics_d["asr_cer_in_domain"] = xax.Scalar(
                    jnp.asarray(self._cer(prompt_in_text, transcript_in_text), dtype=jnp.float32)
                )
                metrics_d["asr_wer_prefix_in_domain"] = xax.Scalar(
                    jnp.asarray(self._wer(prompt_in_text, hyp_in_prefix), dtype=jnp.float32)
                )
                metrics_d["asr_ref_num_words_in_domain"] = xax.Scalar(
                    jnp.asarray(float(len(ref_in_words)), dtype=jnp.float32)
                )
                metrics_d["asr_hyp_num_words_in_domain"] = xax.Scalar(
                    jnp.asarray(float(len(hyp_in_words)), dtype=jnp.float32)
                )

                # Sanity check: Whisper on real audio vs training text.
                gt_text_metric = cast(xax.Tokens, metrics_d["gt_text"])
                gt_transcript_metric = cast(xax.Tokens, metrics_d["gt_transcript"])
                gt_text = self.decode_tokens(np.asarray(jax.device_get(gt_text_metric.value)), "llm")
                gt_transcript = self.decode_tokens(np.asarray(jax.device_get(gt_transcript_metric.value)), "whisper")
                metrics_d["asr_wer_gt"] = xax.Scalar(jnp.asarray(self._wer(gt_text, gt_transcript), dtype=jnp.float32))
                metrics_d["asr_cer_gt"] = xax.Scalar(jnp.asarray(self._cer(gt_text, gt_transcript), dtype=jnp.float32))
            except KeyError:
                # Heavy eval is optional; if the required keys aren't present, skip.
                pass
            except Exception:
                logger.exception("Failed to compute ASR WER/CER derived metrics")

        for k, v in metrics_d.items():
            try:
                self.logger.log_metric(k, v)
            except Exception as e:
                raise ValueError(f"Error logging metric {k}") from e

        self.log_state_timers(state)
        self.write_logs(state, heavy)

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

    @xax.dataset_fn("unpadded", dependencies=["tokenized"])
    def unpadded_dataset(self) -> Dataset:
        ds = cast(Dataset, self.load_dataset("tokenized"))

        def make_dataset_for_text_key(text_key: str) -> Dataset:
            def prepare_sample(example: dict) -> dict:
                text = cast(str, example[text_key])
                text_tokens = np.asarray(self.tokenizer.encode(text), dtype=np.int32)
                # Text segment with explicit boundaries.
                text_with_special = np.concatenate([[self.text_start_id], text_tokens, [self.text_end_id]])

                audio_codes_tc = np.asarray(example["audio_codes"], dtype=np.int32)  # (T, 8)
                q0_codes_t = audio_codes_tc[:, 0]
                q0_tokens = q0_codes_t + self.first_q0_id
                audio_with_special = np.concatenate([[self.audio_start_id], q0_tokens, [self.audio_end_id]])

                codes = np.concatenate([text_with_special, audio_with_special]).astype(np.int32)
                return {"codes": codes, "audio_codes": audio_codes_tc.astype(np.int32)}

            return cast(Dataset, ds.map(prepare_sample, desc=f"Preparing sequences ({text_key})"))

        text_source = self.config.text_source.strip().lower()
        if text_source == "normalized":
            result = make_dataset_for_text_key("text_norm")
        elif text_source == "raw":
            result = make_dataset_for_text_key("text_raw")
        elif text_source == "both":
            result = concatenate_datasets(
                [
                    make_dataset_for_text_key("text_norm"),
                    make_dataset_for_text_key("text_raw"),
                ]
            )
        elif text_source == "both_plus_normalized":
            result = concatenate_datasets(
                [
                    make_dataset_for_text_key("text_norm"),
                    make_dataset_for_text_key("text_norm"),
                    make_dataset_for_text_key("text_raw"),
                ]
            )
        elif text_source == "both_weighted":
            norm_repeats = max(0, int(self.config.text_source_weighted_normalized_repeats))
            raw_repeats = max(0, int(self.config.text_source_weighted_raw_repeats))
            result_parts = [make_dataset_for_text_key("text_norm") for _ in range(norm_repeats)]
            result_parts.extend(make_dataset_for_text_key("text_raw") for _ in range(raw_repeats))
            if not result_parts:
                raise ValueError("text_source=both_weighted requires at least one normalized or raw repeat.")
            result = concatenate_datasets(result_parts)
        else:
            raise ValueError(
                "Invalid text_source: "
                f"{self.config.text_source!r} (expected normalized, raw, both, both_plus_normalized, or both_weighted)"
            )
        cols_to_keep = ["codes", "audio_codes"]
        cols_to_remove = [c for c in result.column_names if c not in cols_to_keep]
        if cols_to_remove:
            result = result.remove_columns(cols_to_remove)
        return cast(Dataset, result)

    def _load_legacy_tokenized_cache(self) -> Dataset | None:
        """Loads a compatible tokenized cache from disk when remote rebuilds are unavailable.

        Older caches stored `text_tokens` directly instead of `text_norm` / `text_raw`.
        We can recover a usable normalized-text view by decoding those tokens with the
        current tokenizer, which is enough to keep training/evaluation moving even when
        the upstream dataset script is no longer supported by `datasets`.
        """
        tokenized_root = self.dataset_cache_dir / "tokenized"
        candidate_names = ("v4", "v3", "default")
        for candidate_name in candidate_names:
            candidate_path = tokenized_root / candidate_name
            if not candidate_path.exists():
                continue

            logger.info("Attempting to reuse tokenized cache from %s", candidate_path)
            ds = cast(Dataset, load_from_disk(str(candidate_path)))
            column_names = set(ds.column_names)
            if {"text_norm", "text_raw", "audio_codes"}.issubset(column_names):
                cols_to_keep = ["text_norm", "text_raw", "audio_codes"]
                cols_to_remove = [column for column in ds.column_names if column not in cols_to_keep]
                if cols_to_remove:
                    ds = ds.remove_columns(cols_to_remove)
                return ds

            if {"text_tokens", "audio_codes"}.issubset(column_names):
                logger.warning(
                    "Reusing legacy tokenized cache from %s; `text_raw` will mirror the decoded normalized text",
                    candidate_path,
                )

                def upgrade_example(example: dict) -> dict:
                    text_tokens_s = np.asarray(example["text_tokens"], dtype=np.int32)
                    text_norm = self.tokenizer.decode(text_tokens_s.tolist(), skip_special_tokens=True)
                    return {
                        "text_norm": text_norm,
                        "text_raw": text_norm,
                        "audio_codes": np.asarray(example["audio_codes"], dtype=np.int32),
                    }

                upgraded_ds = cast(
                    Dataset,
                    ds.map(upgrade_example, desc=f"Upgrading tokenized cache ({candidate_name})"),
                )
                cols_to_keep = ["text_norm", "text_raw", "audio_codes"]
                cols_to_remove = [column for column in upgraded_ds.column_names if column not in cols_to_keep]
                if cols_to_remove:
                    upgraded_ds = upgraded_ds.remove_columns(cols_to_remove)
                return upgraded_ds

        return None

    # NOTE: This stage performs Mimi encoding on GPU and can be expensive. We
    # intentionally use a manual hash so the cache is stable across unrelated
    # code changes; bump this hash when the encoding/tokenization logic changes.
    @xax.dataset_fn("tokenized", hash="v4")
    def tokenized_v4_dataset(self) -> Dataset:
        columns = ["text_norm", "text_raw", "audio_codes"]

        if (legacy_ds := self._load_legacy_tokenized_cache()) is not None:
            logger.info("Using local tokenized cache instead of rebuilding from the remote LJSpeech script")
            return legacy_ds

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

                outputs["text_norm"].append(cast(str, examples["normalized_text"][idx]))
                outputs["text_raw"].append(cast(str, examples["text"][idx]))
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
        logger.info("Columns: text_norm (str), text_raw (str), audio_codes (T, %d)", NUM_QUANTIZERS)
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
