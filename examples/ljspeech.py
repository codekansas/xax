#!/usr/bin/env -S uv run --no-project --script
"""Two-stage TTS architecture for LJSpeech using Mimi codec tokens with BPE.

Architecture:
- Stage 1 (Semantic Model): LLM predicts BPE tokens learned only on Q0 (semantic) codes
- Stage 2 (Residual Model): 7 independent autoregressive transformers predict Q1-Q7,
  each generating all timesteps before moving to the next layer

- Sequence format: [TEXT_START] [TEXT] [TEXT_END] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]
- BPE compression on Q0 (semantic) codec tokens only
- Joint end-to-end training of both stages
"""

import functools
import json
import logging
from dataclasses import dataclass
from typing import Iterator, TypedDict, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.whisper.tokenization_whisper_fast import WhisperTokenizerFast

import xax
from xax.arch.llm import LLMConfig

logger = logging.getLogger(__name__)

# Mimi constants
LJSPEECH_SAMPLE_RATE = 22050
NUM_QUANTIZERS = 8  # Q0 (semantic) + Q1-Q7 (acoustic)

# Adds additional special tokens for modeling the audio tokens.
AUDIO_BOS_TOKEN_ID = xax.MIMI_CODEBOOK_SIZE
AUDIO_EOS_TOKEN_ID = xax.MIMI_CODEBOOK_SIZE + 1
AUDIO_PAD_TOKEN_ID = xax.MIMI_CODEBOOK_SIZE + 2
AUDIO_VOCAB_SIZE = xax.MIMI_CODEBOOK_SIZE + 3

# Special token string markers (for text tokenizer)
TEXT_START_TOKEN = "<|text_start|>"
TEXT_END_TOKEN = "<|text_end|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

# LoRA targets
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "gate", "up")


class Batch(TypedDict):
    codes: Array  # (batch, seq_len) - combined text + audio tokens
    audio_codes: Array  # (batch, T, 8) - full audio codes for Stage 2 training


class SemanticTTSModel(eqx.Module):
    """Stage 1: LLM for predicting Q0 (semantic) BPE tokens.

    The model processes sequences of the form:
    [TEXT_START] [TEXT] [TEXT_END] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]

    Text tokens use the LLM's original embeddings/output head (frozen).
    Audio BPE tokens (learned on Q0 only) use separate learned embeddings/output head.
    """

    llm: xax.LLM

    def generate_audio(
        self,
        prompt_tokens_s: Array,
        max_audio_tokens: int,
        audio_end_id: int,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Generate audio BPE tokens given text prefix using sampling.

        Args:
            prompt_tokens_s: Prompt token IDs, with shape (text_len + 2,)
            max_audio_tokens: Maximum number of audio tokens to generate
            audio_end_id: Audio EOS token ID
            key: PRNG key for sampling

        Returns:
            Tuple of (generated audio BPE token IDs, final sequence length)
        """
        return xax.llm_generate_jit(
            self.llm,
            prompt_tokens_s,
            eos_id=audio_end_id,
            max_new_tokens=max_audio_tokens,
            context_tn=None,
            temperature=0.8,
            top_p=0.9,
            key=key,
        )


# Residual model configuration (kept small due to 7 separate LLMs)
RESIDUAL_HEAD_DIM = 64
RESIDUAL_NUM_HEADS = 4
RESIDUAL_NUM_LAYERS = 2
RESIDUAL_MLP_DIM = 256


class ResidualModel(eqx.Module):
    """Stage 2: 7 independent LLMs for predicting Q1-Q7.

    Each LLM predicts all timesteps for its layer before moving to the next.
    Uses xax.LLM with context_tn for conditioning on lower layers and semantic hidden states.
    """

    hidden_proj: eqx.nn.Linear
    layer_llms: tuple[xax.LLM, ...]
    codebook_embeddings: tuple[eqx.nn.Embedding, ...]

    @staticmethod
    def build(
        llm_embed_dim: int,
        head_dim: int = RESIDUAL_HEAD_DIM,
        num_heads: int = RESIDUAL_NUM_HEADS,
        num_layers: int = RESIDUAL_NUM_LAYERS,
        mlp_dim: int = RESIDUAL_MLP_DIM,
        *,
        key: PRNGKeyArray,
    ) -> "ResidualModel":
        """Build the residual model using xax.LLM for each layer.

        Args:
            llm_embed_dim: Dimension of the LLM hidden states
            head_dim: Dimension of each head of the transformer model.
            num_heads: Number of attention heads per layer
            num_layers: Number of transformer layers per LLM
            mlp_dim: MLP hidden dimension
            key: PRNG key for initialization

        Returns:
            ResidualModel instance
        """
        embed_dim = head_dim * num_heads

        # Project LLM hidden states to residual dimension
        key, hidden_key = jax.random.split(key)
        hidden_proj = eqx.nn.Linear(llm_embed_dim, embed_dim, key=hidden_key)

        # LLM config for residual layers.
        llm_config = LLMConfig(
            vocab_size=AUDIO_VOCAB_SIZE,
            embed_dim=embed_dim,
            q_heads=num_heads,
            kv_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            mlp_hidden_dim=mlp_dim,
        )

        # Build 8 LLMs for Q0-Q7
        layer_llms = []
        codebook_embeddings = []
        for _ in range(1, NUM_QUANTIZERS):
            key, layer_key = jax.random.split(key)
            llm = xax.LLM.build(llm_config, key=layer_key)
            layer_llms.append(llm)

            key, emb_key = jax.random.split(key)
            weight = jax.random.normal(emb_key, (AUDIO_VOCAB_SIZE, embed_dim)) * 0.02
            emb = eqx.nn.Embedding(AUDIO_VOCAB_SIZE, embed_dim, weight=weight)
            codebook_embeddings.append(emb)

        return ResidualModel(
            hidden_proj=hidden_proj,
            layer_llms=tuple(layer_llms),
            codebook_embeddings=tuple(codebook_embeddings),
        )

    def compute_loss(
        self,
        audio_codes_ft: Array,
        stretched_hidden_td: Array,
    ) -> tuple[Array, Array]:
        """Compute cross-entropy loss for all residual layers (Q1-Q7).

        This computes losses for all layers in a single pass, passing hidden states
        from each layer to the next.

        Args:
            audio_codes_ft: All audio codes, shape (8, T)
            stretched_hidden_td: Stretched hidden states, shape (T, llm_embed_dim)

        Returns:
            The layer losses and accuracies.
        """
        # Common mask for all layers.
        mask_t = audio_codes_ft[0, 1:] != AUDIO_PAD_TOKEN_ID

        # First-layer embeddings.
        # stretched_hidden_td[f] = hidden for frame f (frame-indexed)
        # audio_codes_ft[q, p] = code at seq_pos p (seq-pos-indexed, BOS at 0)
        # Using [:-1] on stretched_hidden gives frames 0 to T-1.
        # Using [1:] on audio_codes gives seq_pos 1 to T = frames 0 to T-1.
        # Both align at frame index i.
        x_td = jax.vmap(self.hidden_proj)(stretched_hidden_td[:-1])

        losses = []
        accuracies = []

        for layer_idx in range(1, NUM_QUANTIZERS):
            llm = self.layer_llms[layer_idx - 1]

            # Adds the previous layer's token embeddings.
            emb = self.codebook_embeddings[layer_idx - 1]
            emb_td = jax.vmap(emb)(audio_codes_ft[layer_idx - 1, 1:])
            x_td = x_td + emb_td

            tokens_t = audio_codes_ft[layer_idx, :-1]
            targets_t = audio_codes_ft[layer_idx, 1:]

            loss, accuracy, _ = llm.get_loss_and_accuracy(tokens_t, targets_t, context_tn=x_td, mask_t=mask_t)
            losses.append(loss)
            accuracies.append(accuracy)

        total_loss = jnp.stack(losses)
        total_accuracy = jnp.stack(accuracies)

        return total_loss, total_accuracy

    def generate_codes(
        self,
        q0_codes_t: Array,
        stretched_hidden_td: Array,
        num_frames: Array,
        max_frames: int,
        key: PRNGKeyArray,
    ) -> Array:
        # Both stretched_hidden_td and q0_codes_t are frame-indexed:
        # stretched_hidden_td[f] = hidden for frame f
        # q0_codes_t[f] = Q0 code for frame f
        all_codes = [q0_codes_t]
        start_t = jnp.array([AUDIO_BOS_TOKEN_ID])

        # Project LLM context vectors.
        x_td = jax.vmap(self.hidden_proj)(stretched_hidden_td)

        for layer_idx in range(1, NUM_QUANTIZERS):
            llm = self.layer_llms[layer_idx - 1]

            # Adds the previous layer's token embeddings.
            emb = self.codebook_embeddings[layer_idx - 1]
            emb_td = jax.vmap(emb)(all_codes[-1])
            x_td = x_td + emb_td

            key, layer_key = jax.random.split(key)
            pred_tokens_t, _ = xax.llm_generate_jit(
                llm,
                start_t,
                eos_id=AUDIO_EOS_TOKEN_ID,
                max_new_tokens=max_frames,
                context_tn=x_td,
                temperature=0.8,
                top_p=0.9,
                key=layer_key,
            )
            all_codes.append(pred_tokens_t[1:])

        return jnp.stack(all_codes, axis=0)


class FullTTSModel(eqx.Module):
    """Full two-stage TTS model with Mimi decoder and Whisper for evaluation.

    Stage 1: Semantic model predicts Q0 BPE tokens
    Stage 2: Residual model predicts Q1-Q7 codes given Q0 and semantic hidden states
    """

    semantic: SemanticTTSModel  # Stage 1
    residual: ResidualModel  # Stage 2
    mimi: xax.MimiModel
    whisper_transcriber: xax.WhisperTranscriber

    @staticmethod
    def build(
        llm: xax.LLM,
        whisper_eos_token_id: int,
        *,
        key: PRNGKeyArray,
        residual_head_dim: int = RESIDUAL_HEAD_DIM,
        residual_num_heads: int = RESIDUAL_NUM_HEADS,
        residual_num_layers: int = RESIDUAL_NUM_LAYERS,
        residual_mlp_dim: int = RESIDUAL_MLP_DIM,
    ) -> "FullTTSModel":
        mimi = xax.build_pretrained_mimi()

        # Build Stage 1: Semantic model
        semantic = SemanticTTSModel(llm)

        # Build Stage 2: Residual model
        residual = ResidualModel.build(
            llm.config.embed_dim,
            head_dim=residual_head_dim,
            num_heads=residual_num_heads,
            num_layers=residual_num_layers,
            mlp_dim=residual_mlp_dim,
            key=key,
        )

        whisper_model = xax.build_pretrained_whisper()
        whisper_transcriber = xax.WhisperTranscriber(
            model=whisper_model,
            eos_token_id=whisper_eos_token_id,
        )
        return FullTTSModel(
            semantic=semantic,
            residual=residual,
            mimi=mimi,
            whisper_transcriber=whisper_transcriber,
        )


def decode_bpe_to_q0(
    bpe_tokens_b: Array,
    bpe_length: Array,
    decode_table_vs: Array,
    span_table_v: Array,
    max_frames: int,
) -> tuple[Array, Array]:
    """Decode Q0-only BPE tokens back to Q0 codec codes.

    Args:
        bpe_tokens_b: Audio BPE token IDs, shape (B,) where B is max BPE length
        bpe_length: Actual number of valid BPE tokens (scalar)
        decode_table_vs: Lookup table mapping token ID -> Q0 codes, shape (V, S)
        span_table_v: Lookup table mapping token ID -> span length, shape (V,)
        max_frames: Maximum number of audio frames

    Returns:
        Tuple of (q0_codes, num_frames) where:
        - q0_codes: Decoded Q0 codes, shape (max_frames,)
        - num_frames: Actual number of frames (scalar)
    """
    # Look up spans and Q0 codes for each BPE token
    spans_b = span_table_v[bpe_tokens_b]  # (B,)
    code_seqs_bs = decode_table_vs[bpe_tokens_b]  # (B, S) - Q0 codes

    # Compute cumulative positions
    cumsum_b = jnp.cumsum(spans_b)
    start_positions_b = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), cumsum_b[:-1]])

    # Total frames (Q0-only, so total_len = num_frames)
    num_frames = jnp.sum(jnp.where(jnp.arange(len(bpe_tokens_b)) < bpe_length, spans_b, 0))

    # Initialize Q0 output
    q0_codes = jnp.zeros(max_frames, dtype=jnp.int32)

    # Scatter codes from each BPE token
    max_span = decode_table_vs.shape[1]

    def scatter_bpe_token(carry: Array, inputs: tuple[Array, Array, Array, Array]) -> tuple[Array, None]:
        output, (start_pos, span, code_seq, bpe_idx) = carry, inputs
        valid = bpe_idx < bpe_length

        def scatter_positions(pos_in_span: int, out: Array) -> Array:
            target_pos = start_pos + pos_in_span
            valid_pos = valid & (pos_in_span < span) & (target_pos < max_frames)
            return jnp.where(valid_pos, out.at[target_pos].set(code_seq[pos_in_span]), out)

        output = jax.lax.fori_loop(0, max_span, scatter_positions, output)
        return output, None

    q0_codes, _ = jax.lax.scan(
        scatter_bpe_token,
        q0_codes,
        (start_positions_b, spans_b, code_seqs_bs, jnp.arange(len(bpe_tokens_b))),
    )

    return q0_codes, num_frames


def stretch_hidden_states(
    hidden_sd: Array,
    bpe_tokens_s: Array,
    span_table_v: Array,
    bpe_length: Array,
    max_frames: int,
) -> Array:
    """Stretch hidden states to match Q0 frame positions.

    Each BPE token's hidden state is repeated for each Q0 code it represents.

    Args:
        hidden_sd: Hidden states from semantic model, shape (seq_len, embed_dim)
        bpe_tokens_s: BPE token IDs, shape (seq_len,)
        span_table_v: Lookup table mapping token ID -> span length, shape (V,)
        bpe_length: Number of valid BPE tokens (scalar)
        max_frames: Maximum number of Q0 frames

    Returns:
        Stretched hidden states, shape (max_frames, embed_dim)
    """
    # Get spans for each BPE token
    spans_s = span_table_v[bpe_tokens_s]  # (seq_len,)

    # Compute cumulative spans (end positions for each BPE token)
    valid_spans = jnp.where(jnp.arange(len(bpe_tokens_s)) < bpe_length, spans_s, 0)
    cumsum_s = jnp.cumsum(valid_spans)

    # For each Q0 position, find which BPE token it belongs to using searchsorted
    frame_positions = jnp.arange(max_frames)
    bpe_indices = jnp.searchsorted(cumsum_s, frame_positions, side="right")
    bpe_indices = jnp.clip(bpe_indices, 0, hidden_sd.shape[0] - 1)

    # Gather hidden states for each frame position
    stretched_td = hidden_sd[bpe_indices]  # (max_frames, embed_dim)

    return stretched_td


@dataclass
class Config(xax.SupervisedConfig):
    # Model settings.
    llm_repo: xax.LLMRepo = xax.field(xax.LLMRepo.QWEN3_600M, help="Pretrained model")
    num_quantizers: int = xax.field(NUM_QUANTIZERS, help="Number of quantizers (8)")
    residual_head_dim: int = xax.field(RESIDUAL_HEAD_DIM, help="Residual model attention head dimension")
    residual_num_heads: int = xax.field(RESIDUAL_NUM_HEADS, help="Residual model number of attention heads")
    residual_num_layers: int = xax.field(RESIDUAL_NUM_LAYERS, help="Residual model number of layers")
    residual_mlp_dim: int = xax.field(RESIDUAL_MLP_DIM, help="Residual model MLP dimension")
    semantic_loss_weight: float = xax.field(1.0, help="Weight for stage-1 semantic loss in total loss")
    acoustic_loss_weight: float = xax.field(1.0, help="Weight for stage-2 acoustic loss in total loss")

    # LoRA settings.
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(16.0, help="LoRA alpha parameter")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")
    lora_targets: tuple[str, ...] | None = xax.field(DEFAULT_LORA_TARGETS, help="Layer suffixes for LoRA")

    # Training settings.
    learning_rate: float = xax.field(3e-4, help="Peak learning rate")
    warmup_steps: int = xax.field(100, help="Number of warmup steps")
    audio_bpe_vocab_size: int = xax.field(151_669, help="Vocabulary size for audio BPE tokenizer")
    length_percentile: float = xax.field(0.95, help="Percentile to use for padding lengths")

    # Eval settings.
    eval_prompt: str = xax.field("Hello, world! I'm a TTS model.", help="Prompt to use for evaluation")


class LJSpeechTTS(xax.SupervisedTask[Config]):
    """Single LLM TTS with flattened codec BPE."""

    tokenizer: Qwen2TokenizerFast
    whisper_tokenizer: WhisperTokenizerFast
    audio_bpe_tokenizer: Tokenizer | None
    audio_bpe_decode_table: Array | None
    audio_bpe_span_table: Array | None

    text_start_id: int
    text_end_id: int
    audio_start_id: int
    audio_end_id: int
    first_audio_bpe_id: int
    text_vocab_size: int
    audio_bpe_vocab_size: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Load text tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_repo.value)

        # Adds the special text tokens.
        self.tokenizer.add_special_tokens({"additional_special_tokens": [TEXT_START_TOKEN, TEXT_END_TOKEN]})
        self.text_start_id = self.tokenizer.convert_tokens_to_ids(TEXT_START_TOKEN)
        self.text_end_id = self.tokenizer.convert_tokens_to_ids(TEXT_END_TOKEN)

        # Adds the special audio tokens.
        self.tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_START_TOKEN, AUDIO_END_TOKEN]})
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids(AUDIO_START_TOKEN)
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids(AUDIO_END_TOKEN)

        # Reserves special tokens for all of the audio BPE tokens.
        audio_bpe_tokens = [f"<|audio_bpe_{i}|>" for i in range(self.config.audio_bpe_vocab_size)]
        self.tokenizer.add_tokens(audio_bpe_tokens)
        self.first_audio_bpe_id = self.tokenizer.convert_tokens_to_ids(audio_bpe_tokens[0])

        # Load Whisper for ASR evaluation
        logger.info("Loading Whisper model for ASR evaluation")
        path = xax.download_whisper_repo()
        self.whisper_config = xax.load_whisper_config()
        self.whisper_tokenizer = WhisperTokenizerFast.from_pretrained(str(path))

        # Gets the eval prompt as tokens.
        eval_prompt_tokens = self.tokenizer.encode(self.config.eval_prompt)
        eval_prompt_tokens = np.concatenate(
            [
                [self.text_start_id],
                eval_prompt_tokens,
                [self.text_end_id, self.audio_start_id],
            ]
        )
        self.eval_prompt_tokens = jnp.array(eval_prompt_tokens, dtype=jnp.int32)

    @functools.cached_property
    def audio_bpe_tokenizer(self) -> Tokenizer:
        """Load audio BPE tokenizer from cache directory."""
        tokenizer_path = self.dataset_cache_dir / "audio_bpe_tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Audio BPE tokenizer not found at {tokenizer_path}. Run with --launcher dataset first to generate it."
            )
        return Tokenizer.from_file(str(tokenizer_path))

    @property
    def audio_bpe_decode_table(self) -> Array:
        return self._audio_tables[0]

    @property
    def audio_bpe_span_table(self) -> Array:
        return self._audio_tables[1]

    @functools.cached_property
    def _audio_tables(self) -> tuple[Array, Array]:
        """Build lookup tables for decoding Q0-only audio BPE tokens.

        Each BPE token maps to a sequence of Q0 (semantic) codes.
        """
        tokenizer = self.audio_bpe_tokenizer
        vocab_size = tokenizer.get_vocab_size()
        base_char = 0xE000

        # Find max span (number of Q0 codes per BPE token)
        max_span = 1
        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)
            if token_str is not None:
                max_span = max(max_span, len(token_str))

        # Build decode table: token_id -> sequence of Q0 codes
        decode_table = np.zeros((vocab_size, max_span), dtype=np.int32)
        span_table = np.zeros(vocab_size, dtype=np.int32)

        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)
            if token_str is None:
                span_table[token_id] = 0
                continue

            # Convert unicode characters back to Q0 codec values
            q0_codes = [ord(c) - base_char for c in token_str]
            span = len(q0_codes)
            span_table[token_id] = span
            decode_table[token_id, :span] = q0_codes

        audio_bpe_decode_table = jnp.array(decode_table)
        audio_bpe_span_table = jnp.array(span_table)
        logger.info(
            "Built Q0-only audio BPE decode tables: vocab_size=%d, max_span=%d",
            vocab_size,
            max_span,
        )
        return audio_bpe_decode_table, audio_bpe_span_table

    @functools.cached_property
    def _padding_lengths(self) -> tuple[int, int]:
        """Padding lengths for the dataset."""
        tokenizer_path = self.dataset_cache_dir / "maximum_lengths.json"
        with open(tokenizer_path, "r") as f:
            data = json.load(f)
        return data["max_seq_len"], data["max_audio_frames"]

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
            extra_tokens=self.config.audio_bpe_vocab_size,
            tied_extra_embed=True,
            key=llm_key,
        )

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
        model = FullTTSModel.build(
            llm=llm,
            whisper_eos_token_id=self.whisper_config.eos_token_id,
            residual_head_dim=self.config.residual_head_dim,
            residual_num_heads=self.config.residual_num_heads,
            residual_num_layers=self.config.residual_num_layers,
            residual_mlp_dim=self.config.residual_mlp_dim,
            key=mimi_key,
        )

        return model

    @override
    def get_trainable_filter_spec(self, model: FullTTSModel) -> FullTTSModel:
        # Stage 1: Semantic model
        # LLM LoRA parameters.
        llm_spec = xax.lora_filter_spec(model.semantic.llm)

        # Extra head and embeddings should be trainable too.
        extra_embed_spec = jax.tree.map(lambda _: True, llm_spec.extra_embed)
        llm_spec = eqx.tree_at(lambda m: m.extra_embed, llm_spec, extra_embed_spec)

        semantic_spec = SemanticTTSModel(llm=llm_spec)

        # Stage 2: Residual model - all trainable except frozen Mimi codebook embeddings
        residual_spec = jax.tree.map(eqx.is_inexact_array, model.residual)

        # Mimi and Whisper: Frozen
        mimi_spec = jax.tree.map(lambda _: False, model.mimi)
        whisper_spec = jax.tree.map(lambda _: False, model.whisper_transcriber)

        return FullTTSModel(
            semantic=semantic_spec,
            residual=residual_spec,
            mimi=mimi_spec,
            whisper_transcriber=whisper_spec,
        )

    @override
    def get_optimizer(self) -> xax.Optimizer:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=self.config.learning_rate,
            transition_steps=self.config.warmup_steps,
        )
        constant_schedule = optax.constant_schedule(
            value=self.config.learning_rate,
        )
        learning_rate_schedule = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule],
            boundaries=[self.config.warmup_steps],
        )
        return optax.adamw(learning_rate=learning_rate_schedule, weight_decay=0.001)

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
        audio_codes_btf = batch["audio_codes"]  # (B, T, 8)

        # Stage 1: Semantic BPE prediction loss
        tokens_bs, targets_bs = codes_bs[:, :-1], codes_bs[:, 1:]
        mask_bs = targets_bs > self.text_end_id

        # Forward pass for semantic model (Stage 1)
        stage1_loss, stage1_acc, hidden_bsd = jax.vmap(
            model.semantic.llm.get_loss_and_accuracy,
            in_axes=(0, 0, None, 0, None),
        )(tokens_bs, targets_bs, None, mask_bs, 32)
        stage1_loss = stage1_loss.mean()
        stage1_acc = stage1_acc.mean()

        # Compute Stage 2 loss for each sample
        stage2_loss, stage2_acc = self._compute_stage2_loss_batch(
            model,
            codes_bs,
            audio_codes_btf,
            hidden_bsd,
        )
        stage2_loss = stage2_loss.mean()
        stage2_acc = stage2_acc.mean()

        # Weighted total loss allows balancing semantic/acoustic priorities.
        weighted_stage1_loss = self.config.semantic_loss_weight * stage1_loss
        weighted_stage2_loss = self.config.acoustic_loss_weight * stage2_loss
        total_loss = weighted_stage1_loss + weighted_stage2_loss
        metrics: dict[str, xax.Metric] = {
            "loss": xax.Scalar(total_loss),
            "stage1_loss": xax.Scalar(stage1_loss),
            "stage2_loss": xax.Scalar(stage2_loss),
            "weighted_stage1_loss": xax.Scalar(weighted_stage1_loss),
            "weighted_stage2_loss": xax.Scalar(weighted_stage2_loss),
            "unweighted_total_loss": xax.Scalar(stage1_loss + stage2_loss),
            "stage1_accuracy": xax.Scalar(stage1_acc),
            "stage2_accuracy": xax.Scalar(stage2_acc),
        }

        if heavy:
            audio_t, gt_audio_t, gt_text_t = self._generate_audio(model, batch, key)
            metrics["generated_audio"] = xax.Audio(audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["real_audio"] = xax.Audio(gt_audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)

            # Transcribe generated audio with Whisper
            transcript_tokens, _, _ = model.whisper_transcriber.transcribe(audio_t, max_tokens=64)
            metrics["transcript"] = xax.Tokens(transcript_tokens, tokenizer="whisper")

            # Also transcribe ground truth
            gt_transcript_tokens, _, _ = model.whisper_transcriber.transcribe(gt_audio_t, max_tokens=64)
            metrics["gt_transcript"] = xax.Tokens(gt_transcript_tokens, tokenizer="whisper")
            metrics["gt_text"] = xax.Tokens(gt_text_t, tokenizer="llm")

        return total_loss, metrics

    def _compute_stage2_loss_batch(
        self,
        model: FullTTSModel,
        codes_bs: Array,
        audio_codes_btf: Array,
        hidden_bsd: Array,
    ) -> tuple[Array, Array]:
        """Compute Stage 2 (residual) loss for a batch.

        Args:
            model: Full TTS model
            codes_bs: BPE token sequence (includes text + audio), shape (B, S)
            audio_codes_btf: Full audio codes, shape (B, T, 8)
            hidden_bsd: Hidden states from semantic model, shape (B, S, D)

        Returns:
            Stage 2 loss and accuracy across batches.
        """
        bsz = codes_bs.shape[0]
        max_frames = audio_codes_btf.shape[1]  # Use actual batch dimension
        max_bpe_len = codes_bs.shape[1]

        def compute_sample_loss(sample_idx: int) -> tuple[Array, Array]:
            codes_s = codes_bs[sample_idx]
            audio_codes_tf = audio_codes_btf[sample_idx]  # (T, 8)
            hidden_sd = hidden_bsd[sample_idx]

            # Find audio BPE tokens (after AUDIO_START token)
            audio_start_mask = codes_s == self.audio_start_id
            audio_start_idx = jnp.where(audio_start_mask.any(), jnp.argmax(audio_start_mask), max_bpe_len)
            audio_end_mask = codes_s == self.audio_end_id
            audio_end_idx = jnp.where(audio_end_mask.any(), jnp.argmax(audio_end_mask), max_bpe_len)

            # Extract audio BPE tokens (excluding special tokens)
            bpe_start = audio_start_idx + 1
            bpe_length = jnp.maximum(audio_end_idx - bpe_start, 0)

            # Get BPE tokens (shifted to 0-indexed for table lookup)
            audio_bpe_s = jnp.roll(codes_s, -bpe_start)[:max_bpe_len] - self.first_audio_bpe_id

            # Stretch hidden states to Q0 frame positions
            # Get hidden states starting from audio_start position
            audio_hidden_sd = jnp.roll(hidden_sd, -bpe_start, axis=0)[:max_bpe_len]
            stretched_hidden_td = stretch_hidden_states(
                audio_hidden_sd,
                audio_bpe_s,
                self.audio_bpe_span_table,
                bpe_length,
                max_frames,
            )

            # Audio codes transposed for easier access: (8, T)
            audio_codes_ft = audio_codes_tf.T

            # Compute loss for all residual layers using the new unified method
            total_loss, total_accuracy = model.residual.compute_loss(
                audio_codes_ft=audio_codes_ft,
                stretched_hidden_td=stretched_hidden_td,
            )

            # Average loss across layers (7 layers)
            return total_loss, total_accuracy

        # Compute loss for each sample in batch
        sample_losses, sample_accuracies = jax.vmap(compute_sample_loss)(jnp.arange(bsz))

        return sample_losses, sample_accuracies

    def _generate_audio(
        self,
        model: FullTTSModel,
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array, Array]:
        """Generate audio using two-stage pipeline.

        Stage 1: Generate Q0 BPE tokens and get hidden states
        Stage 2: Generate Q1-Q7 codes layer by layer

        Returns:
            Tuple of (generated audio, ground truth audio, ground truth text)
        """
        k1, k2 = jax.random.split(key)
        max_frames = self.max_audio_frames

        # Stage 1: Generate audio BPE tokens from the eval prompt
        max_audio_bpe = self.max_seq_length - len(self.eval_prompt_tokens)
        audio_bpe_tokens, audio_bpe_length = model.semantic.generate_audio(
            prompt_tokens_s=self.eval_prompt_tokens,
            max_audio_tokens=max_audio_bpe,
            audio_end_id=self.audio_end_id,
            key=k1,
        )

        # Strip prompt prefix and EOS from generated tokens
        prompt_len = len(self.eval_prompt_tokens)
        gen_tokens = audio_bpe_tokens[prompt_len:]
        gen_length = audio_bpe_length - prompt_len

        # Exclude EOS token if generated
        last_idx = jnp.clip(gen_length - 1, 0)
        has_eos = gen_tokens[last_idx] == self.audio_end_id
        gen_length = gen_length - has_eos.astype(jnp.int32)

        # Decode BPE to Q0 codes
        bpe_tokens_for_decode = gen_tokens - self.first_audio_bpe_id
        q0_codes_t, num_frames = decode_bpe_to_q0(
            bpe_tokens_b=bpe_tokens_for_decode,
            bpe_length=gen_length,
            decode_table_vs=self.audio_bpe_decode_table,
            span_table_v=self.audio_bpe_span_table,
            max_frames=max_frames,
        )

        # Get hidden states from semantic model for the full generated sequence
        # Use static slicing with max length to avoid dynamic shape issues in JIT
        hidden_sd = model.semantic.llm.forward_hidden(audio_bpe_tokens)

        # Stretch hidden states to Q0 frame positions
        # Use hidden states from the audio portion only (after prompt)
        audio_hidden_sd = hidden_sd[prompt_len:]
        stretched_hidden_td = stretch_hidden_states(
            audio_hidden_sd,
            bpe_tokens_for_decode,
            self.audio_bpe_span_table,
            gen_length,
            max_frames,
        )

        # Stage 2: Generate Q1-Q7 codes layer by layer
        all_codes_ft = model.residual.generate_codes(
            q0_codes_t=q0_codes_t,
            stretched_hidden_td=stretched_hidden_td,
            num_frames=num_frames,
            max_frames=max_frames,
            key=k2,
        )

        # Decode with Mimi
        all_codes_ft = all_codes_ft.clip(max=xax.MIMI_CODEBOOK_SIZE - 1)
        audio_gen = model.mimi.decode(all_codes_ft)

        # Get ground truth audio from batch.
        gt_codes_tf = batch["audio_codes"][0]  # (T, 8)
        gt_codes_ft = gt_codes_tf[1:max_frames, :].T
        gt_codes_ft = gt_codes_ft.clip(max=xax.MIMI_CODEBOOK_SIZE - 1)
        audio_gt = model.mimi.decode(gt_codes_ft)

        # Parse ground truth text
        gt_ids = batch["codes"][0]
        gt_text_ids = jax.lax.dynamic_slice(gt_ids, (1,), (gt_ids.shape[0] - 1,))

        return audio_gen[0], audio_gt[0], gt_text_ids

    @override
    def decode_tokens(self, tokens: np.ndarray, token_type: str) -> str:
        token_list: list[int] = tokens.tolist()

        match token_type:
            case "whisper":
                transcript_tokens = [t for t in token_list[4:]]
                return self.whisper_tokenizer.decode(transcript_tokens, skip_special_tokens=True)

            case "llm":
                transcript_tokens = [t for t in token_list if t < self.first_audio_bpe_id]
                return self.tokenizer.decode(transcript_tokens, skip_special_tokens=True)

            case _:
                raise ValueError(f"Invalid token type: {token_type}")

    @override
    def get_dataset(self) -> Dataset:
        return cast(Dataset, self.load_dataset("train"))

    @xax.dataset_fn("train", dependencies=["unpadded"], use_hash=False)
    def train_dataset(self) -> Dataset:
        """Pads sequences to 95th percentile length for efficient batching.

        Columns:
        - codes: Padded text + audio token IDs, shape (max_seq_len,)
        - audio_codes: Padded audio codes, shape (max_audio_frames, 8)
        """
        ds = cast(Dataset, self.load_dataset("unpadded"))

        code_lengths = np.array([len(c) for c in ds["codes"]])
        audio_lengths = np.array([len(c) for c in ds["audio_codes"]])
        max_seq_len = int(np.percentile(code_lengths, self.config.length_percentile * 100))
        max_audio_frames = int(np.percentile(audio_lengths, self.config.length_percentile * 100))

        tokenizer_path = self.dataset_cache_dir / "maximum_lengths.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tokenizer_path, "w") as f:
            json.dump({"max_seq_len": max_seq_len, "max_audio_frames": max_audio_frames}, f)

        logger.info(
            "Padding to 95th percentile: max_seq_len=%d, max_audio_frames=%d",
            max_seq_len,
            max_audio_frames,
        )

        pre_length = len(ds)
        ds = ds.filter(
            lambda example: len(example["codes"]) <= max_seq_len and len(example["audio_codes"]) <= max_audio_frames,
            desc="Filtering to 95th percentile",
        )
        post_length = len(ds)
        logger.info("Filtered %d examples to %d", pre_length, post_length)

        def pad_sample(example: dict) -> dict:
            codes_raw = np.asarray(example["codes"])
            audio_codes_raw = np.asarray(example["audio_codes"])

            # Truncate and pad codes
            seq_len = min(len(codes_raw), max_seq_len)
            codes = np.full(max_seq_len, self.tokenizer.pad_token_id, dtype=np.int32)
            codes[:seq_len] = codes_raw[:seq_len]

            # Truncate and pad audio codes
            num_frames = min(len(audio_codes_raw), max_audio_frames)
            audio_codes = np.full((max_audio_frames, NUM_QUANTIZERS), AUDIO_PAD_TOKEN_ID, dtype=np.int32)
            audio_codes[:num_frames] = audio_codes_raw[:num_frames]

            return {"codes": codes, "audio_codes": audio_codes}

        ds = cast(Dataset, ds.map(pad_sample, desc="Padding to 95th percentile"))

        return ds

    @xax.dataset_fn("unpadded", dependencies=["bpe"], use_hash=False)
    def unpadded_dataset(self) -> Dataset:
        """Creates unpadded sequences for training.

        Format: [TEXT_START] [TEXT] [TEXT_END] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]

        Columns:
        - codes: Combined text + audio token IDs (variable length)
        - audio_codes: Full audio codes (T, 8) for Stage 2 training (variable length)
        """
        ds = cast(Dataset, self.load_dataset("bpe"))

        def prepare_sample(example: dict) -> dict:
            # Prepare tokens
            text_tokens = np.array(example["text_tokens"], dtype=np.int32)
            text_tokens_with_special = np.concatenate([[self.text_start_id], text_tokens, [self.text_end_id]])
            audio_bpe = np.array(example["audio_bpe"], dtype=np.int32) + self.first_audio_bpe_id
            audio_bpe_with_special = np.concatenate([[self.audio_start_id], audio_bpe, [self.audio_end_id]])

            seq_parts = [text_tokens_with_special, audio_bpe_with_special]
            codes = np.concatenate(seq_parts)

            # Keep full audio codes for Stage 2 training
            audio_codes = np.asarray(example["audio_codes"])  # (T, 8)
            audio_codes = np.pad(audio_codes, ((1, 0), (0, 0)), mode="constant", constant_values=AUDIO_BOS_TOKEN_ID)
            audio_codes = np.pad(audio_codes, ((0, 1), (0, 0)), mode="constant", constant_values=AUDIO_EOS_TOKEN_ID)

            return {"codes": codes, "audio_codes": audio_codes}

        result = cast(Dataset, ds.map(prepare_sample, desc="Preparing unpadded sequences"))

        # Remove unused columns
        cols_to_keep = ["codes", "audio_codes"]
        cols_to_remove = [c for c in result.column_names if c not in cols_to_keep]
        if cols_to_remove:
            result = result.remove_columns(cols_to_remove)

        return result

    @xax.dataset_fn("bpe", dependencies=["tokenized"], use_hash=False)
    def bpe_dataset(self) -> Dataset:
        """Learns a BPE tokenizer on Q0 (semantic) codec tokens only and applies it.

        BPE is learned only on the first quantizer (Q0) which captures semantic content.
        This allows for better compression of the semantic structure while the acoustic
        details (Q1-Q7) are handled separately by the residual model.

        Returns dataset with columns:
        - text_tokens: shape (T_text,) - original text tokens
        - audio_codes: shape (T_audio, 8) - original audio codes (all quantizers)
        - audio_bpe: shape (T_bpe,) - BPE-encoded Q0 codes
        - bpe_spans: shape (T_bpe,) - number of Q0 codes each BPE token represents
        """
        ds = cast(Dataset, self.load_dataset("tokenized"))

        # Map Q0 codec values to unique unicode characters
        base_char = 0xE000

        def codes_to_chars_q0_only(codes_tf: np.ndarray) -> str:
            """Convert only Q0 codes to unicode characters.

            Args:
                codes_tf: Codec tokens, shape (T, 8)

            Returns:
                String where each character represents one Q0 codec value
            """
            q0_codes_t = codes_tf[:, 0]  # Extract first column (Q0) only
            return "".join(chr(base_char + int(c)) for c in q0_codes_t)

        # Step 1: Extract Q0 codec tokens and convert to character strings
        logger.info("Extracting Q0 codec tokens for BPE training...")

        def get_audio_chars(example: dict) -> dict:
            audio_codes = np.asarray(example["audio_codes"])
            audio_chars = codes_to_chars_q0_only(audio_codes)
            return {"audio_chars": audio_chars}

        ds_with_chars = cast(Dataset, ds.map(get_audio_chars, desc="Extracting Q0 codec tokens"))

        # Step 2: Train BPE tokenizer on Q0 codes only
        logger.info("Training BPE tokenizer on Q0 (semantic) codec tokens...")

        # Create initial alphabet from all possible Q0 codec values (0-2047)
        initial_alphabet = [chr(base_char + i) for i in range(xax.MIMI_CODEBOOK_SIZE)]

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])

        bpe_vocab_size = self.config.audio_bpe_vocab_size
        trainer = trainers.BpeTrainer(
            vocab_size=bpe_vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )

        def batch_iterator(batch_size: int = 1000) -> Iterator[list[str]]:
            for idx in range(0, len(ds_with_chars), batch_size):
                yield ds_with_chars[idx : idx + batch_size]["audio_chars"]

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        logger.info("Q0-only audio BPE tokenizer trained with vocab size %d", tokenizer.get_vocab_size())

        # Save tokenizer
        tokenizer_path = self.dataset_cache_dir / "audio_bpe_tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        logger.info("Saved Q0-only audio BPE tokenizer to %s", tokenizer_path)

        # Step 3: Apply BPE encoding
        logger.info("Applying BPE encoding to dataset...")

        def apply_bpe(example: dict) -> dict:
            audio_codes = np.asarray(example["audio_codes"])
            audio_chars = codes_to_chars_q0_only(audio_codes)

            encoding = tokenizer.encode(audio_chars)
            audio_bpe = np.array(encoding.ids, dtype=np.int32)
            bpe_spans = np.array([len(token) for token in encoding.tokens], dtype=np.int32)

            return {"audio_bpe": audio_bpe, "bpe_spans": bpe_spans}

        ds_bpe = cast(Dataset, ds.map(apply_bpe, desc="Applying BPE"))

        # Remove temporary columns
        if "audio_chars" in ds_bpe.column_names:
            ds_bpe = cast(Dataset, ds_bpe.remove_columns(["audio_chars"]))

        # Log compression statistics
        # For Q0-only, original count is just the number of Q0 tokens (T frames)
        audio_codes_column = cast(list[object], ds_bpe["audio_codes"])
        audio_bpe_column = cast(list[object], ds_bpe["audio_bpe"])
        total_original_q0 = sum(len(np.asarray(audio_codes)) for audio_codes in audio_codes_column)
        total_bpe = sum(len(np.asarray(audio_bpe)) for audio_bpe in audio_bpe_column)
        compression_ratio = total_original_q0 / total_bpe if total_bpe > 0 else 0
        logger.info(
            "Q0-only BPE compression: %d -> %d tokens (%.2fx compression)",
            total_original_q0,
            total_bpe,
            compression_ratio,
        )

        return ds_bpe

    @xax.dataset_fn("tokenized", use_hash=False)
    def tokenized_dataset(self) -> Dataset:
        """Tokenizes the LJSpeech dataset text and audio."""
        columns = ["text_tokens", "audio_codes"]

        # Load raw dataset
        logger.info("Loading LJSpeech dataset...")
        raw_ds = load_dataset("keithito/lj_speech", split="train")

        # Stage 1: CPU-parallel audio resampling
        def resample_audio(example: dict) -> dict:
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]

            if sr != xax.MIMI_SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=xax.MIMI_SAMPLE_RATE)

            audio = audio.astype(np.float32)
            max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
            audio = audio / max_val

            return {"resampled_audio": audio, "audio_length": len(audio)}

        logger.info("Stage 1: Resampling audio on CPU (parallel)...")
        resampled_ds = cast(Dataset, raw_ds.map(resample_audio, num_proc=32, desc="Resampling audio"))

        # Stage 2: GPU-batched Mimi encoding
        logger.info("Stage 2: Encoding audio with Mimi on GPU (batched)...")
        mimi = xax.build_pretrained_mimi(dtype=jnp.bfloat16)
        num_quantizers = self.config.num_quantizers

        max_audio_len = max(resampled_ds["audio_length"])
        logger.info("Max audio length: %d samples", max_audio_len)

        @jax.jit
        def batch_encode(audio_bct: Array) -> Array:
            return jax.vmap(lambda audio_ct: mimi.encode(audio_ct, num_quantizers=num_quantizers))(audio_bct)

        def encode_and_tokenize_batch(examples: dict[str, list]) -> dict[str, list]:
            outputs: dict[str, list] = {column: [] for column in columns}
            audio_list = examples["resampled_audio"]
            bsz = len(audio_list)

            audio_batch_np = np.zeros(shape=(bsz, max_audio_len), dtype=np.float32)
            for idx, audio in enumerate(audio_list):
                audio_batch_np[idx, : len(audio)] = audio
            audio_batch = jnp.array(audio_batch_np, dtype=jnp.bfloat16)

            codes_batch_bct = batch_encode(audio_batch[:, None, :])
            codes_batch_np = np.array(codes_batch_bct)

            for idx in range(bsz):
                audio_codes_ct = codes_batch_np[idx]
                orig_audio_len = examples["audio_length"][idx]

                estimated_frames = (orig_audio_len + 319) // 320
                actual_mimi_frames = audio_codes_ct.shape[1]
                valid_frames = min(estimated_frames, actual_mimi_frames)

                audio_codes_tc = audio_codes_ct[:, :valid_frames].T

                text = examples["normalized_text"][idx]
                text_tokens = np.array(self.tokenizer.encode(text, add_special_tokens=True), dtype=np.int32)

                outputs["text_tokens"].append(text_tokens)
                outputs["audio_codes"].append(audio_codes_tc.astype(np.int32))

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
        logger.info("Columns: text_tokens (T,), audio_codes (T, %d)", num_quantizers)
        return ds


if __name__ == "__main__":
    LJSpeechTTS.launch(
        Config(
            batch_size=16,
            max_grad_norm=2.0,
            gradient_accumulation_steps=1,
            log_heavy_every_n_seconds=300,
            step_kind="second",
        ),
    )
