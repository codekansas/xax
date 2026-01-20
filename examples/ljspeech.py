#!/usr/bin/env -S uv run --no-project --script
"""Fine-tuning Qwen3 for text-to-speech using flattened codec tokens with BPE.

Architecture:
- Single LLM with LoRA fine-tuning
- Sequence format: [TEXT_START] [TEXT] [TEXT_END] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]
- BPE compression on flattened codec tokens (Q0-Q7 interleaved)
- Audio embeddings and output head trained separately from text embeddings/logits
"""

import logging
from dataclasses import dataclass
from typing import Iterator, TypedDict, override

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

logger = logging.getLogger(__name__)

# Mimi constants
LJSPEECH_SAMPLE_RATE = 22050
NUM_QUANTIZERS = 8  # Q0 (semantic) + Q1-Q7 (acoustic)

# Special token string markers (for text tokenizer)
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

# Audio BPE special token IDs
AUDIO_BPE_PAD_TOKEN = 0
AUDIO_BPE_UNK_TOKEN = 1
AUDIO_BPE_BOS_TOKEN = 2
AUDIO_BPE_EOS_TOKEN = 3

# LoRA targets
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "gate", "up")


class Batch(TypedDict):
    input_ids: Array  # (batch, seq_len) - combined text + audio tokens
    labels: Array  # (batch, seq_len) - targets for next-token prediction
    loss_mask: Array  # (batch, seq_len) - mask for loss computation (audio only)
    audio_mask: Array  # (batch, seq_len) - mask indicating audio token positions
    audio_start_pos: Array  # (batch,) - position where audio starts
    audio_codes: Array  # (batch, max_audio_frames, 8) - original audio codes for decoding


class TTSModel(eqx.Module):
    """Single LLM for text-to-speech with separate audio embeddings and head.

    The model processes sequences of the form:
    [TEXT_START] [TEXT] [TEXT_END] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]

    Text tokens use the LLM's original embeddings/output head (frozen).
    Audio BPE tokens use separate learned embeddings/output head.
    """

    llm: xax.LLM  # Base LLM with LoRA
    audio_embed: eqx.nn.Embedding  # Embeddings for audio BPE tokens
    audio_head: eqx.nn.Linear  # Output projection for audio BPE tokens
    audio_bpe_vocab_size: int = eqx.field(static=True)
    text_vocab_size: int = eqx.field(static=True)

    @staticmethod
    def build(
        llm: xax.LLM,
        audio_bpe_vocab_size: int,
        key: PRNGKeyArray,
    ) -> "TTSModel":
        """Build TTS model from pretrained LLM.

        Args:
            llm: Pretrained LLM (can have LoRA applied)
            audio_bpe_vocab_size: Vocabulary size for audio BPE tokens
            key: PRNG key for initialization

        Returns:
            TTSModel instance
        """
        k1, k2 = jax.random.split(key)
        embed_dim = llm.config.embed_dim

        # Audio embedding and output head (not tied, since BPE vocab differs from LLM vocab)
        audio_embed = eqx.nn.Embedding(audio_bpe_vocab_size, embed_dim, key=k1)
        audio_head = eqx.nn.Linear(embed_dim, audio_bpe_vocab_size, key=k2)

        return TTSModel(
            llm=llm,
            audio_embed=audio_embed,
            audio_head=audio_head,
            audio_bpe_vocab_size=audio_bpe_vocab_size,
            text_vocab_size=llm.config.vocab_size,
        )

    def embed_tokens(
        self,
        input_ids_t: Array,
        audio_mask_t: Array,
    ) -> Array:
        """Embed tokens, using text embeddings for text and audio embeddings for audio.

        Args:
            input_ids_t: Token IDs, shape (T,). Text tokens are in [0, text_vocab_size),
                audio tokens are offset by text_vocab_size.
            audio_mask_t: Boolean mask, True for audio positions, shape (T,)

        Returns:
            Embeddings, shape (T, embed_dim)
        """
        # For text tokens, use the original LLM embeddings
        text_embeds_td = jax.vmap(self.llm.embed)(input_ids_t)

        # For audio tokens, subtract text_vocab_size to get audio BPE token ID,
        # then look up in audio embeddings
        audio_ids_t = jnp.maximum(input_ids_t - self.text_vocab_size, 0)
        audio_embeds_td = jax.vmap(self.audio_embed)(audio_ids_t)

        # Select based on audio_mask
        embeds_td = jnp.where(audio_mask_t[:, None], audio_embeds_td, text_embeds_td)
        return embeds_td

    def forward(
        self,
        input_ids_t: Array,
        audio_mask_t: Array,
    ) -> tuple[Array, Array]:
        """Forward pass returning separate text and audio logits.

        Args:
            input_ids_t: Token IDs, shape (T,)
            audio_mask_t: Boolean mask for audio positions, shape (T,)

        Returns:
            Tuple of (text_logits, audio_logits) where:
            - text_logits: shape (T, text_vocab_size)
            - audio_logits: shape (T, audio_bpe_vocab_size)
        """
        # Get embeddings (mixed text/audio)
        x_td = self.embed_tokens(input_ids_t, audio_mask_t)

        # Run through LLM transformer blocks
        for block in self.llm.blocks:
            x_td, _ = block.forward(x_td, cache=None)

        # Apply final norm
        x_td = self.llm.norm(x_td)

        # Compute both text and audio logits
        text_logits_tv = self.llm.lm_head(x_td)
        audio_logits_ta = jax.vmap(self.audio_head)(x_td)

        return text_logits_tv, audio_logits_ta

    def generate_audio(
        self,
        text_ids_s: Array,
        audio_start_token: int,
        max_audio_tokens: int,
        eos_token: int = AUDIO_BPE_EOS_TOKEN,
    ) -> Array:
        """Generate audio BPE tokens given text prefix.

        Args:
            text_ids_s: Text token IDs including [TEXT_END], shape (S,)
            audio_start_token: Token ID for audio start marker
            max_audio_tokens: Maximum number of audio tokens to generate
            eos_token: Audio EOS token ID

        Returns:
            Generated audio BPE token IDs (without text_vocab_size offset), shape (max_audio_tokens,)
        """
        # Build initial sequence: text + audio_start
        text_len = text_ids_s.shape[0]
        total_len = text_len + 1 + max_audio_tokens  # text + audio_start + audio

        # Initialize sequence
        tokens = jnp.zeros(total_len, dtype=jnp.int32)
        tokens = tokens.at[:text_len].set(text_ids_s)
        tokens = tokens.at[text_len].set(audio_start_token)

        # Audio mask: False for text, True for audio positions
        audio_mask = jnp.zeros(total_len, dtype=jnp.bool_)
        audio_mask = audio_mask.at[text_len + 1 :].set(True)

        # Generate autoregressively
        def generate_step(idx: int, tokens_t: Array) -> Array:
            # Forward pass
            _, audio_logits_ta = self.forward(tokens_t, audio_mask)

            # Get logits at position idx (predicting idx+1)
            next_logits = audio_logits_ta[idx]

            # Mask special tokens during generation (except EOS)
            next_logits = next_logits.at[AUDIO_BPE_PAD_TOKEN].set(-jnp.inf)
            next_logits = next_logits.at[AUDIO_BPE_UNK_TOKEN].set(-jnp.inf)
            next_logits = next_logits.at[AUDIO_BPE_BOS_TOKEN].set(-jnp.inf)

            next_token = jnp.argmax(next_logits)
            # Store with text_vocab_size offset for proper embedding lookup
            tokens_t = tokens_t.at[idx + 1].set(next_token + self.text_vocab_size)
            return tokens_t

        # Generate audio tokens
        audio_start_idx = text_len  # Position of audio_start token
        tokens = jax.lax.fori_loop(
            audio_start_idx,
            audio_start_idx + max_audio_tokens - 1,
            generate_step,
            tokens,
        )

        # Extract audio tokens (remove text_vocab_size offset)
        audio_tokens = tokens[text_len + 1 :] - self.text_vocab_size

        return audio_tokens


class FullTTSModel(eqx.Module):
    """Full TTS model with Mimi decoder and Whisper for evaluation."""

    tts: TTSModel
    mimi: xax.MimiModel
    whisper_transcriber: xax.WhisperTranscriber

    @staticmethod
    def build(
        llm: xax.LLM,
        audio_bpe_vocab_size: int,
        whisper_eos_token_id: int,
        key: PRNGKeyArray,
    ) -> "FullTTSModel":
        tts = TTSModel.build(llm, audio_bpe_vocab_size, key)
        mimi = xax.build_pretrained_mimi()
        whisper_model = xax.build_pretrained_whisper()
        whisper_transcriber = xax.WhisperTranscriber(
            model=whisper_model,
            eos_token_id=whisper_eos_token_id,
        )
        return FullTTSModel(tts=tts, mimi=mimi, whisper_transcriber=whisper_transcriber)


def decode_audio_bpe_to_codes(
    bpe_tokens_b: Array,
    bpe_length: Array,
    decode_table_vs: Array,
    span_table_v: Array,
    num_quantizers: int,
    max_frames: int,
) -> tuple[Array, Array]:
    """Decode audio BPE tokens back to codec tokens.

    Args:
        bpe_tokens_b: Audio BPE token IDs, shape (B,) where B is max BPE length
        bpe_length: Actual number of valid BPE tokens (scalar)
        decode_table_vs: Lookup table mapping token ID -> flattened codes, shape (V, S)
        span_table_v: Lookup table mapping token ID -> span length, shape (V,)
        num_quantizers: Number of quantizers (8)
        max_frames: Maximum number of audio frames

    Returns:
        Tuple of (audio_codes, num_frames) where:
        - audio_codes: Decoded codec tokens, shape (8, max_frames)
        - num_frames: Actual number of frames (scalar)
    """
    # Look up spans and flattened codes for each BPE token
    spans_b = span_table_v[bpe_tokens_b]  # (B,)
    code_seqs_bs = decode_table_vs[bpe_tokens_b]  # (B, S) - flattened codes

    # Compute cumulative positions
    cumsum_b = jnp.cumsum(spans_b)
    start_positions_b = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), cumsum_b[:-1]])

    # Total flattened length
    total_len = jnp.sum(jnp.where(jnp.arange(len(bpe_tokens_b)) < bpe_length, spans_b, 0))
    num_frames = total_len // num_quantizers

    # Initialize flattened output
    max_flat_len = max_frames * num_quantizers
    flat_codes = jnp.zeros(max_flat_len, dtype=jnp.int32)

    # Scatter codes from each BPE token
    max_span = decode_table_vs.shape[1]

    def scatter_bpe_token(carry: Array, inputs: tuple[Array, Array, Array, Array]) -> tuple[Array, None]:
        output, (start_pos, span, code_seq, bpe_idx) = carry, inputs
        valid = bpe_idx < bpe_length

        def scatter_positions(pos_in_span: int, out: Array) -> Array:
            target_pos = start_pos + pos_in_span
            valid_pos = valid & (pos_in_span < span) & (target_pos < max_flat_len)
            return jnp.where(valid_pos, out.at[target_pos].set(code_seq[pos_in_span]), out)

        output = jax.lax.fori_loop(0, max_span, scatter_positions, output)
        return output, None

    flat_codes, _ = jax.lax.scan(
        scatter_bpe_token,
        flat_codes,
        (start_positions_b, spans_b, code_seqs_bs, jnp.arange(len(bpe_tokens_b))),
    )

    # Reshape to (num_quantizers, max_frames) - interleaved format
    # flat_codes is [q0_t0, q1_t0, ..., q7_t0, q0_t1, q1_t1, ..., q7_t1, ...]
    audio_codes_tf = flat_codes[: max_frames * num_quantizers].reshape(max_frames, num_quantizers)
    audio_codes_ft = audio_codes_tf.T  # (8, max_frames)

    return audio_codes_ft, num_frames


@dataclass
class Config(xax.SupervisedConfig):
    # Model settings
    llm_repo: xax.LLMRepo = xax.field(xax.LLMRepo.QWEN3_600M, help="Pretrained model")
    num_quantizers: int = xax.field(NUM_QUANTIZERS, help="Number of quantizers (8)")

    # LoRA settings
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(16.0, help="LoRA alpha parameter")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")
    lora_targets: tuple[str, ...] | None = xax.field(DEFAULT_LORA_TARGETS, help="Layer suffixes for LoRA")

    # Training settings
    learning_rate: float = xax.field(5e-4, help="Peak learning rate")
    min_learning_rate: float = xax.field(1e-5, help="Minimum learning rate")
    warmup_steps: int = xax.field(50, help="Number of warmup steps")
    max_text_length: int = xax.field(128, help="Maximum text token length")
    max_audio_frames: int = xax.field(256, help="Maximum audio frames")
    max_seq_length: int = xax.field(512, help="Maximum combined sequence length")

    # Audio BPE settings
    audio_bpe_vocab_size: int = xax.field(151_669, help="Vocabulary size for audio BPE tokenizer")
    max_audio_bpe_length: int = xax.field(256, help="Maximum audio BPE sequence length")

    # Data settings
    processed_data_path: str | None = xax.field(None, help="Path to pre-processed data")


class LJSpeechTTS(xax.SupervisedTask[Config]):
    """Single LLM TTS with flattened codec BPE."""

    tokenizer: Qwen2TokenizerFast
    whisper_tokenizer: WhisperTokenizerFast
    audio_bpe_tokenizer: Tokenizer | None
    audio_bpe_decode_table: Array | None
    audio_bpe_span_table: Array | None

    # Special token IDs (added to tokenizer)
    audio_start_id: int
    audio_end_id: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Load text tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_repo.value)

        # Add audio boundary tokens
        special_tokens = [AUDIO_START_TOKEN, AUDIO_END_TOKEN]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids(AUDIO_START_TOKEN)
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids(AUDIO_END_TOKEN)

        # Load Whisper for ASR evaluation
        logger.info("Loading Whisper model for ASR evaluation")
        path = xax.download_whisper_repo()
        self.whisper_config = xax.load_whisper_config()
        self.whisper_tokenizer = WhisperTokenizerFast.from_pretrained(str(path))

        # Audio BPE tokenizer and decode tables are loaded lazily
        self.audio_bpe_tokenizer = None
        self.audio_bpe_decode_table = None
        self.audio_bpe_span_table = None

    def _load_audio_bpe_tokenizer(self) -> Tokenizer:
        """Load audio BPE tokenizer from cache directory."""
        if self.audio_bpe_tokenizer is None:
            tokenizer_path = self.dataset_cache_dir / "audio_bpe_tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Audio BPE tokenizer not found at {tokenizer_path}. "
                    "Run with --launcher dataset first to generate it."
                )
            self.audio_bpe_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return self.audio_bpe_tokenizer

    def _build_audio_bpe_decode_tables(self) -> tuple[Array, Array]:
        """Build lookup tables for decoding audio BPE tokens to codec tokens."""
        if self.audio_bpe_decode_table is not None and self.audio_bpe_span_table is not None:
            return self.audio_bpe_decode_table, self.audio_bpe_span_table

        tokenizer = self._load_audio_bpe_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        base_char = 0xE000

        # Find max span
        max_span = 1
        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)
            if token_str is not None:
                max_span = max(max_span, len(token_str))

        # Build decode table
        decode_table = np.zeros((vocab_size, max_span), dtype=np.int32)
        span_table = np.zeros(vocab_size, dtype=np.int32)

        special_token_ids = {AUDIO_BPE_PAD_TOKEN, AUDIO_BPE_UNK_TOKEN, AUDIO_BPE_BOS_TOKEN, AUDIO_BPE_EOS_TOKEN}

        for token_id in range(vocab_size):
            if token_id in special_token_ids:
                span_table[token_id] = 0
                continue

            token_str = tokenizer.id_to_token(token_id)
            if token_str is None:
                span_table[token_id] = 0
                continue

            # Convert unicode characters back to flattened codec tokens
            flat_codes = [ord(c) - base_char for c in token_str]
            span = len(flat_codes)
            span_table[token_id] = span
            decode_table[token_id, :span] = flat_codes

        self.audio_bpe_decode_table = jnp.array(decode_table)
        self.audio_bpe_span_table = jnp.array(span_table)
        logger.info(
            "Built audio BPE decode tables: vocab_size=%d, max_span=%d",
            vocab_size,
            max_span,
        )
        return self.audio_bpe_decode_table, self.audio_bpe_span_table

    @override
    def get_model(self, params: xax.InitParams) -> FullTTSModel:
        # Build LLM with LoRA
        llm = xax.build_pretrained_llm(self.config.llm_repo)

        # Resize embeddings to account for new special tokens
        new_vocab_size = len(self.tokenizer)
        if new_vocab_size > llm.config.vocab_size:
            logger.info("Resizing LLM embeddings from %d to %d", llm.config.vocab_size, new_vocab_size)
            # Create new embedding matrix with random initialization for new tokens
            old_embed = llm.embed.weight
            k_embed = jax.random.fold_in(params.key, 42)
            new_rows = jax.random.normal(k_embed, (new_vocab_size - llm.config.vocab_size, llm.config.embed_dim)) * 0.02
            new_embed_weight = jnp.concatenate([old_embed, new_rows], axis=0)
            new_embed = eqx.nn.Embedding(new_vocab_size, llm.config.embed_dim, weight=new_embed_weight)

            # Update lm_head as well
            old_head_weight = llm.lm_head.weight
            old_head_bias = llm.lm_head.bias
            k_head = jax.random.fold_in(params.key, 43)
            new_head_rows = jax.random.normal(k_head, (new_vocab_size - llm.config.vocab_size, llm.config.embed_dim))
            new_head_rows = new_head_rows * 0.02
            new_head_weight = jnp.concatenate([old_head_weight, new_head_rows], axis=0)
            new_head_bias_rows = jnp.zeros(new_vocab_size - llm.config.vocab_size)
            new_head_bias = jnp.concatenate([old_head_bias, new_head_bias_rows]) if old_head_bias is not None else None
            new_head = eqx.nn.Linear(
                llm.config.embed_dim, new_vocab_size, key=k_head, use_bias=new_head_bias is not None
            )
            new_head = eqx.tree_at(lambda h: h.weight, new_head, new_head_weight)
            if new_head_bias is not None:
                new_head = eqx.tree_at(lambda h: h.bias, new_head, new_head_bias)

            # Update config
            new_config = eqx.tree_at(lambda c: c.vocab_size, llm.config, new_vocab_size)
            llm = eqx.tree_at(lambda m: (m.embed, m.lm_head, m.config), llm, (new_embed, new_head, new_config))

        llm = xax.loraize_by_path(
            llm,
            rank=self.config.lora_rank,
            include_suffixes=list(self.config.lora_targets) if self.config.lora_targets else None,
            alpha=self.config.lora_alpha,
            dropout_rate=self.config.lora_dropout,
            key=params.key,
        )

        # Load audio BPE tokenizer to get vocab size
        audio_bpe_tokenizer = self._load_audio_bpe_tokenizer()
        audio_bpe_vocab_size = audio_bpe_tokenizer.get_vocab_size()

        k_model = jax.random.fold_in(params.key, 1)
        model = FullTTSModel.build(
            llm=llm,
            audio_bpe_vocab_size=audio_bpe_vocab_size,
            whisper_eos_token_id=self.whisper_config.eos_token_id,
            key=k_model,
        )

        return model

    @override
    def get_trainable_filter_spec(self, model: FullTTSModel) -> FullTTSModel:
        # LLM: Only LoRA parameters are trainable (embeddings and head frozen)
        llm_spec = xax.lora_filter_spec(model.tts.llm)

        # Audio embeddings and head: trainable
        audio_embed_spec = jax.tree.map(lambda _: True, model.tts.audio_embed)
        audio_head_spec = jax.tree.map(lambda _: True, model.tts.audio_head)

        tts_spec = TTSModel(
            llm=llm_spec,
            audio_embed=audio_embed_spec,
            audio_head=audio_head_spec,
            audio_bpe_vocab_size=model.tts.audio_bpe_vocab_size,
            text_vocab_size=model.tts.text_vocab_size,
        )

        # Mimi and Whisper: Frozen
        mimi_spec = jax.tree.map(lambda _: False, model.mimi)
        whisper_spec = jax.tree.map(lambda _: False, model.whisper_transcriber)

        return FullTTSModel(tts=tts_spec, mimi=mimi_spec, whisper_transcriber=whisper_spec)

    @override
    def get_optimizer(self) -> xax.Optimizer:
        if self.config.max_steps is not None:
            warmup_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=self.config.learning_rate,
                transition_steps=self.config.warmup_steps,
            )
            cosine_schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=max(self.config.max_steps - self.config.warmup_steps, 1),
                alpha=self.config.min_learning_rate / self.config.learning_rate,
            )
            learning_rate_schedule = optax.join_schedules(
                schedules=[warmup_schedule, cosine_schedule],
                boundaries=[self.config.warmup_steps],
            )
        else:
            learning_rate_schedule = self.config.learning_rate

        return optax.adamw(learning_rate=learning_rate_schedule, weight_decay=0.01)

    @override
    def compute_loss(
        self,
        model: FullTTSModel,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        input_ids_bs = batch["input_ids"]
        labels_bs = batch["labels"]
        loss_mask_bs = batch["loss_mask"]
        audio_mask_bs = batch["audio_mask"]

        # Forward pass
        _, audio_logits_bsa = jax.vmap(model.tts.forward)(input_ids_bs, audio_mask_bs)

        # Compute loss only on audio tokens (where loss_mask is True)
        # Shift labels and logits for next-token prediction
        shifted_logits_bsa = audio_logits_bsa[:, :-1, :]
        shifted_labels_bs = labels_bs[:, 1:]
        shifted_mask_bs = loss_mask_bs[:, 1:]

        # Labels for audio positions need offset removed
        audio_labels_bs = shifted_labels_bs - model.tts.text_vocab_size
        audio_labels_bs = jnp.maximum(audio_labels_bs, 0)  # Clamp negative

        loss_bs = optax.softmax_cross_entropy_with_integer_labels(shifted_logits_bsa, audio_labels_bs)
        loss = jnp.sum(loss_bs * shifted_mask_bs) / jnp.maximum(jnp.sum(shifted_mask_bs), 1)

        # Accuracy
        preds_bs = jnp.argmax(shifted_logits_bsa, axis=-1)
        correct_bs = (preds_bs == audio_labels_bs) & shifted_mask_bs
        accuracy = jnp.sum(correct_bs) / jnp.maximum(jnp.sum(shifted_mask_bs), 1)

        metrics: dict[str, xax.Metric] = {
            "loss": xax.Scalar(loss),
            "accuracy": xax.Scalar(accuracy),
        }

        if heavy:
            gen_key = jax.random.fold_in(key, state.num_steps)
            decode_table, span_table = self._build_audio_bpe_decode_tables()
            audio_t, gt_audio_t = self._generate_audio(model, batch, gen_key, decode_table, span_table)
            metrics["generated_audio"] = xax.Audio(audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["real_audio"] = xax.Audio(gt_audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)

            # Transcribe generated audio with Whisper
            transcript_tokens, _, _ = model.whisper_transcriber.transcribe(audio_t, max_tokens=64)
            metrics["transcript"] = xax.Tokens(transcript_tokens, tokenizer="whisper")

            # Also transcribe ground truth
            gt_transcript_tokens, _, _ = model.whisper_transcriber.transcribe(gt_audio_t, max_tokens=64)
            metrics["gt_transcript"] = xax.Tokens(gt_transcript_tokens, tokenizer="whisper")

        return loss, metrics

    def _generate_audio(
        self,
        model: FullTTSModel,
        batch: Batch,
        key: PRNGKeyArray,
        decode_table: Array,
        span_table: Array,
    ) -> tuple[Array, Array]:
        """Generate audio using the model.

        Returns:
            Tuple of (generated audio, ground truth audio)
        """
        # Get first sample
        input_ids = batch["input_ids"][0]
        audio_start_pos = batch["audio_start_pos"][0]
        audio_codes_gt = batch["audio_codes"][0]  # (max_frames, 8)

        # Extract text tokens (up to and including TEXT_END, which is just before audio_start)
        text_ids = input_ids[:audio_start_pos]

        # Generate audio BPE tokens
        max_audio_bpe = self.config.max_audio_bpe_length
        audio_bpe_tokens = model.tts.generate_audio(
            text_ids,
            audio_start_token=self.audio_start_id + model.tts.text_vocab_size,  # Offset for embedding lookup
            max_audio_tokens=max_audio_bpe,
            eos_token=AUDIO_BPE_EOS_TOKEN,
        )

        # Find actual length (first EOS or max)
        eos_positions = jnp.where(audio_bpe_tokens == AUDIO_BPE_EOS_TOKEN, jnp.arange(max_audio_bpe), max_audio_bpe)
        bpe_length = jnp.min(eos_positions)

        # Decode BPE to codec tokens
        audio_codes_gen, num_frames = decode_audio_bpe_to_codes(
            bpe_tokens_b=audio_bpe_tokens,
            bpe_length=bpe_length,
            decode_table_vs=decode_table,
            span_table_v=span_table,
            num_quantizers=self.config.num_quantizers,
            max_frames=self.config.max_audio_frames,
        )

        # Decode with Mimi
        audio_gen = model.mimi.decode(audio_codes_gen)

        # Ground truth audio
        gt_codes = audio_codes_gt.T  # (8, max_frames)
        audio_gt = model.mimi.decode(gt_codes)

        return audio_gen[0], audio_gt[0]

    @override
    def decode_tokens(self, tokens: np.ndarray, token_type: str) -> str:
        token_list: list[int] = tokens.tolist()

        match token_type:
            case "whisper":
                transcript_tokens = [t for t in token_list[4:]]
                return self.whisper_tokenizer.decode(transcript_tokens, skip_special_tokens=True)

            case "llm":
                return self.tokenizer.decode(token_list, skip_special_tokens=False)

            case _:
                raise ValueError(f"Invalid token type: {token_type}")

    @override
    def get_dataset(self) -> Dataset:
        return self.load_dataset("train")

    @xax.dataset_fn("train", dependencies=["bpe"], use_hash=False)
    def train_dataset(self) -> Dataset:
        """Creates the final training dataset.

        Format: [TEXT] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]

        Columns:
        - input_ids: Combined text + audio token IDs (audio offset by text_vocab_size)
        - labels: Same as input_ids (for next-token prediction)
        - loss_mask: True only for audio positions (we only train on audio prediction)
        - audio_mask: True for audio token positions (for embedding selection)
        - audio_start_pos: Position where audio starts
        - audio_codes: Original codec tokens for ground truth decoding
        """
        ds = self.load_dataset("bpe")

        max_text_len = self.config.max_text_length
        max_audio_bpe_len = self.config.max_audio_bpe_length
        max_seq_len = self.config.max_seq_length
        text_vocab_size = len(self.tokenizer)

        def prepare_sample(example: dict) -> dict:
            # Text tokens
            text_tokens = np.array(example["text_tokens"], dtype=np.int32)
            text_len = min(len(text_tokens), max_text_len)
            text_tokens = text_tokens[:text_len]

            # Audio BPE tokens with BOS/EOS
            audio_bpe = np.array(example["audio_bpe"], dtype=np.int32)
            audio_bpe_with_special = np.concatenate([[AUDIO_BPE_BOS_TOKEN], audio_bpe, [AUDIO_BPE_EOS_TOKEN]])
            audio_bpe_len = min(len(audio_bpe_with_special), max_audio_bpe_len)
            audio_bpe_with_special = audio_bpe_with_special[:audio_bpe_len]

            # Build combined sequence: [TEXT] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]
            # Audio tokens are offset by text_vocab_size
            audio_start_id = self.audio_start_id
            audio_end_id = self.audio_end_id

            seq_parts = [
                text_tokens,  # Text
                np.array([audio_start_id], dtype=np.int32),  # AUDIO_START
                audio_bpe_with_special + text_vocab_size,  # Audio BPE (offset)
                np.array([audio_end_id], dtype=np.int32),  # AUDIO_END
            ]
            sequence = np.concatenate(seq_parts)

            # Truncate to max_seq_len
            seq_len = min(len(sequence), max_seq_len)
            sequence = sequence[:seq_len]

            # Pad to max_seq_len
            input_ids = np.zeros(max_seq_len, dtype=np.int32)
            input_ids[:seq_len] = sequence

            # Labels are the same (for next-token prediction)
            labels = input_ids.copy()

            # Audio start position (after text + AUDIO_START)
            audio_start_pos = text_len + 1

            # Loss mask: only on audio tokens (after AUDIO_START, excluding AUDIO_END)
            loss_mask = np.zeros(max_seq_len, dtype=np.bool_)
            audio_end_pos = audio_start_pos + audio_bpe_len
            loss_mask[audio_start_pos:audio_end_pos] = True

            # Audio mask: True for audio token positions (for embedding lookup)
            audio_mask = np.zeros(max_seq_len, dtype=np.bool_)
            audio_mask[audio_start_pos:audio_end_pos] = True

            # Keep original audio codes for ground truth generation
            audio_codes = np.array(example["audio_codes"], dtype=np.int32)
            max_audio_frames = self.config.max_audio_frames
            if len(audio_codes) < max_audio_frames:
                audio_codes = np.pad(audio_codes, ((0, max_audio_frames - len(audio_codes)), (0, 0)))
            else:
                audio_codes = audio_codes[:max_audio_frames]

            return {
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                "audio_mask": audio_mask,
                "audio_start_pos": np.int32(audio_start_pos),
                "audio_codes": audio_codes,
            }

        result = ds.map(prepare_sample, desc="Preparing training data")

        # Remove unused columns
        cols_to_keep = ["input_ids", "labels", "loss_mask", "audio_mask", "audio_start_pos", "audio_codes"]
        cols_to_remove = [c for c in result.column_names if c not in cols_to_keep]
        if cols_to_remove:
            result = result.remove_columns(cols_to_remove)
        return result

    @xax.dataset_fn("bpe", dependencies=["tokenized"], use_hash=False)
    def bpe_dataset(self) -> Dataset:
        """Learns a BPE tokenizer on flattened codec tokens (Q0-Q7) and applies it.

        The codec tokens are flattened by interleaving: [q0_t0, q1_t0, ..., q7_t0, q0_t1, ...]
        This preserves the temporal structure while allowing BPE to learn patterns
        across all quantizer levels.

        Returns dataset with columns:
        - text_tokens: shape (T_text,) - original text tokens
        - audio_codes: shape (T_audio, 8) - original audio codes
        - audio_bpe: shape (T_bpe,) - BPE-encoded flattened audio
        - bpe_spans: shape (T_bpe,) - number of original tokens each BPE token represents
        """
        ds = self.load_dataset("tokenized")

        # Map flattened codec values to unique unicode characters
        # Total possible values: 2048 * 8 = 16384, but we use a simpler scheme:
        # just map each individual codec value (0-2047) to a character
        base_char = 0xE000

        def codes_to_chars(codes_tf: np.ndarray) -> str:
            """Convert codec tokens to a string of unique unicode characters.

            Args:
                codes_tf: Codec tokens, shape (T, 8)

            Returns:
                String where each character represents one codec value (interleaved)
            """
            # Flatten by interleaving: [q0_t0, q1_t0, ..., q7_t0, q0_t1, ...]
            flat = codes_tf.flatten()  # Flattens in row-major order: t0q0, t0q1, ..., t0q7, t1q0, ...
            return "".join(chr(base_char + int(c)) for c in flat)

        # Step 1: Extract flattened codec tokens and convert to character strings
        logger.info("Extracting flattened codec tokens for BPE training...")

        def get_audio_chars(example: dict) -> dict:
            audio_codes = np.asarray(example["audio_codes"])
            audio_chars = codes_to_chars(audio_codes)
            return {"audio_chars": audio_chars}

        ds_with_chars = ds.map(get_audio_chars, desc="Extracting codec tokens")

        # Step 2: Train BPE tokenizer
        logger.info("Training BPE tokenizer on flattened codec tokens...")

        # Create initial alphabet from all possible codec values (0-2047)
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
        logger.info("Audio BPE tokenizer trained with vocab size %d", tokenizer.get_vocab_size())

        # Save tokenizer
        tokenizer_path = self.dataset_cache_dir / "audio_bpe_tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        logger.info("Saved audio BPE tokenizer to %s", tokenizer_path)

        # Step 3: Apply BPE encoding
        logger.info("Applying BPE encoding to dataset...")

        def apply_bpe(example: dict) -> dict:
            audio_codes = np.asarray(example["audio_codes"])
            audio_chars = codes_to_chars(audio_codes)

            encoding = tokenizer.encode(audio_chars)
            audio_bpe = np.array(encoding.ids, dtype=np.int32)
            bpe_spans = np.array([len(token) for token in encoding.tokens], dtype=np.int32)

            return {"audio_bpe": audio_bpe, "bpe_spans": bpe_spans}

        ds_bpe = ds.map(apply_bpe, desc="Applying BPE")

        # Remove temporary columns
        if "audio_chars" in ds_bpe.column_names:
            ds_bpe = ds_bpe.remove_columns(["audio_chars"])

        # Log compression statistics
        total_original = sum(len(np.asarray(ex["audio_codes"]).flatten()) for ex in ds_bpe)
        total_bpe = sum(len(ex["audio_bpe"]) for ex in ds_bpe)
        compression_ratio = total_original / total_bpe if total_bpe > 0 else 0
        logger.info(
            "BPE compression: %d -> %d tokens (%.2fx compression)",
            total_original,
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
        resampled_ds = raw_ds.map(resample_audio, num_proc=32, desc="Resampling audio")

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

        ds = resampled_ds.map(
            encode_and_tokenize_batch,
            batched=True,
            batch_size=32,
            remove_columns=resampled_ds.column_names,
            desc="Encoding with Mimi",
        )

        logger.info("Dataset preprocessing complete. %d samples", len(ds))
        logger.info("Columns: text_tokens (T,), audio_codes (T, %d)", num_quantizers)
        return ds


if __name__ == "__main__":
    LJSpeechTTS.launch(
        Config(
            batch_size=64,
            max_grad_norm=10.0,
            gradient_accumulation_steps=2,
            log_heavy_every_n_seconds=60,
            step_kind="second",
        ),
    )
