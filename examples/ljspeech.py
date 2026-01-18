#!/usr/bin/env -S uv run --no-project --script
"""Fine-tuning Qwen3 for text-to-speech using hierarchical Mimi token prediction.

Architecture:
- Text-to-Semantic (T2S): Qwen3 generates first quantizer (Q0) codes from text
- Semantic-to-Acoustic (S2A): Lightweight MLP generates Q1-Qn codes from Qwen embeddings + prev codes

This allows efficient generation:
1. Qwen autoregressively generates semantic codes (Q0)
2. Acoustic codes (Q1-Qn) are generated layer-by-layer using cached embeddings
"""

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, override

import equinox as eqx
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.whisper.tokenization_whisper_fast import WhisperTokenizerFast

import xax

logger = logging.getLogger(__name__)

# Mimi constants
MIMI_SAMPLE_RATE = 24000
MIMI_CODEBOOK_SIZE = 2048
LJSPEECH_SAMPLE_RATE = 22050
NUM_ACOUSTIC_QUANTIZERS = 7  # Use Q1-Q7 for acoustic (8 total quantizers)

# Special tokens
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

# LoRA targets (same as shakespeare.py)
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "gate", "up")


class Batch(TypedDict):
    input_ids: Array
    attention_mask: Array
    labels: Array
    acoustic_codes: Array  # (batch, num_acoustic_quantizers, seq_len)


class AcousticHead(eqx.Module):
    """Lightweight MLP to predict acoustic codes from Qwen embeddings + previous layer code.

    For each acoustic layer i (1 to num_acoustic_quantizers):
    - Input: [hidden_state, embedding(code_{i-1})]
    - Output: logits for code_i
    """

    code_embed: eqx.nn.Embedding  # Embed previous layer's code
    hidden_proj: eqx.nn.Linear  # Project hidden state
    output_proj: eqx.nn.Linear  # Output logits
    num_layers: int = eqx.field(static=True)

    @staticmethod
    def build(hidden_dim: int, num_layers: int, key: PRNGKeyArray) -> "AcousticHead":
        k1, k2, k3 = jax.random.split(key, 3)
        mlp_dim = 512  # Lightweight MLP
        return AcousticHead(
            code_embed=eqx.nn.Embedding(MIMI_CODEBOOK_SIZE, mlp_dim, key=k1),
            hidden_proj=eqx.nn.Linear(hidden_dim, mlp_dim, key=k2),
            output_proj=eqx.nn.Linear(mlp_dim, MIMI_CODEBOOK_SIZE, key=k3),
            num_layers=num_layers,
        )

    def __call__(self, hidden_d: Array, prev_code: Array) -> Array:
        """Predict next acoustic layer's code.

        Args:
            hidden_d: Hidden state from Qwen, shape (hidden_dim,)
            prev_code: Previous layer's code, scalar

        Returns:
            Logits for next code, shape (codebook_size,)
        """
        code_emb_d = self.code_embed(prev_code)
        hidden_proj_d = self.hidden_proj(hidden_d)
        combined_d = jax.nn.gelu(code_emb_d + hidden_proj_d)
        return self.output_proj(combined_d)


class TTSModel(eqx.Module):
    """Combined model for text-to-speech with hierarchical codec prediction."""

    llm: xax.LLM  # Qwen for semantic codes
    acoustic_head: AcousticHead  # MLP for acoustic codes
    mimi: xax.MimiModel
    whisper: xax.WhisperModel

    @staticmethod
    def build(llm: xax.LLM, num_acoustic_layers: int, key: PRNGKeyArray) -> "TTSModel":
        hidden_dim = llm.config.embed_dim
        acoustic_head = AcousticHead.build(hidden_dim, num_acoustic_layers, key)
        return TTSModel(
            llm=llm,
            acoustic_head=acoustic_head,
            mimi=xax.build_pretrained_mimi(),
            whisper=xax.build_pretrained_whisper(),
        )


@dataclass
class Config(xax.SupervisedConfig):
    # Model settings
    llm_repo: xax.LLMRepo = xax.field(xax.LLMRepo.QWEN3_600M, help="Pretrained model")
    num_acoustic_quantizers: int = xax.field(NUM_ACOUSTIC_QUANTIZERS, help="Number of acoustic quantizers")

    # LoRA settings (same as shakespeare.py)
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(16.0, help="LoRA alpha parameter")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")
    lora_targets: tuple[str, ...] | None = xax.field(DEFAULT_LORA_TARGETS, help="Layer suffixes for LoRA")

    # Training settings
    learning_rate: float = xax.field(5e-4, help="Peak learning rate")
    min_learning_rate: float = xax.field(1e-5, help="Minimum learning rate")
    warmup_steps: int = xax.field(50, help="Number of warmup steps")
    max_text_length: int = xax.field(128, help="Maximum text token length")
    max_audio_length: int = xax.field(256, help="Maximum audio token length")

    # Audio settings
    max_audio_seconds: float = xax.field(5.0, help="Maximum audio duration")

    # Data settings
    processed_data_path: str | None = xax.field(None, help="Path to pre-processed data")
    max_examples: int | None = xax.field(None, help="Max examples to use (None for all)")

    # Eval settings
    eval_prompt: str = xax.field("Hello world, this is a test of the text to speech system.", help="Text for eval")


class LJSpeechTTS(xax.SupervisedTask[Config]):
    """Fine-tune Qwen3 for text-to-speech with hierarchical codec prediction."""

    tokenizer: Qwen2TokenizerFast
    whisper_tokenizer: WhisperTokenizerFast
    mimi: xax.MimiModel
    audio_start_id: int
    audio_end_id: int
    text_vocab_size: int
    audio_token_offset: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Load tokenizer and add special tokens
        self.tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(config.llm_repo.value)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_START_TOKEN, AUDIO_END_TOKEN]})

        self.audio_start_id = self.tokenizer.convert_tokens_to_ids(AUDIO_START_TOKEN)
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids(AUDIO_END_TOKEN)
        self.text_vocab_size = len(self.tokenizer)
        self.audio_token_offset = self.text_vocab_size

        # Pre-tokenize evaluation prompt for audio generation
        prompt_tokens = self.tokenizer.encode(config.eval_prompt, add_special_tokens=True)
        prompt_tokens.append(self.audio_start_id)  # Add audio start token
        self._eval_prompt_tokens = jnp.array(prompt_tokens, dtype=jnp.int32)

        # Load Whisper model for ASR evaluation (frozen)
        logger.info("Loading Whisper model for ASR evaluation")
        self.whisper_model = xax.build_pretrained_whisper()
        self.whisper_tokenizer = xax.load_whisper_tokenizer()

    @override
    def get_model(self, params: xax.InitParams) -> TTSModel:
        # Build LLM with extended vocabulary
        llm = xax.build_pretrained_llm(self.config.llm_repo)

        # Extend vocabulary for audio tokens
        new_vocab_size = self.audio_token_offset + MIMI_CODEBOOK_SIZE
        llm = _resize_embeddings(llm, new_vocab_size, params.key)

        # Apply LoRA (same config as shakespeare.py)
        llm = xax.loraize_by_path(
            llm,
            rank=self.config.lora_rank,
            include_suffixes=list(self.config.lora_targets) if self.config.lora_targets else None,
            alpha=self.config.lora_alpha,
            dropout_rate=self.config.lora_dropout,
            key=params.key,
        )

        # Build full model with acoustic head
        k_acoustic = jax.random.fold_in(params.key, 1)
        return TTSModel.build(llm, self.config.num_acoustic_quantizers, k_acoustic)

    @override
    def get_trainable_filter_spec(self, model: TTSModel) -> TTSModel:
        llm_spec = xax.lora_filter_spec(model.llm)  # LoRA on LLM.
        acoustic_spec = jax.tree.map(lambda _: True, model.acoustic_head)  # Train acoustic head.
        mimi_spec = jax.tree.map(lambda _: False, model.mimi)  # Don't train Mimi model.
        return TTSModel(llm=llm_spec, acoustic_head=acoustic_spec, mimi=mimi_spec)

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
        model: TTSModel,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        # Semantic loss (Q0 prediction with Qwen)
        input_ids_bt = batch["input_ids"][:, :-1]
        targets_bt = batch["labels"][:, 1:]
        mask_bt = batch["attention_mask"][:, 1:] == 1
        loss_mask_bt = (targets_bt != -100) & mask_bt

        # Forward pass - get hidden states
        hidden_btd = jax.vmap(model.llm.forward_hidden)(input_ids_bt)

        # Semantic loss using chunked cross entropy
        semantic_loss = jax.vmap(xax.chunked_cross_entropy_loss, in_axes=(0, 0, None, 0, None))(
            hidden_btd,
            targets_bt,
            model.llm.lm_head.weight,
            loss_mask_bt,
            256,
        ).mean()

        # Acoustic loss (Q1-Qn prediction with acoustic head)
        acoustic_codes_bqt = batch["acoustic_codes"]  # (B, Q, T)
        bsz, num_q, tsz = acoustic_codes_bqt.shape

        # Find audio region in sequence (where labels are audio tokens)
        # We need to align hidden states with acoustic codes

        def compute_acoustic_loss_single(hidden_td: Array, labels_t: Array, acoustic_qt: Array) -> Array:
            # Find where audio starts in the sequence
            audio_mask = (labels_t >= self.audio_token_offset) & (
                labels_t < self.audio_token_offset + MIMI_CODEBOOK_SIZE
            )
            audio_positions = jnp.where(audio_mask, size=tsz, fill_value=0)[0]
            valid_mask = audio_mask[audio_positions]

            # For each acoustic layer, compute loss
            def layer_loss(prev_codes_t: Array, target_codes_t: Array) -> Array:
                # Get hidden states at audio positions
                def position_loss(pos: Array, valid: Array) -> Array:
                    # Use safe indexing (pos is always valid index due to fill_value=0)
                    hidden_d = hidden_td[pos]
                    prev_code = prev_codes_t[pos]
                    target = target_codes_t[pos]

                    logits = model.acoustic_head(hidden_d, prev_code)
                    loss = -jax.nn.log_softmax(logits)[target]
                    return jnp.where(valid, loss, 0.0)

                losses = jax.vmap(position_loss)(audio_positions, valid_mask)
                num_valid = jnp.sum(valid_mask)
                return jnp.sum(losses) / jnp.maximum(num_valid, 1)

            # Q0 -> Q1, Q1 -> Q2, etc.
            total_loss = jnp.array(0.0)
            for q_idx in range(num_q):
                # Q0 predicts Q1, Q1 predicts Q2, etc.
                prev = acoustic_qt[max(0, q_idx - 1)]  # Use Q0 for first layer
                target = acoustic_qt[q_idx]
                total_loss = total_loss + layer_loss(prev, target)

            return total_loss / jnp.maximum(num_q, 1)

        # Only compute acoustic loss if we have acoustic codes
        has_acoustic = num_q > 0 and tsz > 0
        if has_acoustic:
            acoustic_loss = jax.vmap(compute_acoustic_loss_single)(hidden_btd, targets_bt, acoustic_codes_bqt).mean()
        else:
            acoustic_loss = jnp.array(0.0)

        # Combined loss
        loss = semantic_loss + 0.5 * acoustic_loss

        metrics: dict[str, xax.Metric] = {
            "semantic_loss": xax.Scalar(semantic_loss),
            "acoustic_loss": xax.Scalar(acoustic_loss),
            "perplexity": xax.Scalar(jnp.exp(semantic_loss)),
        }

        if heavy:
            gen_key, _ = jax.random.split(key)
            audio_t = self._generate_audio_jit(model, batch, gen_key)
            metrics["generated_audio"] = xax.Audio(audio_t, sample_rate=MIMI_SAMPLE_RATE)

            # Transcribe generated audio with Whisper for evaluation
            transcript_tokens = self._transcribe_audio_jit(audio_t)
            metrics["transcript"] = xax.Tokens(transcript_tokens)

        return loss, metrics

    def _generate_audio_jit(self, model: TTSModel, batch: Batch, key: PRNGKeyArray) -> Array:
        """Generate audio in a JIT-compatible way.

        Uses a fixed pre-tokenized prompt and fixed shapes throughout.
        """
        max_len = self.config.max_audio_length

        # Use pre-tokenized evaluation prompt
        prompt_tokens = self._eval_prompt_tokens

        # Generate semantic codes with LLM
        gen_key, acoustic_key = jax.random.split(key)
        gen_result = xax.llm_generate_jit(
            model.llm,
            prompt_tokens,
            eos_id=self.audio_end_id,
            max_new_tokens=max_len,
            temperature=0.8,
            top_p=0.95,
            key=gen_key,
        )
        generated = gen_result[0]

        # Find audio region in generated sequence (start after audio_start token)
        gen_audio_start = jnp.argmax(generated == self.audio_start_id) + 1

        # Extract semantic codes (Q0) - they are offset by audio_token_offset
        audio_region = jax.lax.dynamic_slice(generated, (gen_audio_start,), (max_len,))
        semantic_codes = audio_region - self.audio_token_offset
        # Clamp to valid range
        semantic_codes = jnp.clip(semantic_codes, 0, MIMI_CODEBOOK_SIZE - 1)

        # Get hidden states for acoustic generation
        hidden_td = model.llm.forward_hidden(generated)
        hidden_dim = hidden_td.shape[-1]

        # Extract hidden states for audio region
        audio_hidden = jax.lax.dynamic_slice(hidden_td, (gen_audio_start, 0), (max_len, hidden_dim))

        # Generate acoustic codes layer by layer
        all_codes = [semantic_codes]
        prev_codes = semantic_codes

        for q_idx in range(self.config.num_acoustic_quantizers):
            layer_key = jax.random.fold_in(acoustic_key, q_idx)

            def sample_code(hidden_d: Array, prev_code: Array, sample_key: PRNGKeyArray) -> Array:
                logits = model.acoustic_head(hidden_d, prev_code)
                return jax.random.categorical(sample_key, logits / 0.8)

            keys = jax.random.split(layer_key, max_len)
            next_codes = jax.vmap(sample_code)(audio_hidden, prev_codes, keys)
            all_codes.append(next_codes)
            prev_codes = next_codes

        # Stack codes: (num_quantizers, max_len)
        codes_qt = jnp.stack(all_codes, axis=0)

        # Decode with Mimi - decode full sequence
        # The audio will be longer than needed but that's OK for evaluation
        audio_ct = model.mimi.decode(codes_qt)

        return audio_ct[0]  # Return first channel

    def _transcribe_audio_jit(self, audio_t: Array) -> Array:
        """Transcribe audio using frozen Whisper model via pure_callback.

        Uses pure_callback to avoid capturing the large Whisper model in JIT trace.
        Resamples from Mimi sample rate (24kHz) to Whisper sample rate (16kHz).
        """
        max_tokens = 68  # 4 prompt + 64 generated

        def transcribe_fn(audio: np.ndarray) -> np.ndarray:
            """Transcribe audio outside of JIT."""
            # Resample from Mimi 24kHz to Whisper 16kHz
            target_16k_len = xax.WHISPER_SAMPLE_RATE * 30
            indices = np.linspace(0, len(audio) - 1, target_16k_len)
            audio_16k = np.interp(indices, np.arange(len(audio)), audio)
            audio_16k = jnp.array(audio_16k, dtype=jnp.float32)

            # Compute mel spectrogram
            mel_mt = xax.log_mel_spectrogram(audio_16k)

            # Encode audio
            encoder_out = self.whisper_model.encode(mel_mt)

            # Greedy decoding
            tokens = jnp.array([50258, 50259, 50360, 50364], dtype=jnp.int32)
            eos_token = 50257

            for _ in range(64):
                logits_tv = self.whisper_model.decode(tokens, encoder_out)
                next_token = int(jnp.argmax(logits_tv[-1]))
                if next_token == eos_token:
                    break
                tokens = jnp.concatenate([tokens, jnp.array([next_token], dtype=jnp.int32)])

            # Pad to fixed length
            result = np.full(max_tokens, eos_token, dtype=np.int32)
            result[: len(tokens)] = np.array(tokens)
            return result

        result = jax.pure_callback(
            transcribe_fn,
            jax.ShapeDtypeStruct((max_tokens,), jnp.int32),
            audio_t,
        )
        return result

    @override
    def decode_tokens(self, tokens: Array | np.ndarray) -> str:
        """Decode tokens to text.

        Handles both TTS tokens (text + audio) and Whisper transcript tokens.
        """
        token_list: list[int] = tokens.tolist()

        # Check if these are Whisper tokens (start with special tokens 50258, 50259, etc.)
        if len(token_list) >= 4 and token_list[0] == 50258:
            # Whisper tokens - skip the first 4 special tokens and decode with Whisper tokenizer
            transcript_tokens = [t for t in token_list[4:] if t != 50257]  # Skip EOS
            return self.whisper_tokenizer["decode"](transcript_tokens)

        # TTS tokens - filter out audio tokens and decode with Qwen tokenizer
        text_tokens = [t for t in token_list if 0 < t < self.audio_token_offset]
        return self.tokenizer.decode(text_tokens, skip_special_tokens=False)

    @override
    def get_dataset(self) -> Dataset:
        mimi = xax.build_pretrained_mimi()

        # Check for pre-processed data
        cache_path = Path("/tmp/ljspeech_processed")
        if cache_path.exists():
            logger.info("Loading pre-processed dataset from %s", cache_path)
            ds = Dataset.load_from_disk(str(cache_path))

            # Remove Mimi model from memory.
            del mimi
            gc.collect()

            return ds

        # Load and process dataset (should be done before JAX init)
        logger.info("Processing LJSpeech dataset...")
        raw_ds = load_dataset("keithito/lj_speech", split="train")

        # Limit examples for faster testing
        max_examples = self.config.max_examples or len(raw_ds)
        logger.info("Processing up to %d examples", max_examples)

        # Process without multiprocessing to avoid JAX fork issues
        processed_examples = []
        for idx, example in enumerate(raw_ds):
            if idx >= max_examples:
                break
            if idx % 10 == 0:
                logger.info("Processing example %d/%d", idx, max_examples)

            result = _process_example(
                example,
                tokenizer=self.tokenizer,
                mimi=mimi,
                max_text_length=self.config.max_text_length,
                max_audio_length=self.config.max_audio_length,
                max_audio_seconds=self.config.max_audio_seconds,
                audio_token_offset=self.audio_token_offset,
                audio_start_id=self.audio_start_id,
                audio_end_id=self.audio_end_id,
                num_quantizers=self.config.num_acoustic_quantizers + 1,  # Q0 + acoustic
            )
            if result["input_ids"] is not None:
                processed_examples.append(result)

        logger.info("Processed %d examples", len(processed_examples))

        # Create dataset from processed examples
        ds = Dataset.from_dict(
            {
                "input_ids": [e["input_ids"] for e in processed_examples],
                "attention_mask": [e["attention_mask"] for e in processed_examples],
                "labels": [e["labels"] for e in processed_examples],
                "acoustic_codes": [e["acoustic_codes"] for e in processed_examples],
            }
        )

        # Save to cache for faster subsequent runs
        ds.save_to_disk(str(cache_path))
        logger.info("Saved processed dataset to %s", cache_path)

        # Remove Mimi model from memory.
        del mimi
        gc.collect()

        ds.set_format(type="numpy", columns=["input_ids", "attention_mask", "labels", "acoustic_codes"])
        return ds


def _process_example(
    example: dict,
    tokenizer: Qwen2TokenizerFast,
    mimi: xax.MimiModel,
    max_text_length: int,
    max_audio_length: int,
    max_audio_seconds: float,
    audio_token_offset: int,
    audio_start_id: int,
    audio_end_id: int,
    num_quantizers: int,
) -> dict:
    """Process a single example: tokenize text and encode audio."""
    # Get audio
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    # Check duration
    duration = len(audio) / sr
    if duration > max_audio_seconds:
        return {"input_ids": None, "attention_mask": None, "labels": None, "acoustic_codes": None}

    # Resample to Mimi sample rate
    if sr != MIMI_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=MIMI_SAMPLE_RATE)

    # Normalize
    audio = audio.astype(np.float32)
    max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
    audio = audio / max_val

    # Encode with Mimi
    audio_ct = jnp.array(audio)[None, :]  # (1, T)
    audio_codes = mimi.encode(audio_ct, num_quantizers=num_quantizers)  # (Q, T')
    audio_codes = np.array(audio_codes)

    num_frames = audio_codes.shape[1]
    if num_frames > max_audio_length:
        return {"input_ids": None, "attention_mask": None, "labels": None, "acoustic_codes": None}

    # Get semantic codes (Q0) and acoustic codes (Q1+)
    semantic_codes = audio_codes[0]  # (T',)
    acoustic_codes = audio_codes[1:] if num_quantizers > 1 else np.zeros((0, num_frames), dtype=np.int32)

    # Tokenize text
    text = example["normalized_text"]
    text_tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(text_tokens) > max_text_length:
        text_tokens = text_tokens[:max_text_length]

    # Create sequence: [TEXT] [AUDIO_START] [SEMANTIC_CODES] [AUDIO_END]
    audio_tokens = [audio_token_offset + c for c in semantic_codes]
    full_sequence = text_tokens + [audio_start_id] + audio_tokens + [audio_end_id]

    # Labels: -100 for text, actual tokens for audio
    labels = [-100] * len(text_tokens) + [-100] + audio_tokens + [audio_end_id]

    # Pad to fixed length
    max_length = max_text_length + max_audio_length + 3
    pad_length = max_length - len(full_sequence)

    if pad_length > 0:
        full_sequence = full_sequence + [tokenizer.pad_token_id] * pad_length
        labels = labels + [-100] * pad_length
        attention_mask = [1] * (max_length - pad_length) + [0] * pad_length
    else:
        full_sequence = full_sequence[:max_length]
        labels = labels[:max_length]
        attention_mask = [1] * max_length

    # Pad acoustic codes
    if acoustic_codes.shape[0] > 0:
        acoustic_padded = np.zeros((acoustic_codes.shape[0], max_audio_length), dtype=np.int32)
        acoustic_padded[:, :num_frames] = acoustic_codes[:, :max_audio_length]
    else:
        acoustic_padded = np.zeros((0, max_audio_length), dtype=np.int32)

    return {
        "input_ids": full_sequence,
        "attention_mask": attention_mask,
        "labels": labels,
        "acoustic_codes": acoustic_padded,
    }


def _resize_embeddings(model: xax.LLM, new_vocab_size: int, key: PRNGKeyArray) -> xax.LLM:
    """Resize embedding and lm_head layers for new vocabulary size."""
    old_embed = model.embed
    old_vocab_size = old_embed.num_embeddings
    embed_dim = old_embed.embedding_size

    if new_vocab_size <= old_vocab_size:
        return model

    k1, k2 = jax.random.split(key)

    # New embedding
    new_embed = eqx.nn.Embedding(new_vocab_size, embed_dim, key=k1)
    new_weights = new_embed.weight.at[:old_vocab_size].set(old_embed.weight)
    new_token_weights = jax.random.normal(k2, (new_vocab_size - old_vocab_size, embed_dim)) * 0.02
    new_weights = new_weights.at[old_vocab_size:].set(new_token_weights)
    new_embed = eqx.tree_at(lambda e: e.weight, new_embed, new_weights)

    # New lm_head
    new_lm_head = eqx.nn.Linear(embed_dim, new_vocab_size, use_bias=False, key=k1)
    new_lm_weights = new_lm_head.weight.at[:old_vocab_size].set(model.lm_head.weight)
    new_lm_weights = new_lm_weights.at[old_vocab_size:].set(new_token_weights)
    new_lm_head = eqx.tree_at(lambda head: head.weight, new_lm_head, new_lm_weights)

    model = eqx.tree_at(lambda m: m.embed, model, new_embed)
    model = eqx.tree_at(lambda m: m.lm_head, model, new_lm_head)

    return model


if __name__ == "__main__":
    # Disable JAX preallocation to avoid memory issues during data processing
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    LJSpeechTTS.launch(
        Config(
            batch_size=4,
            max_grad_norm=1.0,
            gradient_accumulation_steps=2,
            log_heavy_every_n_seconds=60,
            # max_steps=60 * 10,  # 10 minutes for testing
            step_kind="second",
            # max_examples=50,  # Use 50 examples for quick testing
        ),
    )
