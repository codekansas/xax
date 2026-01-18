#!/usr/bin/env -S uv run --no-project --script
"""Fine-tuning Qwen3 for text-to-speech using hierarchical Mimi token prediction.

Architecture:
- Text-to-Semantic (T2S): Qwen3 generates first quantizer (Q0) codes from text
- Semantic-to-Acoustic (S2A): Lightweight MLP generates Q1-Qn codes from Qwen embeddings + prev codes

This allows efficient generation:
1. Qwen autoregressively generates semantic codes (Q0)
2. Acoustic codes (Q1-Qn) are generated layer-by-layer using cached embeddings
"""

import logging
from dataclasses import dataclass
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
            code_embed=eqx.nn.Embedding(xax.MIMI_CODEBOOK_SIZE, mlp_dim, key=k1),
            hidden_proj=eqx.nn.Linear(hidden_dim, mlp_dim, key=k2),
            output_proj=eqx.nn.Linear(mlp_dim, xax.MIMI_CODEBOOK_SIZE, key=k3),
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
    max_audio_seconds: float = xax.field(10.0, help="Maximum audio duration")

    # Data settings
    processed_data_path: str | None = xax.field(None, help="Path to pre-processed data")

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
        path = xax.download_whisper_repo()
        self.whisper_config = xax.load_whisper_config()
        self.whisper_tokenizer = WhisperTokenizerFast.from_pretrained(str(path))

    @override
    def get_model(self, params: xax.InitParams) -> TTSModel:
        # Build LLM with extended vocabulary
        llm = xax.build_pretrained_llm(self.config.llm_repo)

        # Extend vocabulary for audio tokens
        new_vocab_size = self.audio_token_offset + xax.MIMI_CODEBOOK_SIZE
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
        whisper_spec = jax.tree.map(lambda _: False, model.whisper)  # Don't train Whisper model.
        return TTSModel(llm=llm_spec, acoustic_head=acoustic_spec, mimi=mimi_spec, whisper=whisper_spec)

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
                labels_t < self.audio_token_offset + xax.MIMI_CODEBOOK_SIZE
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
            metrics["generated_audio"] = xax.Audio(audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)

            # Transcribe generated audio with Whisper for evaluation
            transcript_tokens, _, _ = xax.transcribe_with_whisper(
                model=model.whisper,
                audio_t=audio_t,
                eos_token_id=self.whisper_config.eos_token_id,
                max_tokens=64,
            )
            metrics["transcript"] = xax.Tokens(transcript_tokens, tokenizer="whisper")

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
        semantic_codes = jnp.clip(semantic_codes, 0, xax.MIMI_CODEBOOK_SIZE - 1)

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

    @override
    def decode_tokens(self, tokens: np.ndarray, token_type: str) -> str:
        token_list: list[int] = tokens.tolist()

        match token_type:
            case "whisper":
                transcript_tokens = [t for t in token_list[4:]]
                return self.whisper_tokenizer.decode(transcript_tokens, skip_special_tokens=True)

            case "llm":
                text_tokens = [t for t in token_list if 0 < t < self.audio_token_offset]
                return self.tokenizer.decode(text_tokens, skip_special_tokens=False)

            case _:
                raise ValueError(f"Invalid token type: {token_type}")

    @override
    def get_dataset(self) -> Dataset:
        if not self.preprocessed_dataset_path.exists():
            raise FileNotFoundError(
                f"Pre-processed dataset not found at {self.preprocessed_dataset_path}. You should first run this "
                "script with `--launcher dataset` to preprocess the dataset."
            )
        return Dataset.load_from_disk(self.preprocessed_dataset_path)

    @override
    def preprocess_dataset(self) -> Dataset:
        columns = ["input_ids", "attention_mask", "labels", "acoustic_codes"]

        # Loads the pre-trained Mimi model.
        mimi = xax.build_pretrained_mimi()

        # Create batched encode function using vmap
        batched_encode = jax.jit(
            jax.vmap(
                lambda audio_ct: mimi.encode(
                    audio_ct,
                    num_quantizers=self.config.num_acoustic_quantizers + 1,
                )
            )
        )

        # Load and process dataset
        logger.info("Processing LJSpeech dataset...")
        raw_ds = load_dataset("keithito/lj_speech", split="train")

        def process_batch(
            examples: dict[str, list],
        ) -> dict[str, list]:
            outputs: dict[str, list] = {column: [] for column in columns}
            bsz = len(examples["id"])

            # First pass: preprocess audio and filter invalid samples
            preprocessed_audio: list[tuple[int, np.ndarray]] = []  # (index, audio)
            valid_indices: list[int] = []

            for idx in range(bsz):
                audio = examples["audio"][idx]["array"]
                sr = examples["audio"][idx]["sampling_rate"]

                # Check duration
                duration = len(audio) / sr
                if duration > self.config.max_audio_seconds:
                    continue

                # Resample to Mimi sample rate
                if sr != xax.MIMI_SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=xax.MIMI_SAMPLE_RATE)

                # Normalize
                audio = audio.astype(np.float32)
                max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
                audio = audio / max_val

                preprocessed_audio.append((idx, audio))
                valid_indices.append(idx)

            if not preprocessed_audio:
                return outputs

            # Find max audio length in this batch for padding
            max_audio_samples = max(len(audio) for _, audio in preprocessed_audio)

            # Pad all audio to the same length and stack into batch
            audio_batch: list[np.ndarray] = []
            audio_lengths: list[int] = []
            for _, audio in preprocessed_audio:
                audio_lengths.append(len(audio))
                padded = np.zeros(max_audio_samples, dtype=np.float32)
                padded[: len(audio)] = audio
                audio_batch.append(padded[None, :])  # Add channel dim: (1, time)

            audio_batch_bct = np.stack(audio_batch, axis=0)  # (batch, channels, time)

            # Batch encode with Mimi on GPU
            audio_batch_jax = jnp.array(audio_batch_bct)
            codes_batch_bqt = batched_encode(audio_batch_jax)  # (batch, num_quantizers, time')
            codes_batch_np = np.array(codes_batch_bqt)

            # Process each valid sample
            for batch_idx, orig_idx in enumerate(valid_indices):
                audio_codes = codes_batch_np[batch_idx]  # (num_quantizers, time')
                num_frames = audio_codes.shape[1]

                # Calculate actual number of valid frames based on original audio length
                # Mimi downsamples by a factor (typically 320 for 24kHz -> 75Hz frame rate)
                orig_audio_len = audio_lengths[batch_idx]

                # Estimate valid frames (Mimi has ~320x downsampling)
                valid_frames = (orig_audio_len + 319) // 320

                if valid_frames > self.config.max_audio_length:
                    continue

                # Trim to valid frames
                audio_codes = audio_codes[:, :valid_frames]
                num_frames = valid_frames

                # Get semantic codes (Q0) and acoustic codes (Q1+)
                semantic_codes = audio_codes[0]  # (T',)
                num_quantizers = self.config.num_acoustic_quantizers + 1
                acoustic_codes = audio_codes[1:] if num_quantizers > 1 else np.zeros((0, num_frames), dtype=np.int32)

                # Tokenize text
                text = examples["normalized_text"][orig_idx]
                text_tokens = self.tokenizer.encode(text, add_special_tokens=True)
                if len(text_tokens) > self.config.max_text_length:
                    continue

                # Create sequence: [TEXT] [AUDIO_START] [SEMANTIC_CODES] [AUDIO_END]
                audio_tokens = [self.audio_token_offset + int(c) for c in semantic_codes]
                full_sequence = text_tokens + [self.audio_start_id] + audio_tokens + [self.audio_end_id]

                # Labels: -100 for text, actual tokens for audio
                labels = [-100] * len(text_tokens) + [-100] + audio_tokens + [self.audio_end_id]

                # Pad to fixed length
                max_length = self.config.max_text_length + self.config.max_audio_length + 3
                pad_length = max_length - len(full_sequence)

                if pad_length > 0:
                    full_sequence = full_sequence + [self.tokenizer.pad_token_id] * pad_length
                    labels = labels + [-100] * pad_length
                    attention_mask = [1] * (max_length - pad_length) + [0] * pad_length
                else:
                    full_sequence = full_sequence[:max_length]
                    labels = labels[:max_length]
                    attention_mask = [1] * max_length

                # Pad acoustic codes
                if acoustic_codes.shape[0] > 0:
                    acoustic_padded = np.zeros((acoustic_codes.shape[0], self.config.max_audio_length), dtype=np.int32)
                    actual_frames = acoustic_codes.shape[1]
                    acoustic_padded[:, :actual_frames] = acoustic_codes
                else:
                    acoustic_padded = np.zeros((0, self.config.max_audio_length), dtype=np.int32)

                outputs["input_ids"].append(full_sequence)
                outputs["attention_mask"].append(attention_mask)
                outputs["labels"].append(labels)
                outputs["acoustic_codes"].append(acoustic_padded)

            return outputs

        ds = raw_ds.map(
            process_batch,
            batched=True,
            batch_size=32,  # Process 32 samples at a time on GPU
            remove_columns=raw_ds.column_names,
            desc="Processing LJSpeech",
        )

        ds.set_format(type="numpy", columns=columns)

        return ds


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
    LJSpeechTTS.launch(
        Config(
            batch_size=4,
            max_grad_norm=1.0,
            gradient_accumulation_steps=2,
            log_heavy_every_n_seconds=60,
            max_steps=60 * 10,  # 10 minutes for testing
            step_kind="second",
        ),
    )
