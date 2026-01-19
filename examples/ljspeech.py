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
    input_ids: Array  # Full sequence: [text_tokens, audio_start, flattened_audio_tokens, audio_end]
    attention_mask: Array
    labels: Array  # -100 for text tokens, actual tokens for audio portion
    audio_codes: Array  # (batch, num_quantizers, audio_frames) - original unflattened for decoding
    audio_codes_mask: Array  # (batch, audio_frames) - valid audio frame positions
    text_tokens: Array  # (batch, max_text_length) - just the text tokens
    text_mask: Array  # (batch, max_text_length) - valid text positions


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
    """Combined model for text-to-speech with hierarchical codec prediction.

    Audio tokens use separate embeddings so we only train those, not the pretrained text embeddings.
    Each quantizer layer has its own codebook, so we have num_quantizers * codebook_size audio tokens.
    Weight tying is used: audio_embed.weight is used for both input embeddings and output predictions.
    """

    llm: xax.LLM  # Qwen for text (frozen embeddings)
    audio_embed: eqx.nn.Embedding  # Trainable audio token embeddings (num_quantizers * codebook_size)
    quantizer_embed: eqx.nn.Embedding  # Optional: embeddings for quantizer index (0-7) as prior
    acoustic_head: AcousticHead  # MLP for acoustic codes (kept for compatibility)
    mimi: xax.MimiModel
    whisper: xax.WhisperModel

    @staticmethod
    def build(
        llm: xax.LLM,
        num_acoustic_layers: int,
        num_quantizers: int,
        codebook_size: int,
        key: PRNGKeyArray,
        use_quantizer_embed: bool = True,
        init_from_mimi: bool = True,
    ) -> "TTSModel":
        hidden_dim = llm.config.embed_dim
        k1, k2, k3 = jax.random.split(key, 3)
        acoustic_head = AcousticHead.build(hidden_dim, num_acoustic_layers, k1)

        # Load Mimi for audio encoding/decoding
        mimi = xax.build_pretrained_mimi()

        # Separate embeddings for audio tokens: each quantizer has its own codebook
        # Total tokens = num_quantizers * codebook_size
        total_audio_tokens = num_quantizers * codebook_size

        if init_from_mimi:
            # Initialize from Mimi codebook embeddings (projected to LLM dim)
            # This gives audio tokens meaningful acoustic starting points
            mimi_dim = mimi.quantizer.semantic_rvq.layers[0].codebook_dim  # 256
            # Collect all quantizer embeddings
            all_codebook_embeds = []
            # Semantic quantizer (Q0)
            all_codebook_embeds.append(mimi.quantizer.semantic_rvq.layers[0].embeddings_kd)
            # Acoustic quantizers (Q1-Q7)
            for layer in mimi.quantizer.acoustic_rvq.layers[:num_quantizers - 1]:
                all_codebook_embeds.append(layer.embeddings_kd)
            # Stack: (num_quantizers, codebook_size, mimi_dim)
            stacked_embeds = jnp.stack(all_codebook_embeds, axis=0)
            # Flatten to (total_audio_tokens, mimi_dim)
            flat_embeds = stacked_embeds.reshape(total_audio_tokens, mimi_dim)
            # Project to LLM hidden dim with random projection + scaling
            proj_matrix = jax.random.normal(k2, (mimi_dim, hidden_dim)) * 0.02
            audio_weight = flat_embeds @ proj_matrix
            # Create embedding with initialized weights
            audio_embed = eqx.nn.Embedding(total_audio_tokens, hidden_dim, key=k2)
            audio_embed = eqx.tree_at(lambda e: e.weight, audio_embed, audio_weight)
            logger.info("Initialized audio embeddings from Mimi codebook (projected %d -> %d)", mimi_dim, hidden_dim)
        else:
            audio_embed = eqx.nn.Embedding(total_audio_tokens, hidden_dim, key=k2)

        # Quantizer embeddings: small learned prior for each quantizer (0-7)
        # This helps the model learn the structure faster
        quantizer_embed = eqx.nn.Embedding(num_quantizers, hidden_dim, key=k3) if use_quantizer_embed else None
        return TTSModel(
            llm=llm,
            audio_embed=audio_embed,
            quantizer_embed=quantizer_embed,
            acoustic_head=acoustic_head,
            mimi=mimi,
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
    max_audio_frames: int = xax.field(64, help="Maximum audio frames (each frame has 8 quantizer tokens)")
    semantic_weight: float = xax.field(1.0, help="Loss weight for semantic tokens (Q0) vs acoustic (Q1-Q7)")
    use_quantizer_embed: bool = xax.field(True, help="Add quantizer index embeddings as prior")

    # Audio settings
    max_audio_seconds: float = xax.field(5.0, help="Maximum audio duration")
    init_audio_from_mimi: bool = xax.field(True, help="Initialize audio embeddings from Mimi codebook")
    audio_prompt_frames: int = xax.field(0, help="Number of ground truth audio frames to use as prompt (0 = none)")
    scheduled_sampling: float = xax.field(0.0, help="Probability of using model predictions instead of GT (0-1)")

    # Generation settings
    gen_temperature: float = xax.field(0.8, help="Temperature for generation sampling")
    gen_top_k: int = xax.field(50, help="Top-k for generation sampling (0 = no top-k)")

    # Data settings
    processed_data_path: str | None = xax.field(None, help="Path to pre-processed data")

    # Eval settings - use actual training text for debugging
    eval_prompt: str = xax.field(
        "With this change the art of printing touched bottom,",
        help="Text for eval (must be in training set)",
    )


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
        # Build LLM - do NOT extend embeddings, we use separate audio_embed
        llm = xax.build_pretrained_llm(self.config.llm_repo)

        # Apply LoRA (same config as shakespeare.py)
        llm = xax.loraize_by_path(
            llm,
            rank=self.config.lora_rank,
            include_suffixes=list(self.config.lora_targets) if self.config.lora_targets else None,
            alpha=self.config.lora_alpha,
            dropout_rate=self.config.lora_dropout,
            key=params.key,
        )

        # Build full model with separate audio embeddings
        # Each quantizer layer has its own codebook
        num_quantizers = self.config.num_acoustic_quantizers + 1  # 8 total
        k_model = jax.random.fold_in(params.key, 1)
        return TTSModel.build(
            llm,
            self.config.num_acoustic_quantizers,
            num_quantizers,
            xax.MIMI_CODEBOOK_SIZE,
            k_model,
            use_quantizer_embed=self.config.use_quantizer_embed,
            init_from_mimi=self.config.init_audio_from_mimi,
        )

    @override
    def get_trainable_filter_spec(self, model: TTSModel) -> TTSModel:
        # Get LoRA filter spec - only LoRA layers in LLM are trainable
        # The LLM embeddings and lm_head stay frozen (pretrained text tokens)
        llm_spec = xax.lora_filter_spec(model.llm)

        # Train audio token embeddings (weight-tied, so no separate lm_head)
        audio_embed_spec = jax.tree.map(lambda _: True, model.audio_embed)
        # Train quantizer embeddings if present
        quantizer_embed_spec = (
            jax.tree.map(lambda _: True, model.quantizer_embed) if model.quantizer_embed is not None else None
        )
        acoustic_spec = jax.tree.map(lambda _: True, model.acoustic_head)
        mimi_spec = jax.tree.map(lambda _: False, model.mimi)
        whisper_spec = jax.tree.map(lambda _: False, model.whisper)
        return TTSModel(
            llm=llm_spec,
            audio_embed=audio_embed_spec,
            quantizer_embed=quantizer_embed_spec,
            acoustic_head=acoustic_spec,
            mimi=mimi_spec,
            whisper=whisper_spec,
        )

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

    def _get_embeddings(self, model: TTSModel, token_ids: Array) -> Array:
        """Get embeddings for mixed text/audio token IDs.

        Uses LLM embeddings for text tokens, separate audio_embed for audio tokens.
        Each quantizer has its own codebook: total audio tokens = num_quantizers * codebook_size.
        Optionally adds quantizer index embeddings as a prior to help learning.
        """
        num_quantizers = self.config.num_acoustic_quantizers + 1  # 8
        total_audio_tokens = num_quantizers * xax.MIMI_CODEBOOK_SIZE

        # Mask for audio tokens
        is_audio = token_ids >= self.audio_token_offset

        # Get text embeddings (for all tokens, but will be masked out for audio)
        text_embed = jax.vmap(model.llm.embed)(jnp.where(is_audio, 0, token_ids))

        # Get audio embeddings (map audio token IDs to 0-based index into audio_embed)
        audio_ids = jnp.where(is_audio, token_ids - self.audio_token_offset, 0)
        audio_ids = jnp.clip(audio_ids, 0, total_audio_tokens - 1)
        audio_embed = jax.vmap(model.audio_embed)(audio_ids)

        # Optionally add quantizer embeddings as prior
        if model.quantizer_embed is not None:
            # Compute quantizer index (0-7) for each audio token
            quantizer_idx = audio_ids // xax.MIMI_CODEBOOK_SIZE
            quantizer_idx = jnp.clip(quantizer_idx, 0, num_quantizers - 1)
            q_embed = jax.vmap(model.quantizer_embed)(quantizer_idx)
            # Add quantizer embedding to audio embedding (only for audio positions)
            audio_embed = audio_embed + jnp.where(is_audio[..., None], q_embed, 0)

        # Combine: use text embed where is_text, audio embed where is_audio
        return jnp.where(is_audio[..., None], audio_embed, text_embed)

    def _forward_hidden_from_embed(self, model: TTSModel, embed_td: Array) -> Array:
        """Run transformer blocks on pre-computed embeddings."""
        x_tn = embed_td
        for block in model.llm.blocks:
            x_tn, _ = block.forward(x_tn, cache=None)
        return model.llm.norm(x_tn)

    @override
    def compute_loss(
        self,
        model: TTSModel,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        # Simple next-token prediction on flattened sequence
        # The sequence is: [TEXT] [AUDIO_START] [Q0_t0, Q1_t0, ..., Q7_t0, Q0_t1, ...] [AUDIO_END]
        input_ids_bt = batch["input_ids"][:, :-1]
        targets_bt = batch["labels"][:, 1:]
        mask_bt = batch["attention_mask"][:, 1:] == 1
        loss_mask_bt = (targets_bt != -100) & mask_bt

        # Scheduled sampling: with probability p, replace audio input tokens with random tokens
        # This helps the model be robust to imperfect inputs (addresses exposure bias)
        if self.config.scheduled_sampling > 0:
            num_quantizers = self.config.num_acoustic_quantizers + 1
            total_audio_tokens = num_quantizers * xax.MIMI_CODEBOOK_SIZE
            is_audio_input = input_ids_bt >= self.audio_token_offset
            # Generate random replacement tokens
            k1, key = jax.random.split(key)
            random_tokens = jax.random.randint(
                k1, input_ids_bt.shape, 0, total_audio_tokens
            ) + self.audio_token_offset
            # Generate mask for which tokens to replace
            k2, key = jax.random.split(key)
            replace_mask = jax.random.uniform(k2, input_ids_bt.shape) < self.config.scheduled_sampling
            # Only replace audio tokens, not text
            replace_mask = replace_mask & is_audio_input
            # Apply replacement
            input_ids_bt = jnp.where(replace_mask, random_tokens, input_ids_bt)

        # Get embeddings with separate text/audio lookup
        embed_btd = jax.vmap(self._get_embeddings, in_axes=(None, 0))(model, input_ids_bt)

        # Forward pass through LLM layers (skip embedding layer)
        hidden_btd = jax.vmap(self._forward_hidden_from_embed, in_axes=(None, 0))(model, embed_btd)

        # Create prediction weight matrix with weight tying:
        # - LLM lm_head for text tokens
        # - audio_embed for audio tokens (weight tying)
        text_weights = model.llm.lm_head.weight  # (text_vocab, hidden)
        audio_weights = model.audio_embed.weight  # (audio_vocab, hidden) - weight tying
        tied_weights = jnp.concatenate([text_weights, audio_weights], axis=0)

        # Compute per-token logits and loss
        logits_btv = jnp.einsum("btd,vd->btv", hidden_btd, tied_weights)
        per_token_loss_bt = optax.softmax_cross_entropy_with_integer_labels(logits_btv, targets_bt)

        # Apply semantic weighting: Q0 tokens get higher weight, Q1-Q7 get weight 1.0
        # Audio tokens are in range [audio_offset, audio_offset + 16384)
        # Q0 tokens have (token - audio_offset) // CODEBOOK_SIZE == 0
        num_quantizers = self.config.num_acoustic_quantizers + 1  # 8
        is_audio = targets_bt >= self.audio_token_offset
        audio_idx = jnp.where(is_audio, targets_bt - self.audio_token_offset, 0)
        quantizer_idx = audio_idx // xax.MIMI_CODEBOOK_SIZE  # Which quantizer (0-7)
        is_semantic = is_audio & (quantizer_idx == 0)  # Q0 tokens

        # Weight: semantic_weight for Q0, 1.0 for everything else
        token_weight_bt = jnp.where(is_semantic, self.config.semantic_weight, 1.0)
        weighted_mask_bt = loss_mask_bt.astype(jnp.float32) * token_weight_bt

        # Weighted average loss
        loss = jnp.sum(per_token_loss_bt * weighted_mask_bt) / jnp.maximum(jnp.sum(weighted_mask_bt), 1)

        # Compute token accuracy (logits_btv already computed above)
        predictions_bt = jnp.argmax(logits_btv, axis=-1)
        correct_bt = (predictions_bt == targets_bt) & loss_mask_bt
        accuracy = jnp.sum(correct_bt) / jnp.maximum(jnp.sum(loss_mask_bt), 1)

        metrics: dict[str, xax.Metric] = {
            "loss": xax.Scalar(loss),
            "perplexity": xax.Scalar(jnp.exp(loss)),
            "accuracy": xax.Scalar(accuracy),
        }

        if heavy:
            gen_key, _ = jax.random.split(key)
            audio_t, decoded_audio_t = self._generate_audio_jit(model, batch, gen_key)
            metrics["generated_audio"] = xax.Audio(audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)
            metrics["real_audio"] = xax.Audio(decoded_audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)

            # Teacher-forced audio: decode model predictions given ground truth context
            # This shows what the model would generate at each step with perfect context
            tf_audio_t = self._decode_teacher_forced(model, predictions_bt[0], targets_bt[0], loss_mask_bt[0])
            metrics["teacher_forced_audio"] = xax.Audio(tf_audio_t, sample_rate=xax.MIMI_SAMPLE_RATE)

            # Transcribe teacher-forced audio
            tf_transcript_tokens, _, _ = xax.transcribe_with_whisper(
                model=model.whisper,
                audio_t=tf_audio_t,
                eos_token_id=self.whisper_config.eos_token_id,
                max_tokens=64,
            )
            metrics["tf_transcript"] = xax.Tokens(tf_transcript_tokens, tokenizer="whisper")

            # Transcribe generated audio with Whisper for evaluation
            transcript_tokens, _, _ = xax.transcribe_with_whisper(
                model=model.whisper,
                audio_t=audio_t,
                eos_token_id=self.whisper_config.eos_token_id,
                max_tokens=64,
            )
            metrics["transcript"] = xax.Tokens(transcript_tokens, tokenizer="whisper")

            # Also transcribe the ground truth audio for comparison
            gt_transcript_tokens, _, _ = xax.transcribe_with_whisper(
                model=model.whisper,
                audio_t=decoded_audio_t,
                eos_token_id=self.whisper_config.eos_token_id,
                max_tokens=64,
            )
            metrics["gt_transcript"] = xax.Tokens(gt_transcript_tokens, tokenizer="whisper")

        return loss, metrics

    def _decode_teacher_forced(
        self, model: TTSModel, predictions_t: Array, targets_t: Array, mask_t: Array
    ) -> Array:
        """Decode teacher-forced predictions to audio.

        Takes the model's next-token predictions (given ground truth context) and decodes
        them to audio. This shows what the model would generate at each step with perfect
        context, helping debug whether the model is learning the audio distribution.

        Args:
            model: The TTS model
            predictions_t: Model predictions of shape (seq_len,)
            targets_t: Ground truth targets of shape (seq_len,)
            mask_t: Loss mask of shape (seq_len,)

        Returns:
            Audio waveform from teacher-forced predictions
        """
        num_quantizers = self.config.num_acoustic_quantizers + 1  # 8
        max_frames = self.config.max_audio_frames
        total_audio_tokens = num_quantizers * xax.MIMI_CODEBOOK_SIZE

        # Find audio token positions (where targets are audio tokens)
        is_audio = (targets_t >= self.audio_token_offset) & (
            targets_t < self.audio_token_offset + total_audio_tokens
        ) & mask_t

        # Extract audio predictions and convert to Mimi codes
        audio_preds = jnp.where(is_audio, predictions_t, self.audio_token_offset)
        audio_idx = audio_preds - self.audio_token_offset
        audio_idx = jnp.clip(audio_idx, 0, total_audio_tokens - 1)
        audio_codes_flat = audio_idx % xax.MIMI_CODEBOOK_SIZE

        # Count valid audio tokens
        num_audio_tokens = jnp.sum(is_audio)
        expected_tokens = max_frames * num_quantizers

        # Pad or truncate to expected length
        audio_codes_padded = jnp.zeros(expected_tokens, dtype=jnp.int32)
        # Extract only audio positions
        audio_positions = jnp.where(is_audio, jnp.arange(len(is_audio)), len(is_audio))
        sorted_positions = jnp.sort(audio_positions)
        valid_audio_codes = audio_codes_flat[sorted_positions[:expected_tokens]]
        audio_codes_padded = audio_codes_padded.at[:].set(
            jnp.where(jnp.arange(expected_tokens) < num_audio_tokens, valid_audio_codes, 0)
        )

        # Unflatten: (max_tokens,) -> (max_frames, 8) -> (8, max_frames)
        codes_tq = audio_codes_padded.reshape(max_frames, num_quantizers)
        codes_qt = codes_tq.T  # (8, max_frames)

        # Decode with Mimi
        audio_ct = model.mimi.decode(codes_qt)
        return audio_ct[0]

    def _generate_audio_jit(self, model: TTSModel, batch: Batch, key: PRNGKeyArray) -> tuple[Array, Array]:
        """Generate audio with custom generation loop using separate embeddings.

        The LLM generates all 8 quantizers interleaved per frame:
        [Q0_t0, Q1_t0, ..., Q7_t0, Q0_t1, Q1_t1, ..., Q7_t1, ...]

        Then we unflatten to (8, T) for Mimi decode.

        If audio_prompt_frames > 0, uses ground truth audio tokens from batch as prompt.
        """
        num_quantizers = self.config.num_acoustic_quantizers + 1  # 8
        max_frames = self.config.max_audio_frames
        max_tokens = max_frames * num_quantizers  # 8 tokens per frame
        prompt_tokens = self._eval_prompt_tokens
        text_len = prompt_tokens.shape[0]

        # Create combined weight matrix for logits
        text_weights = model.llm.lm_head.weight  # (text_vocab, hidden)
        audio_weights = model.audio_embed.weight  # (audio_vocab, hidden) - weight tying
        tied_weights = jnp.concatenate([text_weights, audio_weights], axis=0)

        # Optional: use ground truth audio as prompt
        audio_prompt_tokens = self.config.audio_prompt_frames * num_quantizers
        if audio_prompt_tokens > 0:
            # Get audio tokens from batch (first sample)
            # batch["input_ids"] has format: [text, audio_start, audio_tokens, audio_end]
            gt_audio_codes = batch["audio_codes"][0]  # (8, max_frames)
            # Flatten and convert to tokens
            gt_audio_flat = []
            for frame_idx in range(min(self.config.audio_prompt_frames, max_frames)):
                for q_idx in range(num_quantizers):
                    code = gt_audio_codes[q_idx, frame_idx]
                    token = self.audio_token_offset + q_idx * xax.MIMI_CODEBOOK_SIZE + code
                    gt_audio_flat.append(token)
            audio_prompt = jnp.array(gt_audio_flat, dtype=jnp.int32)
        else:
            audio_prompt = jnp.array([], dtype=jnp.int32)

        # Initialize output with text prompt + optional audio prompt
        initial_len = text_len + len(audio_prompt)
        tokens_to_generate = max_tokens - len(audio_prompt)
        total_len = text_len + max_tokens
        output = jnp.zeros(total_len, dtype=jnp.int32)
        output = output.at[:text_len].set(prompt_tokens)
        if len(audio_prompt) > 0:
            output = output.at[text_len:initial_len].set(audio_prompt)

        # Sampling parameters
        temperature = self.config.gen_temperature
        top_k = self.config.gen_top_k

        # Autoregressive generation with temperature sampling
        def generate_step(
            carry: tuple[Array, PRNGKeyArray], step_idx: Array
        ) -> tuple[tuple[Array, PRNGKeyArray], None]:
            tokens, rng = carry
            rng, sample_key = jax.random.split(rng)
            # pos is the position we want to generate
            pos = initial_len + step_idx
            # Get embeddings for full sequence
            embed_td = self._get_embeddings(model, tokens)
            # Forward through transformer
            hidden_td = self._forward_hidden_from_embed(model, embed_td)
            # Get hidden state at PREVIOUS position (which predicts current position)
            last_hidden = hidden_td[pos - 1]
            # Compute logits with tied weights
            logits = last_hidden @ tied_weights.T

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                # Get top-k values and indices
                top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
                # Create mask for non-top-k tokens
                mask = jnp.ones_like(logits) * float("-inf")
                mask = mask.at[top_k_indices].set(top_k_logits)
                logits = mask

            # Sample from distribution
            next_token = jax.random.categorical(sample_key, logits)

            # Update tokens at current position
            tokens = tokens.at[pos].set(next_token)
            return (tokens, rng), None

        # Run generation for remaining tokens
        if tokens_to_generate > 1:
            (output, key), _ = jax.lax.scan(
                generate_step,
                (output, key),
                jnp.arange(tokens_to_generate - 1),
            )
            # Generate the last token
            (output, key), _ = generate_step((output, key), jnp.array(tokens_to_generate - 1))
        elif tokens_to_generate == 1:
            (output, key), _ = generate_step((output, key), jnp.array(0))

        # Extract the audio region (after text prompt, includes audio prompt + generated)
        audio_region = jax.lax.dynamic_slice(output, (text_len,), (max_tokens,))

        # Filter to only valid audio tokens (total = num_quantizers * codebook_size)
        total_audio_tokens = num_quantizers * xax.MIMI_CODEBOOK_SIZE
        audio_token_mask = (audio_region >= self.audio_token_offset) & (
            audio_region < self.audio_token_offset + total_audio_tokens
        )

        # Debug: log audio token ratio
        audio_ratio = jnp.sum(audio_token_mask) / max_tokens
        jax.debug.print("Audio ratio: {x:.2f}, first 16 tokens: {y}", x=audio_ratio, y=audio_region[:16])

        # Convert to Mimi codes
        # Token = audio_offset + q_idx * CODEBOOK_SIZE + code
        # So: audio_idx = token - audio_offset, code = audio_idx % CODEBOOK_SIZE
        audio_idx = jnp.where(audio_token_mask, audio_region - self.audio_token_offset, 0)
        audio_codes_flat = audio_idx % xax.MIMI_CODEBOOK_SIZE
        audio_codes_flat = jnp.clip(audio_codes_flat, 0, xax.MIMI_CODEBOOK_SIZE - 1)

        # Unflatten: (max_tokens,) -> (max_frames, 8) -> (8, max_frames)
        # The flattening was: [Q0_t0, Q1_t0, ..., Q7_t0, Q0_t1, Q1_t1, ..., Q7_t1, ...]
        # So we reshape to (T, 8) and transpose to (8, T)
        codes_tq = audio_codes_flat.reshape(max_frames, num_quantizers)
        codes_qt = codes_tq.T  # (8, max_frames)

        # Decode with Mimi
        audio_ct = model.mimi.decode(codes_qt)

        # Also decode the ground truth audio sequence for reference
        gt_audio_codes = batch["audio_codes"][0]  # (8, max_frames)
        decoded_audio_ct = model.mimi.decode(gt_audio_codes)

        return audio_ct[0], decoded_audio_ct[0]

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
        return self.load_dataset("tokenized")

    @xax.dataset_fn("tokenized", use_hash=False)
    def tokenized_dataset(self) -> Dataset:
        columns = ["text_tokens", "audio_codes"]

        # Load raw dataset
        logger.info("Loading LJSpeech dataset...")
        raw_ds = load_dataset("keithito/lj_speech", split="train")

        # Stage 1: CPU-parallel audio resampling and normalization
        def resample_audio(example: dict) -> dict:
            """Resample and normalize audio (CPU-bound, parallelized)."""
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]

            # Resample to Mimi sample rate
            if sr != xax.MIMI_SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=xax.MIMI_SAMPLE_RATE)

            # Normalize
            audio = audio.astype(np.float32)
            max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
            audio = audio / max_val

            return {"resampled_audio": audio, "audio_length": len(audio)}

        logger.info("Stage 1: Resampling audio on CPU (parallel)...")
        resampled_ds = raw_ds.map(
            resample_audio,
            num_proc=32,
            desc="Resampling audio",
        )

        # Stage 2: GPU-batched Mimi encoding
        logger.info("Stage 2: Encoding audio with Mimi on GPU (batched)...")
        mimi = xax.build_pretrained_mimi(dtype=jnp.bfloat16)
        num_quantizers = self.config.num_acoustic_quantizers + 1

        # Find max audio length for batching
        max_audio_len = max(resampled_ds["audio_length"])
        logger.info("Max audio length: %d samples", max_audio_len)

        @jax.jit
        def batch_encode(audio_bct: Array) -> Array:
            return jax.vmap(lambda audio_ct: mimi.encode(audio_ct, num_quantizers=num_quantizers))(audio_bct)

        def encode_and_tokenize_batch(examples: dict[str, list]) -> dict[str, list]:
            """Encode audio with Mimi and tokenize text."""
            outputs: dict[str, list] = {column: [] for column in columns}
            audio_list = examples["resampled_audio"]
            bsz = len(audio_list)

            # Pad all audio to the same length and stack into batch. We always
            # pad to the max audio length to avoid recompiling the JIT function.
            audio_batch_np = np.zeros(shape=(bsz, max_audio_len), dtype=np.float32)
            for idx, audio in enumerate(audio_list):
                audio_batch_np[idx, : len(audio)] = audio
            audio_batch = jnp.array(audio_batch_np, dtype=jnp.bfloat16)

            # Batch encode with Mimi on GPU: (B, 1, T) -> (B, C, T')
            codes_batch_bct = batch_encode(audio_batch[:, None, :])
            codes_batch_np = np.array(codes_batch_bct)

            # Process each sample
            for idx in range(bsz):
                audio_codes_ct = codes_batch_np[idx]  # (C, T')
                orig_audio_len = examples["audio_length"][idx]

                # Estimate valid frames (Mimi has ~320x downsampling)
                estimated_frames = (orig_audio_len + 319) // 320
                actual_mimi_frames = audio_codes_ct.shape[1]
                valid_frames = min(estimated_frames, actual_mimi_frames)

                # Trim to valid frames and transpose to (T, C)
                audio_codes_tc = audio_codes_ct[:, :valid_frames].T  # (T, C)

                # Tokenize text
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
            batch_size=8,
            max_grad_norm=5.0,
            gradient_accumulation_steps=2,
            log_heavy_every_n_seconds=60,
            max_steps=60 * 60,  # 1 hour
            step_kind="second",
        ),
    )
