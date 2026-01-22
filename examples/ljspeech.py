#!/usr/bin/env -S uv run --no-project --script
"""Fine-tuning Qwen3 for text-to-speech using flattened codec tokens with BPE.

Architecture:
- Single LLM with LoRA fine-tuning
- Sequence format: [TEXT_START] [TEXT] [TEXT_END] [AUDIO_START] [AUDIO_BPE] [AUDIO_END]
- BPE compression on flattened codec tokens (Q0-Q7 interleaved)
- Audio embeddings and output head trained separately from text embeddings/logits
"""

import functools
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
TEXT_START_TOKEN = "<|text_start|>"
TEXT_END_TOKEN = "<|text_end|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

# LoRA targets
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "gate", "up")


class Batch(TypedDict):
    codes: Array  # (batch, seq_len) - combined text + audio tokens
    loss_mask: Array  # (batch, seq_len) - mask for loss computation (audio only)


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
        audio_bpe_tokenizer: Tokenizer,
        mimi: xax.MimiModel,
        key: PRNGKeyArray,
    ) -> "TTSModel":
        """Build TTS model from pretrained LLM.

        Args:
            llm: Pretrained LLM (can have LoRA applied)
            audio_bpe_vocab_size: Vocabulary size for audio BPE tokens
            audio_bpe_tokenizer: Tokenizer for decoding BPE tokens to Mimi values
            mimi: Pretrained Mimi model for codebook embeddings
            key: PRNG key for initialization

        Returns:
            TTSModel instance
        """
        k1, k2 = jax.random.split(key)
        embed_dim = llm.config.embed_dim

        # Get Mimi codebook embeddings for all 8 quantizers
        # Q0 is semantic, Q1-Q7 are acoustic
        # Each codebook has shape (2048, 256)
        base_char = 0xE000
        mimi_codebook_size = xax.MIMI_CODEBOOK_SIZE
        num_quantizers = NUM_QUANTIZERS

        # Stack codebook embeddings: (8, 2048, 256)
        codebook_embeds = []
        # Semantic quantizer (Q0)
        codebook_embeds.append(np.array(mimi.quantizer.semantic_rvq.layers[0].embeddings_kd))
        # Acoustic quantizers (Q1-Q7)
        for i in range(num_quantizers - 1):
            codebook_embeds.append(np.array(mimi.quantizer.acoustic_rvq.layers[i].embeddings_kd))
        codebook_embeds = np.stack(codebook_embeds, axis=0)  # (8, 2048, 256)

        mimi_embed_dim = codebook_embeds.shape[-1]  # 256

        # Create a projection from Mimi embed_dim (256) to LLM embed_dim (1024)
        # Use random orthogonal initialization for the projection
        proj_weight = jax.random.normal(k1, (embed_dim, mimi_embed_dim)) * (1.0 / np.sqrt(mimi_embed_dim))

        # Build audio BPE embeddings by averaging projected Mimi codebook embeddings
        # The flat codec sequence interleaves quantizers: [q0_t0, q1_t0, ..., q7_t0, q0_t1, ...]
        audio_embed_weights = np.zeros((audio_bpe_vocab_size, embed_dim), dtype=np.float32)

        for token_id in range(audio_bpe_vocab_size):
            token_str = audio_bpe_tokenizer.id_to_token(token_id)
            if token_str is None or len(token_str) == 0:
                audio_embed_weights[token_id] = np.array(
                    jax.random.normal(jax.random.fold_in(k2, token_id), (embed_dim,)) * 0.01
                )
                continue

            # Decode BPE token to sequence of Mimi codec values
            # Each position i in the flat sequence corresponds to quantizer (i % 8)
            embeds_list = []
            for pos, char in enumerate(token_str):
                codec_value = ord(char) - base_char
                codec_value = max(0, min(codec_value, mimi_codebook_size - 1))
                quantizer_idx = pos % num_quantizers
                # Get the embedding from the appropriate quantizer's codebook
                mimi_embed = codebook_embeds[quantizer_idx, codec_value]  # (256,)
                embeds_list.append(mimi_embed)

            # Average the Mimi embeddings and project to LLM dimension
            mean_mimi_embed = np.mean(np.stack(embeds_list, axis=0), axis=0)  # (256,)
            projected_embed = proj_weight @ mean_mimi_embed  # (embed_dim,)
            audio_embed_weights[token_id] = projected_embed

        audio_embed = eqx.nn.Embedding(audio_bpe_vocab_size, embed_dim, weight=jnp.array(audio_embed_weights))
        audio_head = eqx.nn.Linear(embed_dim, audio_bpe_vocab_size, key=k2)

        return TTSModel(
            llm=llm,
            audio_embed=audio_embed,
            audio_head=audio_head,
            audio_bpe_vocab_size=audio_bpe_vocab_size,
            text_vocab_size=llm.config.vocab_size,
        )

    def generate_audio(
        self,
        prompt_tokens_s: Array,
        max_audio_tokens: int,
        audio_end_id: int,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Generate audio BPE tokens given text prefix using greedy decoding.

        Builds a sequence [prompt] [audio_start] [generated...] and autoregressively
        generates audio BPE tokens by running the forward pass and taking argmax
        of the audio logits at each step.

        Args:
            prompt_tokens_s: Prompt token IDs, with shape (text_len + 2,)
            max_audio_tokens: Maximum number of audio tokens to generate
            audio_end_id: Audio EOS token ID
            key: PRNG key for sampling

        Returns:
            Tuple of (generated audio BPE token IDs, generated audio logits)
        """
        return xax.llm_generate_jit(
            self.llm,
            prompt_tokens_s,
            audio_end_id,
            max_audio_tokens,
            temperature=0.8,
            top_p=0.9,
            key=key,
        )


class FullTTSModel(eqx.Module):
    """Full TTS model with Mimi decoder and Whisper for evaluation."""

    tts: TTSModel
    mimi: xax.MimiModel
    whisper_transcriber: xax.WhisperTranscriber

    @staticmethod
    def build(
        llm: xax.LLM,
        audio_bpe_tokenizer: Tokenizer,
        whisper_eos_token_id: int,
        key: PRNGKeyArray,
    ) -> "FullTTSModel":
        audio_bpe_vocab_size = audio_bpe_tokenizer.get_vocab_size()
        mimi = xax.build_pretrained_mimi()
        tts = TTSModel.build(llm, audio_bpe_vocab_size, audio_bpe_tokenizer, mimi, key)
        whisper_model = xax.build_pretrained_whisper()
        whisper_transcriber = xax.WhisperTranscriber(
            model=whisper_model,
            eos_token_id=whisper_eos_token_id,
        )
        return FullTTSModel(
            tts=tts,
            mimi=mimi,
            whisper_transcriber=whisper_transcriber,
        )


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
    max_audio_frames: int = xax.field(256, help="Maximum audio frames")
    max_seq_length: int = xax.field(512, help="Maximum combined sequence length")
    audio_bpe_vocab_size: int = xax.field(151_669, help="Vocabulary size for audio BPE tokenizer")

    # Eval settings.
    eval_prompt: str = xax.field("Hello, world! I'm a TTS model.", help="Prompt to use for evaluation")


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
        """Build lookup tables for decoding audio BPE tokens to codec tokens."""
        tokenizer = self.audio_bpe_tokenizer
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

        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)
            if token_str is None:
                span_table[token_id] = 0
                continue

            # Convert unicode characters back to flattened codec tokens
            flat_codes = [ord(c) - base_char for c in token_str]
            span = len(flat_codes)
            span_table[token_id] = span
            decode_table[token_id, :span] = flat_codes

        audio_bpe_decode_table = jnp.array(decode_table)
        audio_bpe_span_table = jnp.array(span_table)
        logger.info(
            "Built audio BPE decode tables: vocab_size=%d, max_span=%d",
            vocab_size,
            max_span,
        )
        return audio_bpe_decode_table, audio_bpe_span_table

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
            audio_bpe_tokenizer=self.audio_bpe_tokenizer,
            whisper_eos_token_id=self.whisper_config.eos_token_id,
            key=mimi_key,
        )

        return model

    @override
    def get_trainable_filter_spec(self, model: FullTTSModel) -> FullTTSModel:
        # LLM LoRA parameters.
        llm_spec = xax.lora_filter_spec(model.tts.llm)

        # Extra head and embeddings should be trainable too.
        extra_embed_spec = jax.tree.map(lambda _: True, llm_spec.extra_embed)
        llm_spec = eqx.tree_at(lambda m: m.extra_embed, llm_spec, extra_embed_spec)

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
        codes_bs = batch["codes"]
        loss_mask_bs = batch["loss_mask"]

        # Forward pass
        loss, accuracy = jax.vmap(model.tts.llm.get_loss_and_accuracy, in_axes=(0, 0, 0, None))(
            codes_bs[:, :-1],
            codes_bs[:, 1:],
            loss_mask_bs,
            32,
        )

        loss = loss.mean()
        accuracy = accuracy.mean()

        metrics: dict[str, xax.Metric] = {
            "loss": xax.Scalar(loss),
            "accuracy": xax.Scalar(accuracy),
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

        return loss, metrics

    def _generate_audio(
        self,
        model: FullTTSModel,
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[Array, Array, Array]:
        """Generate audio using the model.

        Returns:
            Tuple of (generated audio, ground truth audio, ground truth text)
        """
        # Generate audio BPE tokens from the eval prompt
        max_audio_bpe = self.config.max_seq_length - len(self.eval_prompt_tokens)
        audio_bpe_tokens, audio_bpe_length = model.tts.generate_audio(
            prompt_tokens_s=self.eval_prompt_tokens,
            max_audio_tokens=max_audio_bpe,
            audio_end_id=self.audio_end_id,
            key=key,
        )

        # Decode BPE to codec tokens
        audio_codes_gen, _ = decode_audio_bpe_to_codes(
            bpe_tokens_b=audio_bpe_tokens - self.first_audio_bpe_id,
            bpe_length=audio_bpe_length,
            decode_table_vs=self.audio_bpe_decode_table,
            span_table_v=self.audio_bpe_span_table,
            num_quantizers=self.config.num_quantizers,
            max_frames=self.config.max_audio_frames,
        )

        # Get the ground truth codes from the batch.
        gt_ids = batch["codes"][0]
        gt_start_mask = gt_ids == self.audio_start_id
        first_start_idx = jnp.where(gt_start_mask.any(), jnp.argmax(gt_start_mask), len(gt_ids))
        gt_end_mask = gt_ids == self.audio_end_id
        first_end_idx = jnp.where(gt_end_mask.any(), jnp.argmax(gt_end_mask), len(gt_ids))
        gt_audio_ids = jax.lax.dynamic_slice(gt_ids, (first_start_idx,), (self.config.max_seq_length,))
        bpe_length = first_end_idx - first_start_idx

        gt_codes, _ = decode_audio_bpe_to_codes(
            bpe_tokens_b=gt_audio_ids - self.first_audio_bpe_id,
            bpe_length=bpe_length,
            decode_table_vs=self.audio_bpe_decode_table,
            span_table_v=self.audio_bpe_span_table,
            num_quantizers=self.config.num_quantizers,
            max_frames=self.config.max_audio_frames,
        )

        # Decode with Mimi
        audio_gen = model.mimi.decode(audio_codes_gen)
        audio_gt = model.mimi.decode(gt_codes)

        # Parses the ground truth text.
        gt_text_ids = jax.lax.dynamic_slice(gt_ids, (1,), (self.config.max_seq_length,))

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
                return self.tokenizer.decode(token_list, skip_special_tokens=True)

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
        - codes: Combined text + audio token IDs (audio offset by text_vocab_size)
        - loss_mask: True only for audio positions (we only train on audio prediction)
        """
        ds = self.load_dataset("bpe")
        max_seq_len = self.config.max_seq_length

        def prepare_sample(example: dict) -> dict:
            # Prepare tokens.
            text_tokens = np.array(example["text_tokens"], dtype=np.int32)
            text_tokens_with_special = np.concatenate([[self.text_start_id], text_tokens, [self.text_end_id]])
            text_with_special_len = len(text_tokens_with_special)
            audio_bpe = np.array(example["audio_bpe"], dtype=np.int32) + self.first_audio_bpe_id
            audio_bpe_len = len(audio_bpe)
            audio_bpe_with_special = np.concatenate([[self.audio_start_id], audio_bpe, [self.audio_end_id]])

            seq_parts = [text_tokens_with_special, audio_bpe_with_special]
            sequence = np.concatenate(seq_parts)

            # Truncate to max_seq_len
            seq_len = min(len(sequence), max_seq_len)
            sequence = sequence[:seq_len]

            # Pad to max_seq_len
            codes = np.full(max_seq_len, self.tokenizer.pad_token_id, dtype=np.int32)
            codes[:seq_len] = sequence

            # Loss mask: only on audio tokens (after AUDIO_START, excluding AUDIO_END)
            loss_start = text_with_special_len + 1
            loss_end_exclusive = loss_start + audio_bpe_len
            loss_mask = np.zeros(max_seq_len - 1, dtype=np.bool_)
            loss_mask[loss_start:loss_end_exclusive] = True

            return {"codes": codes, "loss_mask": loss_mask}

        result = ds.map(prepare_sample, desc="Preparing training data")

        # Remove unused columns
        cols_to_keep = ["codes", "loss_mask"]
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
            batch_size=16,
            max_grad_norm=5.0,
            gradient_accumulation_steps=4,
            log_heavy_every_n_seconds=60,
            step_kind="second",
        ),
    )
