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
NUM_ACOUSTIC_QUANTIZERS = 7  # Use Q1-Q7 for acoustic (8 total quantizers)

# Special tokens
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

# BPE special token IDs (from tokenizer config)
BPE_PAD_TOKEN = 0
BPE_UNK_TOKEN = 1
BPE_BOS_TOKEN = 2
BPE_EOS_TOKEN = 3

# Acoustic token special IDs (added to Mimi codebook)
# Mimi codebook is 2048, so we use 2048 and 2049 for BOS/EOS
ACOUSTIC_BOS_TOKEN = 2048
ACOUSTIC_EOS_TOKEN = 2049
ACOUSTIC_PAD_TOKEN = 2050
ACOUSTIC_VOCAB_SIZE = 2051  # 2048 Mimi codes + BOS + EOS + PAD

# LoRA targets (same as shakespeare.py)
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "gate", "up")


class Batch(TypedDict):
    text_tokens: Array  # (batch, max_text_length) - text for cross-attention
    text_mask: Array  # (batch, max_text_length) - valid text positions
    semantic_bpe: Array  # (batch, max_bpe_length) - BPE semantic tokens (T2S target)
    semantic_bpe_mask: Array  # (batch, max_bpe_length) - valid BPE positions
    bpe_spans: Array  # (batch, max_bpe_length) - spans for each BPE token
    dilated_semantic: Array  # (batch, max_audio_frames) - expanded semantic tokens (S2A input)
    audio_codes: Array  # (batch, max_audio_frames, 7) - codes S2A (BOS + tokens + EOS)
    audio_mask: Array  # (batch, max_audio_frames) - valid audio positions


class T2SModel(eqx.Module):
    """Text-to-Semantic model: generates BPE semantic tokens from text.

    Uses cross-attention to text embeddings for conditioning. The model autoregressively
    generates BPE-compressed semantic tokens, which are later expanded and used to
    generate acoustic tokens.

    Note: BPE embedding and output head weights are tied (shared) for parameter efficiency.
    """

    llm: xax.LLM  # Base LLM (with LoRA)
    cross_attn: xax.CrossAttentionBlock  # Shared cross-attention layer
    cross_attn_norm: xax.RMSNorm  # Pre-norm for cross-attention
    bpe_embed: eqx.nn.Embedding  # Embeddings for BPE tokens (also used as output head)
    text_embed_proj: eqx.nn.Linear | None  # Optional projection for text embeddings
    bpe_vocab_size: int = eqx.field(static=True)

    @staticmethod
    def build(
        llm: xax.LLM,
        bpe_vocab_size: int,
        key: PRNGKeyArray,
        bpe_embed_weights: Array | None = None,
    ) -> "T2SModel":
        """Build T2S model from pretrained LLM.

        Args:
            llm: Pretrained LLM (can have LoRA applied)
            bpe_vocab_size: Vocabulary size for BPE tokens
            key: PRNG key for initialization
            bpe_embed_weights: Optional pre-computed BPE embedding weights from Mimi,
                shape (bpe_vocab_size, embed_dim). If provided, uses these instead of random init.

        Returns:
            T2SModel instance
        """
        k1, k2 = jax.random.split(key, 2)
        embed_dim = llm.config.embed_dim

        cross_attn = xax.CrossAttentionBlock.build(
            embed_dim=embed_dim,
            num_heads=llm.config.q_heads,
            key=k1,
            use_rotary_embeddings=False,
        )
        cross_attn_norm = xax.RMSNorm.build(embed_dim, eps=llm.config.rms_eps)

        # BPE embedding - weights are also used for output projection (tied weights)
        if bpe_embed_weights is not None:
            assert bpe_embed_weights.shape == (bpe_vocab_size, embed_dim), (
                f"BPE embed shape mismatch: {bpe_embed_weights.shape} vs ({bpe_vocab_size}, {embed_dim})"
            )
            bpe_embed = eqx.nn.Embedding(bpe_vocab_size, embed_dim, weight=bpe_embed_weights)
        else:
            bpe_embed = eqx.nn.Embedding(bpe_vocab_size, embed_dim, key=k2)

        return T2SModel(
            llm=llm,
            cross_attn=cross_attn,
            cross_attn_norm=cross_attn_norm,
            bpe_embed=bpe_embed,
            text_embed_proj=None,
            bpe_vocab_size=bpe_vocab_size,
        )

    def encode_text(self, text_tokens_t: Array) -> Array:
        """Encode text tokens to embeddings for cross-attention.

        Args:
            text_tokens_t: Text token IDs, shape (T,)

        Returns:
            Text embeddings, shape (T, embed_dim)
        """
        text_embed_td = jax.vmap(self.llm.embed)(text_tokens_t)
        if self.text_embed_proj is not None:
            text_embed_td = jax.vmap(self.text_embed_proj)(text_embed_td)
        return text_embed_td

    def forward_hidden(
        self,
        bpe_tokens_t: Array,
        text_embeddings_sd: Array,
    ) -> Array:
        """Forward pass that returns pre-logit hidden features.

        Args:
            bpe_tokens_t: Input BPE token IDs, shape (T,)
            text_embeddings_sd: Encoded text, shape (S, embed_dim)

        Returns:
            Hidden features before logit projection, shape (T, embed_dim)
        """
        x_td = jax.vmap(self.bpe_embed)(bpe_tokens_t)

        for block in self.llm.blocks:
            x_td, _ = block.forward(x_td, cache=None)

            norm_x = jax.vmap(self.cross_attn_norm)(x_td)
            cross_out, _, _ = self.cross_attn.forward(
                q_tn=norm_x,
                kv_sn=text_embeddings_sd,
            )
            x_td = x_td + cross_out

        return self.llm.norm(x_td)

    def forward(
        self,
        bpe_tokens_t: Array,
        text_embeddings_sd: Array,
    ) -> Array:
        """Forward pass for training.

        Args:
            bpe_tokens_t: Input BPE token IDs, shape (T,)
            text_embeddings_sd: Encoded text, shape (S, embed_dim)

        Returns:
            Logits for next BPE token, shape (T, bpe_vocab_size)
        """
        x_td = self.forward_hidden(bpe_tokens_t, text_embeddings_sd)
        # Tied weights: use bpe_embed.weight.T for output projection
        logits_tv = x_td @ self.bpe_embed.weight.T
        return logits_tv

    def forward_with_hidden(
        self,
        bpe_tokens_t: Array,
        text_embeddings_sd: Array,
    ) -> tuple[Array, Array]:
        """Forward pass that returns both logits and hidden features.

        Args:
            bpe_tokens_t: Input BPE token IDs, shape (T,)
            text_embeddings_sd: Encoded text, shape (S, embed_dim)

        Returns:
            Tuple of (logits, hidden) where:
            - logits: shape (T, bpe_vocab_size)
            - hidden: shape (T, embed_dim)
        """
        x_td = self.forward_hidden(bpe_tokens_t, text_embeddings_sd)
        logits_tv = x_td @ self.bpe_embed.weight.T
        return logits_tv, x_td

    def generate(
        self,
        text_embeddings_sd: Array,
        max_length: int,
        bos_token: int = BPE_BOS_TOKEN,
        eos_token: int = BPE_EOS_TOKEN,
    ) -> tuple[Array, Array, Array]:
        """Autoregressively generate BPE tokens and return hidden features.

        Args:
            text_embeddings_sd: Encoded text, shape (S, embed_dim)
            max_length: Maximum number of tokens to generate
            bos_token: BOS token ID (default 2)
            eos_token: EOS token ID (default 3)

        Returns:
            Tuple of (tokens, length, hidden) where:
            - tokens: Generated token IDs, shape (max_length,)
            - length: Actual sequence length (position of first EOS or max_length)
            - hidden: Hidden features for each token, shape (max_length, embed_dim)
        """
        # Initialize with BOS token
        tokens = jnp.zeros(max_length, dtype=jnp.int32)
        tokens = tokens.at[0].set(bos_token)

        def generate_step(idx: int, tokens: Array) -> Array:
            # Forward pass with full sequence (uses causal masking internally)
            # We pass the entire tokens array; positions after idx are zeros (PAD)
            logits_tv = self.forward(tokens, text_embeddings_sd)
            # Get logits at position idx (predicting token at idx+1)
            next_logits = logits_tv[idx]
            next_token = jnp.argmax(next_logits)
            tokens = tokens.at[idx + 1].set(next_token)
            return tokens

        # Generate tokens autoregressively
        tokens = jax.lax.fori_loop(0, max_length - 1, generate_step, tokens)

        # Find actual length (first EOS position, or max_length if no EOS)
        eos_positions = jnp.where(tokens == eos_token, jnp.arange(max_length), max_length)
        length = jnp.min(eos_positions)

        # Get hidden features for the generated sequence
        hidden_td = self.forward_hidden(tokens, text_embeddings_sd)

        return tokens, length, hidden_td


def decode_bpe_to_semantic(
    bpe_tokens_b: Array,
    bpe_length: Array,
    decode_table_vs: Array,
    span_table_v: Array,
    max_semantic_length: int,
) -> tuple[Array, Array]:
    """Decode BPE tokens back to semantic tokens using lookup tables.

    Args:
        bpe_tokens_b: BPE token IDs, shape (B,) where B is max BPE length
        bpe_length: Actual number of valid BPE tokens (scalar)
        decode_table_vs: Lookup table mapping token ID -> semantic tokens, shape (V, S)
        span_table_v: Lookup table mapping token ID -> span length, shape (V,)
        max_semantic_length: Maximum output length for semantic tokens

    Returns:
        Tuple of (semantic_tokens, semantic_length) where:
        - semantic_tokens: Decoded semantic token IDs, shape (max_semantic_length,)
        - semantic_length: Actual number of semantic tokens (scalar)
    """
    # Look up spans and semantic tokens for each BPE token
    spans_b = span_table_v[bpe_tokens_b]  # (B,)
    semantic_seqs_bs = decode_table_vs[bpe_tokens_b]  # (B, S)

    # Compute cumulative positions for scattering
    # Position for BPE token i is sum of spans[0:i]
    cumsum_b = jnp.cumsum(spans_b)  # Exclusive end positions
    start_positions_b = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), cumsum_b[:-1]])

    # Total semantic length
    semantic_length = jnp.sum(jnp.where(jnp.arange(len(bpe_tokens_b)) < bpe_length, spans_b, 0))
    semantic_length = jnp.minimum(semantic_length, max_semantic_length)

    # Initialize output array
    semantic_tokens = jnp.zeros(max_semantic_length, dtype=jnp.int32)

    # Scatter semantic tokens from each BPE token into output
    max_span = decode_table_vs.shape[1]

    def scatter_bpe_token(carry: Array, inputs: tuple[Array, Array, Array, Array]) -> tuple[Array, None]:
        output, (start_pos, span, semantic_seq, bpe_idx) = carry, inputs
        # Only scatter if this BPE token is valid (idx < bpe_length)
        valid = bpe_idx < bpe_length

        def scatter_positions(pos_in_span: int, out: Array) -> Array:
            target_pos = start_pos + pos_in_span
            valid_pos = valid & (pos_in_span < span) & (target_pos < max_semantic_length)
            return jnp.where(valid_pos, out.at[target_pos].set(semantic_seq[pos_in_span]), out)

        output = jax.lax.fori_loop(0, max_span, scatter_positions, output)
        return output, None

    semantic_tokens, _ = jax.lax.scan(
        scatter_bpe_token,
        semantic_tokens,
        (start_positions_b, spans_b, semantic_seqs_bs, jnp.arange(len(bpe_tokens_b))),
    )

    return semantic_tokens, semantic_length


def dilate_bpe_features(
    bpe_hidden_bd: Array,
    bpe_spans_b: Array,
    max_semantic_length: int,
) -> Array:
    """Dilate BPE hidden features to semantic token positions.

    Each BPE token's hidden features are replicated for each semantic token
    it represents (based on its span). This allows the S2A model to receive
    context from the T2S model at each semantic token position.

    Args:
        bpe_hidden_bd: Hidden features for each BPE token, shape (B, D)
        bpe_spans_b: Number of semantic tokens each BPE token represents, shape (B,)
        max_semantic_length: Maximum output length

    Returns:
        Dilated hidden features, shape (max_semantic_length, D)
    """
    # Compute cumulative spans (end positions for each BPE token)
    cumsum_b = jnp.cumsum(bpe_spans_b)

    # For each output position, find which BPE token it belongs to
    # bpe_idx[i] = index of the BPE token that covers output position i
    output_positions_a = jnp.arange(max_semantic_length)
    bpe_idx_a = jnp.searchsorted(cumsum_b, output_positions_a, side="right")

    # Clamp to valid indices (positions beyond total span get last BPE token)
    bpe_idx_a = jnp.minimum(bpe_idx_a, bpe_hidden_bd.shape[0] - 1)

    # Index into BPE features to get dilated output
    return bpe_hidden_bd[bpe_idx_a]


def compute_bpe_embeddings_from_mimi(
    bpe_tokenizer: Tokenizer,
    mimi_semantic_embed_kd: Array,
    target_embed_dim: int,
    base_char: int = 0xE000,
) -> Array:
    """Compute BPE token embeddings as mean of constituent Mimi semantic embeddings.

    Each BPE token represents a sequence of semantic tokens. This function computes
    the embedding for each BPE token as the mean of the Mimi embeddings for its
    constituent semantic tokens, then tiles to match the target embedding dimension.

    Args:
        bpe_tokenizer: Trained BPE tokenizer
        mimi_semantic_embed_kd: Mimi semantic embeddings, shape (codebook_size, codebook_dim)
        target_embed_dim: Target embedding dimension (e.g., LLM embed_dim)
        base_char: Unicode base character used for semantic token encoding (default 0xE000)

    Returns:
        BPE embeddings, shape (bpe_vocab_size, target_embed_dim)
    """
    vocab_size = bpe_tokenizer.get_vocab_size()
    codebook_dim = mimi_semantic_embed_kd.shape[1]

    # Compute number of tiles needed to reach target_embed_dim
    num_tiles = (target_embed_dim + codebook_dim - 1) // codebook_dim  # Ceiling division

    # Special token IDs (these don't map to semantic tokens)
    special_token_ids = {BPE_PAD_TOKEN, BPE_UNK_TOKEN, BPE_BOS_TOKEN, BPE_EOS_TOKEN}

    bpe_embeddings = []
    for token_id in range(vocab_size):
        if token_id in special_token_ids:
            # Special tokens use zero embedding
            embed = np.zeros(codebook_dim)
        else:
            token_str = bpe_tokenizer.id_to_token(token_id)

            if token_str is None or len(token_str) == 0:
                embed = np.zeros(codebook_dim)
            else:
                # Decode unicode characters back to semantic token IDs
                semantic_ids = [ord(c) - base_char for c in token_str]
                # Filter valid semantic IDs (0 to codebook_size-1)
                valid_ids = [sid for sid in semantic_ids if 0 <= sid < mimi_semantic_embed_kd.shape[0]]

                if valid_ids:
                    # Mean of Mimi embeddings for constituent semantic tokens
                    embeds = np.array([mimi_semantic_embed_kd[sid] for sid in valid_ids])
                    embed = np.mean(embeds, axis=0)
                else:
                    embed = np.zeros(codebook_dim)

        # Tile to match target dimension, then truncate
        tiled = np.tile(embed, num_tiles)[:target_embed_dim]
        bpe_embeddings.append(tiled)

    return jnp.array(np.stack(bpe_embeddings, axis=0), dtype=jnp.float32)


class S2AModel(eqx.Module):
    """GRU-based autoregressive decoder for semantic-to-acoustic token generation.

    Architecture:
    - Takes dilated context vectors from T2S model (one per semantic token position)
    - Projects context to hidden_dim, then for each acoustic quantizer (Q1-Q7):
      - Applies a SwiGLU layer to the cumulative embedding
      - GRU autoregressively generates tokens within the layer
      - Adds the final token embedding to condition upper layers

    Uses BOS/EOS tokens for each quantizer layer's autoregressive generation.
    Embedding and head weights are untied but both initialized from Mimi embeddings.
    """

    context_proj: eqx.nn.Linear  # Project T2S context to hidden_dim
    acoustic_embeds: tuple[eqx.nn.Embedding, ...]  # One embedding per quantizer level (input)
    acoustic_heads: tuple[eqx.nn.Embedding, ...]  # One head per quantizer level (output, untied)
    acoustic_grus: tuple[eqx.nn.GRUCell, ...]  # GRU cells for autoregressive generation
    swiglu_layers: tuple[xax.SwiGLU, ...]  # SwiGLU layers between codec levels
    hidden_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    num_acoustic: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)  # Mimi codebook + BOS + EOS

    @staticmethod
    def build(
        vocab_size: int = ACOUSTIC_VOCAB_SIZE,
        hidden_dim: int = 256,
        context_dim: int = 1024,
        num_acoustic: int = 7,
        key: PRNGKeyArray = None,
        acoustic_embed_weights: tuple[Array, ...] | None = None,
    ) -> "S2AModel":
        """Build S2A model with GRU-based autoregressive generation.

        Args:
            vocab_size: Size of acoustic vocabulary (Mimi codebook + BOS + EOS = 2050)
            hidden_dim: Hidden dimension for embeddings (default 256 to match Mimi)
            context_dim: Dimension of T2S context vectors (LLM embed_dim)
            num_acoustic: Number of acoustic quantizers (Q1-Q7 = 7)
            key: PRNG key
            acoustic_embed_weights: Optional pre-trained acoustic embedding weights from Mimi,
                tuple of arrays each with shape (codebook_size, hidden_dim). BOS/EOS
                embeddings will be initialized randomly and appended.

        Returns:
            S2AModel instance
        """
        keys = jax.random.split(key, 1 + 4 * num_acoustic)

        # Project T2S context (LLM embed_dim) to S2A hidden_dim
        context_proj = eqx.nn.Linear(context_dim, hidden_dim, key=keys[0])

        # Initialize acoustic embeddings (from Mimi or random, plus BOS/EOS/PAD)
        def make_embed_with_special_tokens(mimi_weights: Array | None, key: PRNGKeyArray) -> eqx.nn.Embedding:
            if mimi_weights is not None:
                # Append random embeddings for BOS, EOS, and PAD tokens
                k1, k2 = jax.random.split(key)
                bos_emb = jax.random.normal(k1, (1, hidden_dim)) * 0.02
                eos_emb = jax.random.normal(k2, (1, hidden_dim)) * 0.02
                pad_emb = jnp.zeros((1, hidden_dim))  # PAD is zero embedding
                full_weights = jnp.concatenate([mimi_weights, bos_emb, eos_emb, pad_emb], axis=0)
                return eqx.nn.Embedding(vocab_size, hidden_dim, weight=full_weights)
            else:
                return eqx.nn.Embedding(vocab_size, hidden_dim, key=key)

        if acoustic_embed_weights is not None:
            assert len(acoustic_embed_weights) == num_acoustic, (
                f"Expected {num_acoustic} acoustic embeddings, got {len(acoustic_embed_weights)}"
            )
            acoustic_embeds = tuple(
                make_embed_with_special_tokens(acoustic_embed_weights[i], keys[1 + i]) for i in range(num_acoustic)
            )
        else:
            acoustic_embeds = tuple(
                eqx.nn.Embedding(vocab_size, hidden_dim, key=keys[1 + i]) for i in range(num_acoustic)
            )

        # Initialize acoustic heads (untied from embeddings)
        if acoustic_embed_weights is not None:
            acoustic_heads = tuple(
                make_embed_with_special_tokens(acoustic_embed_weights[i], keys[1 + num_acoustic + i])
                for i in range(num_acoustic)
            )
        else:
            acoustic_heads = tuple(
                eqx.nn.Embedding(vocab_size, hidden_dim, key=keys[1 + num_acoustic + i]) for i in range(num_acoustic)
            )

        # GRU cells for autoregressive generation within each layer
        acoustic_grus = tuple(
            eqx.nn.GRUCell(hidden_dim, hidden_dim, key=keys[1 + 2 * num_acoustic + i]) for i in range(num_acoustic)
        )

        # SwiGLU layers between codec levels for added capacity
        swiglu_layers = tuple(
            xax.SwiGLU.build(embed_dim=hidden_dim, key=keys[1 + 3 * num_acoustic + i]) for i in range(num_acoustic)
        )

        return S2AModel(
            context_proj=context_proj,
            acoustic_embeds=acoustic_embeds,
            acoustic_heads=acoustic_heads,
            acoustic_grus=acoustic_grus,
            swiglu_layers=swiglu_layers,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_acoustic=num_acoustic,
            vocab_size=vocab_size,
        )

    def forward_step(
        self,
        hidden_states_7d: Array,
        context_d: Array,
        acoustic_input_7: Array,
        acoustic_target_7: Array,
    ) -> tuple[Array, Array]:
        """Compute logits for one timestep using teacher forcing.

        Each quantizer layer uses a GRU that maintains hidden state across timesteps.
        The GRU input is the embedding of the previous token (from acoustic_input).

        Args:
            hidden_states_7d: GRU hidden states from previous timestep, shape (7, hidden_dim)
            context_d: Context vector from T2S (dilated), shape (context_dim,)
            acoustic_input_7: Input tokens for this timestep (prev tokens or BOS), shape (7,)
            acoustic_target_7: Target tokens for this timestep (for layer conditioning), shape (7,)

        Returns:
            Tuple of (new_hidden_states, logits) where:
            - new_hidden_states: Updated GRU hidden states, shape (7, hidden_dim)
            - logits: Logits for all acoustic tokens, shape (7, vocab_size)
        """
        # Project context to hidden_dim
        projected_d = self.context_proj(context_d)

        logits_list = []
        new_hidden_list = []
        cumulative_emb_d = projected_d

        for layer_idx in range(self.num_acoustic):
            # SwiGLU layer for added capacity (expects 2D input)
            swiglu_out_d = self.swiglu_layers[layer_idx](cumulative_emb_d[None, :])[0]
            cumulative_emb_d = cumulative_emb_d + swiglu_out_d

            # GRU input: cumulative embedding + input token embedding (prev token or BOS)
            input_emb_d = self.acoustic_embeds[layer_idx](acoustic_input_7[layer_idx])
            gru_input_d = cumulative_emb_d + input_emb_d

            # GRU step with hidden state from previous timestep
            prev_hidden_d = hidden_states_7d[layer_idx]
            new_hidden_d = self.acoustic_grus[layer_idx](gru_input_d, prev_hidden_d)
            new_hidden_list.append(new_hidden_d)

            # Compute logits with head weights
            logits_v = new_hidden_d @ self.acoustic_heads[layer_idx].weight.T
            logits_list.append(logits_v)

            # Add this layer's target embedding for conditioning upper layers (teacher forcing)
            target_emb_d = self.acoustic_embeds[layer_idx](acoustic_target_7[layer_idx])
            cumulative_emb_d = cumulative_emb_d + target_emb_d

        return jnp.stack(new_hidden_list, axis=0), jnp.stack(logits_list, axis=0)

    def forward_all(
        self,
        context_td: Array,
        acoustic_inputs_t7: Array,
        acoustic_targets_t7: Array,
    ) -> Array:
        """Forward pass for training: compute logits for all positions and quantizers.

        Uses scan to carry GRU hidden states across timesteps for temporal autoregression.
        Teacher forcing: input at timestep t is the token from t-1 (or BOS for t=0).

        Args:
            context_td: Dilated context vectors from T2S, shape (T, context_dim)
            acoustic_inputs_t7: Input tokens (shifted targets with BOS), shape (T, 7)
            acoustic_targets_t7: Target acoustic tokens Q1-Q7, shape (T, 7)

        Returns:
            Logits for all acoustic tokens, shape (T, 7, vocab_size)
        """
        # Initialize hidden states to zeros
        init_hidden_7d = jnp.zeros((self.num_acoustic, self.hidden_dim))

        def scan_fn(hidden_7d: Array, inputs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
            ctx_d, inp_7, tgt_7 = inputs
            new_hidden_7d, logits_7v = self.forward_step(hidden_7d, ctx_d, inp_7, tgt_7)
            return new_hidden_7d, logits_7v

        # Scan over timesteps, carrying hidden state
        _, logits_t7v = jax.lax.scan(
            scan_fn,
            init_hidden_7d,
            (context_td, acoustic_inputs_t7, acoustic_targets_t7),
        )

        return logits_t7v

    def generate_step(
        self,
        hidden_states_7d: Array,
        context_d: Array,
        prev_tokens_7: Array,
    ) -> tuple[Array, Array]:
        """Generate acoustic tokens for one timestep autoregressively across layers.

        Args:
            hidden_states_7d: GRU hidden states from previous timestep, shape (7, hidden_dim)
            context_d: Context vector from T2S (dilated), shape (context_dim,)
            prev_tokens_7: Tokens from previous timestep (or BOS for t=0), shape (7,)

        Returns:
            Tuple of (new_hidden_states, generated_tokens) where:
            - new_hidden_states: Updated GRU hidden states, shape (7, hidden_dim)
            - generated_tokens: Generated acoustic tokens, shape (7,)
        """
        # Project context to hidden_dim
        projected_d = self.context_proj(context_d)

        tokens_list = []
        new_hidden_list = []
        cumulative_emb_d = projected_d

        for layer_idx in range(self.num_acoustic):
            # SwiGLU layer for added capacity (expects 2D input)
            swiglu_out_d = self.swiglu_layers[layer_idx](cumulative_emb_d[None, :])[0]
            cumulative_emb_d = cumulative_emb_d + swiglu_out_d

            # GRU input: cumulative embedding + prev token embedding
            prev_emb_d = self.acoustic_embeds[layer_idx](prev_tokens_7[layer_idx])
            gru_input_d = cumulative_emb_d + prev_emb_d

            # GRU step with hidden state from previous timestep
            prev_hidden_d = hidden_states_7d[layer_idx]
            new_hidden_d = self.acoustic_grus[layer_idx](gru_input_d, prev_hidden_d)
            new_hidden_list.append(new_hidden_d)

            # Compute logits and take argmax (excluding BOS/EOS from generation)
            logits_v = new_hidden_d @ self.acoustic_heads[layer_idx].weight.T
            # Mask out BOS/EOS tokens during generation
            logits_v = logits_v.at[ACOUSTIC_BOS_TOKEN].set(-jnp.inf)
            logits_v = logits_v.at[ACOUSTIC_EOS_TOKEN].set(-jnp.inf)
            token = jnp.argmax(logits_v)
            tokens_list.append(token)

            # Add this layer's predicted embedding for conditioning upper layers
            layer_emb_d = self.acoustic_embeds[layer_idx](token)
            cumulative_emb_d = cumulative_emb_d + layer_emb_d

        return jnp.stack(new_hidden_list, axis=0), jnp.stack(tokens_list, axis=0)

    def generate(self, context_td: Array) -> Array:
        """Generate acoustic tokens given context vectors.

        Uses scan to carry GRU hidden states across timesteps for temporal autoregression.

        Args:
            context_td: Dilated context vectors from T2S, shape (T, context_dim)

        Returns:
            Generated acoustic tokens, shape (T, 7)
        """
        # Initialize hidden states to zeros and prev tokens to BOS
        init_hidden_7d = jnp.zeros((self.num_acoustic, self.hidden_dim))
        init_tokens_7 = jnp.full((self.num_acoustic,), ACOUSTIC_BOS_TOKEN, dtype=jnp.int32)

        def scan_fn(carry: tuple[Array, Array], ctx_d: Array) -> tuple[tuple[Array, Array], Array]:
            hidden_7d, prev_tokens_7 = carry
            new_hidden_7d, new_tokens_7 = self.generate_step(hidden_7d, ctx_d, prev_tokens_7)
            return (new_hidden_7d, new_tokens_7), new_tokens_7

        # Scan over timesteps, carrying hidden state and previous tokens
        _, tokens_t7 = jax.lax.scan(scan_fn, (init_hidden_7d, init_tokens_7), context_td)

        return tokens_t7


class TwoPartTTSModel(eqx.Module):
    """Combined two-part TTS model: T2S + S2A + Mimi + Whisper."""

    t2s: T2SModel
    s2a: S2AModel
    mimi: xax.MimiModel
    whisper_transcriber: xax.WhisperTranscriber

    @staticmethod
    def build(
        llm: xax.LLM,
        bpe_vocab_size: int,
        whisper_eos_token_id: int,
        s2a_hidden_dim: int | None = None,
        bpe_tokenizer: Tokenizer | None = None,
        key: PRNGKeyArray = None,
    ) -> "TwoPartTTSModel":
        """Build the two-part TTS model.

        Args:
            llm: Pretrained LLM (with LoRA applied)
            bpe_vocab_size: Vocabulary size for BPE tokens
            whisper_eos_token_id: EOS token ID for Whisper transcription
            s2a_hidden_dim: Hidden dimension for S2A stacked GRU. If None, uses Mimi's
                codebook_dim (256) to enable embedding initialization from Mimi.
            bpe_tokenizer: Optional BPE tokenizer for computing BPE embeddings from Mimi.
                If provided, BPE embeddings are initialized as the mean of Mimi semantic
                embeddings for each BPE token's constituent tokens.
            key: PRNG key

        Returns:
            TwoPartTTSModel instance
        """
        k1, k2 = jax.random.split(key)

        # Load Mimi first to extract embeddings for S2A initialization
        mimi = xax.build_pretrained_mimi()

        # Use Mimi's codebook_dim for S2A hidden_dim to enable embedding initialization
        if s2a_hidden_dim is None:
            s2a_hidden_dim = mimi.config.codebook_dim  # 256

        # Extract embeddings from Mimi for S2A and BPE initialization
        # Semantic embedding from Q0 (for BPE embedding computation)
        semantic_embed_weights = mimi.quantizer.semantic_rvq.layers[0].embeddings_kd

        # Acoustic embeddings from Q1-Q7 (acoustic RVQ layers 0-6)
        acoustic_embed_weights = tuple(
            mimi.quantizer.acoustic_rvq.layers[i].embeddings_kd for i in range(NUM_ACOUSTIC_QUANTIZERS)
        )

        # Compute BPE embeddings from Mimi semantic embeddings if tokenizer provided
        bpe_embed_weights = None
        if bpe_tokenizer is not None:
            logger.info("Computing BPE embeddings from Mimi semantic embeddings...")
            bpe_embed_weights = compute_bpe_embeddings_from_mimi(
                bpe_tokenizer=bpe_tokenizer,
                mimi_semantic_embed_kd=np.array(semantic_embed_weights),
                target_embed_dim=llm.config.embed_dim,
            )
            logger.info("BPE embeddings computed: shape %s", bpe_embed_weights.shape)

        t2s = T2SModel.build(
            llm=llm,
            bpe_vocab_size=bpe_vocab_size,
            key=k1,
            bpe_embed_weights=bpe_embed_weights,
        )
        s2a = S2AModel.build(
            vocab_size=ACOUSTIC_VOCAB_SIZE,
            hidden_dim=s2a_hidden_dim,
            context_dim=llm.config.embed_dim,
            num_acoustic=NUM_ACOUSTIC_QUANTIZERS,
            key=k2,
            acoustic_embed_weights=acoustic_embed_weights,
        )
        whisper_model = xax.build_pretrained_whisper()
        whisper_transcriber = xax.WhisperTranscriber(
            model=whisper_model,
            eos_token_id=whisper_eos_token_id,
            dtype=jnp.bfloat16,  # Match model dtype
        )

        return TwoPartTTSModel(t2s=t2s, s2a=s2a, mimi=mimi, whisper_transcriber=whisper_transcriber)


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
    max_audio_frames: int = xax.field(256, help="Maximum audio frames")

    # Two-part architecture settings
    s2a_hidden_dim: int = xax.field(256, help="S2A hidden dimension (256 = Mimi codebook_dim)")
    t2s_weight: float = xax.field(1.0, help="Loss weight for T2S (semantic) loss")
    s2a_weight: float = xax.field(1.0, help="Loss weight for S2A (acoustic) loss")
    s2a_layer_weights: tuple[float, ...] = xax.field(
        (1.0, 0.7, 0.5, 0.4, 0.3, 0.3, 0.3),
        help="Per-layer weights for S2A loss (Q1-Q7)",
    )
    max_bpe_length: int = xax.field(64, help="Maximum BPE sequence length")

    # Generation settings
    gen_temperature: float = xax.field(0.8, help="Temperature for generation sampling")
    gen_top_k: int = xax.field(50, help="Top-k for generation sampling (0 = no top-k)")
    max_gen_bpe_tokens: int = xax.field(256, help="Max BPE tokens to generate")

    # Data settings
    processed_data_path: str | None = xax.field(None, help="Path to pre-processed data")
    bpe_vocab_size: int = xax.field(151_669, help="Vocabulary size for BPE tokenizer on semantic tokens")


class LJSpeechTTS(xax.SupervisedTask[Config]):
    """Two-part TTS: T2S (text-to-semantic) + S2A (semantic-to-acoustic)."""

    tokenizer: Qwen2TokenizerFast
    whisper_tokenizer: WhisperTokenizerFast
    bpe_tokenizer: Tokenizer | None
    # BPE decode tables: map token ID -> semantic tokens
    bpe_decode_table: Array | None  # (vocab_size, max_span)
    bpe_span_table: Array | None  # (vocab_size,)

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_repo.value)

        # Load Whisper model for ASR evaluation (frozen)
        logger.info("Loading Whisper model for ASR evaluation")
        path = xax.download_whisper_repo()
        self.whisper_config = xax.load_whisper_config()
        self.whisper_tokenizer = WhisperTokenizerFast.from_pretrained(str(path))

        # BPE tokenizer and decode tables are loaded lazily
        self.bpe_tokenizer = None
        self.bpe_decode_table = None
        self.bpe_span_table = None

    def _load_bpe_tokenizer(self) -> Tokenizer:
        """Load BPE tokenizer from cache directory."""
        if self.bpe_tokenizer is None:
            tokenizer_path = self.dataset_cache_dir / "bpe_tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"BPE tokenizer not found at {tokenizer_path}. Run with --launcher dataset first to generate it."
                )
            self.bpe_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return self.bpe_tokenizer

    def _build_bpe_decode_tables(self) -> tuple[Array, Array]:
        """Build lookup tables for decoding BPE tokens to semantic tokens.

        Returns:
            Tuple of (decode_table, span_table) where:
            - decode_table: shape (vocab_size, max_span) - semantic tokens for each BPE token
            - span_table: shape (vocab_size,) - number of semantic tokens per BPE token
        """
        if self.bpe_decode_table is not None:
            return self.bpe_decode_table, self.bpe_span_table

        tokenizer = self._load_bpe_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        base_char = 0xE000  # Private Use Area start (matches BPE encoding)

        # Find max span by checking all tokens
        max_span = 1
        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)
            if token_str is not None:
                max_span = max(max_span, len(token_str))

        # Build decode table: each row is the semantic token sequence for that BPE token
        decode_table = np.zeros((vocab_size, max_span), dtype=np.int32)
        span_table = np.zeros(vocab_size, dtype=np.int32)

        # Special token IDs (these don't map to semantic tokens)
        special_token_ids = {BPE_PAD_TOKEN, BPE_UNK_TOKEN, BPE_BOS_TOKEN, BPE_EOS_TOKEN}

        for token_id in range(vocab_size):
            if token_id in special_token_ids:
                # Special tokens don't map to semantic tokens
                span_table[token_id] = 0
                continue

            token_str = tokenizer.id_to_token(token_id)
            if token_str is None:
                span_table[token_id] = 0
                continue

            # Convert unicode characters back to semantic tokens
            semantic_tokens = [ord(c) - base_char for c in token_str]
            span = len(semantic_tokens)
            span_table[token_id] = span
            decode_table[token_id, :span] = semantic_tokens

        self.bpe_decode_table = jnp.array(decode_table)
        self.bpe_span_table = jnp.array(span_table)
        logger.info(
            "Built BPE decode tables: vocab_size=%d, max_span=%d",
            vocab_size,
            max_span,
        )
        return self.bpe_decode_table, self.bpe_span_table

    @override
    def get_model(self, params: xax.InitParams) -> TwoPartTTSModel:
        # Build LLM with LoRA
        llm = xax.build_pretrained_llm(self.config.llm_repo)
        llm = xax.loraize_by_path(
            llm,
            rank=self.config.lora_rank,
            include_suffixes=list(self.config.lora_targets) if self.config.lora_targets else None,
            alpha=self.config.lora_alpha,
            dropout_rate=self.config.lora_dropout,
            key=params.key,
        )

        # Load BPE tokenizer for embedding initialization
        bpe_tokenizer = self._load_bpe_tokenizer()

        k_model = jax.random.fold_in(params.key, 1)
        model = TwoPartTTSModel.build(
            llm=llm,
            bpe_vocab_size=self.config.bpe_vocab_size,
            whisper_eos_token_id=self.whisper_config.eos_token_id,
            s2a_hidden_dim=self.config.s2a_hidden_dim,
            bpe_tokenizer=bpe_tokenizer,
            key=k_model,
        )

        return model

    @override
    def get_trainable_filter_spec(self, model: TwoPartTTSModel) -> TwoPartTTSModel:
        # T2S: LoRA layers + cross-attention + BPE embeddings (tied with head)
        llm_spec = xax.lora_filter_spec(model.t2s.llm)
        cross_attn_spec = jax.tree.map(lambda _: True, model.t2s.cross_attn)
        cross_attn_norm_spec = jax.tree.map(lambda _: True, model.t2s.cross_attn_norm)
        bpe_embed_spec = jax.tree.map(lambda _: True, model.t2s.bpe_embed)
        text_embed_proj_spec = (
            jax.tree.map(lambda _: True, model.t2s.text_embed_proj) if model.t2s.text_embed_proj is not None else None
        )

        t2s_spec = T2SModel(
            llm=llm_spec,
            cross_attn=cross_attn_spec,
            cross_attn_norm=cross_attn_norm_spec,
            bpe_embed=bpe_embed_spec,
            text_embed_proj=text_embed_proj_spec,
            bpe_vocab_size=model.t2s.bpe_vocab_size,
        )

        # S2A: All layers trainable
        s2a_spec = jax.tree.map(lambda _: True, model.s2a)

        # Mimi and Whisper: Frozen
        mimi_spec = jax.tree.map(lambda _: False, model.mimi)
        whisper_transcriber_spec = jax.tree.map(lambda _: False, model.whisper_transcriber)

        return TwoPartTTSModel(
            t2s=t2s_spec,
            s2a=s2a_spec,
            mimi=mimi_spec,
            whisper_transcriber=whisper_transcriber_spec,
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

    @override
    def compute_loss(
        self,
        model: TwoPartTTSModel,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        # T2S Loss: next-token prediction on BPE semantic sequence
        # Encode text for cross-attention
        text_embeddings_bsd = jax.vmap(model.t2s.encode_text)(batch["text_tokens"])

        # BPE input/target: shift by one for next-token prediction
        bpe_input_bs = batch["semantic_bpe"][:, :-1]
        bpe_target_bs = batch["semantic_bpe"][:, 1:]
        bpe_mask_bs = batch["semantic_bpe_mask"][:, 1:]

        # T2S forward: get both logits and hidden features
        t2s_logits_bsv, t2s_hidden_bsd = jax.vmap(model.t2s.forward_with_hidden)(bpe_input_bs, text_embeddings_bsd)
        t2s_loss_bs = optax.softmax_cross_entropy_with_integer_labels(t2s_logits_bsv, bpe_target_bs)
        t2s_loss = jnp.sum(t2s_loss_bs * bpe_mask_bs) / jnp.maximum(jnp.sum(bpe_mask_bs), 1)

        # T2S accuracy
        t2s_preds_bs = jnp.argmax(t2s_logits_bsv, axis=-1)
        t2s_correct_bs = (t2s_preds_bs == bpe_target_bs) & bpe_mask_bs
        t2s_accuracy = jnp.sum(t2s_correct_bs) / jnp.maximum(jnp.sum(bpe_mask_bs), 1)

        # Dilate T2S hidden features to semantic token positions
        # Use BPE spans (excluding BOS at position 0 to match bpe_input_bs)
        # Context length is max_audio_frames - 1 since audio_codes has BOS prepended
        bpe_spans_bs = batch["bpe_spans"][:, :-1]
        dilated_context_bad = jax.vmap(dilate_bpe_features, in_axes=(0, 0, None))(
            t2s_hidden_bsd, bpe_spans_bs, self.config.max_audio_frames - 1
        )

        # S2A Loss: autoregressive prediction of Q1-Q7 given dilated context
        # audio_codes has shape (B, A, 7) with BOS at position 0, EOS at the end
        # Derive inputs (shifted right) and targets (shifted left)
        audio_codes_ba7 = batch["audio_codes"]
        acoustic_inputs_ba7 = audio_codes_ba7[:, :-1, :]  # BOS through second-to-last
        acoustic_targets_ba7 = audio_codes_ba7[:, 1:, :]  # First token through EOS
        audio_mask_ba = batch["audio_mask"][:, 1:]  # Shift mask to match targets

        # S2A forward: compute logits for all quantizers using dilated context
        # Uses scan to carry GRU hidden states across timesteps
        s2a_logits_ba7v = jax.vmap(model.s2a.forward_all)(
            dilated_context_bad, acoustic_inputs_ba7, acoustic_targets_ba7
        )

        # S2A loss: cross-entropy for each quantizer
        s2a_loss_ba7 = optax.softmax_cross_entropy_with_integer_labels(
            s2a_logits_ba7v.reshape(-1, s2a_logits_ba7v.shape[-1]),
            acoustic_targets_ba7.reshape(-1),
        ).reshape(acoustic_targets_ba7.shape)

        # Apply per-layer weights (Q1-Q7)
        layer_weights_7 = jnp.array(self.config.s2a_layer_weights)
        s2a_loss_ba7 = s2a_loss_ba7 * layer_weights_7

        # Mask and average (normalize by sum of weights, not count)
        s2a_mask_ba7 = audio_mask_ba[..., None]  # (B, A, 1) broadcast to (B, A, 7)
        weighted_mask_ba7 = s2a_mask_ba7 * layer_weights_7
        s2a_loss = jnp.sum(s2a_loss_ba7 * s2a_mask_ba7) / jnp.maximum(jnp.sum(weighted_mask_ba7), 1)

        # S2A accuracy (unweighted)
        s2a_preds_ba7 = jnp.argmax(s2a_logits_ba7v, axis=-1)
        s2a_correct_ba7 = (s2a_preds_ba7 == acoustic_targets_ba7) & audio_mask_ba[..., None]
        s2a_accuracy = jnp.sum(s2a_correct_ba7) / jnp.maximum(jnp.sum(audio_mask_ba) * 7, 1)

        # Combined loss
        loss = self.config.t2s_weight * t2s_loss + self.config.s2a_weight * s2a_loss

        metrics: dict[str, xax.Metric] = {
            "loss": xax.Scalar(loss),
            "t2s_loss": xax.Scalar(t2s_loss),
            "s2a_loss": xax.Scalar(s2a_loss),
            "t2s_accuracy": xax.Scalar(t2s_accuracy),
            "s2a_accuracy": xax.Scalar(s2a_accuracy),
        }

        if heavy:
            gen_key = jax.random.fold_in(key, state.num_steps)
            # Build BPE decode tables if needed (lazy initialization)
            bpe_decode_table, bpe_span_table = self._build_bpe_decode_tables()
            audio_t, gt_audio_t = self._generate_audio(model, batch, gen_key, bpe_decode_table, bpe_span_table)
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
        model: TwoPartTTSModel,
        batch: Batch,
        key: PRNGKeyArray,
        bpe_decode_table: Array,
        bpe_span_table: Array,
    ) -> tuple[Array, Array]:
        """Generate audio using full end-to-end pipeline.

        Pipeline:
        1. T2S: Encode text and autoregressively generate BPE tokens + hidden features
        2. Decode BPE to semantic tokens
        3. Dilate T2S hidden features to semantic token positions
        4. S2A: Generate acoustic tokens from dilated context
        5. Mimi: Decode all codes to audio

        Returns:
            Tuple of (generated audio, ground truth audio)
        """
        # Step 1: Encode text with T2S
        text_tokens = batch["text_tokens"][0]
        text_embeddings_sd = model.t2s.encode_text(text_tokens)

        # Step 2: Generate BPE tokens and hidden features autoregressively
        max_bpe_length = self.config.max_bpe_length
        bpe_tokens, bpe_length, bpe_hidden_bd = model.t2s.generate(
            text_embeddings_sd,
            max_length=max_bpe_length,
            bos_token=BPE_BOS_TOKEN,
            eos_token=BPE_EOS_TOKEN,
        )

        # Step 3: Decode BPE to semantic tokens
        max_semantic = self.config.max_audio_frames
        semantic_tokens, semantic_length = decode_bpe_to_semantic(
            bpe_tokens_b=bpe_tokens,
            bpe_length=bpe_length,
            decode_table_vs=bpe_decode_table,
            span_table_v=bpe_span_table,
            max_semantic_length=max_semantic,
        )

        # Step 4: Dilate T2S hidden features to semantic token positions
        bpe_spans_b = bpe_span_table[bpe_tokens]
        dilated_context_ad = dilate_bpe_features(bpe_hidden_bd, bpe_spans_b, max_semantic)

        # Step 5: Generate acoustic tokens using dilated context
        acoustic_tokens_a7 = model.s2a.generate(dilated_context_ad)  # (A, 7)

        # Step 5: Combine Q0 (semantic) + Q1-Q7 (predicted acoustic) and decode
        all_codes_8a = jnp.concatenate(
            [
                semantic_tokens[None, :],  # (1, A)
                acoustic_tokens_a7.T,  # (7, A)
            ],
            axis=0,
        )  # (8, A)

        audio_ct = model.mimi.decode(all_codes_8a)

        # Ground truth audio: Q0 + GT Q1-Q7
        # audio_codes has BOS at position 0, so skip it for actual tokens
        gt_semantic = batch["dilated_semantic"][0]
        gt_acoustic_a7 = batch["audio_codes"][0, 1:, :]  # Skip BOS, shape (A-1, 7)
        gt_acoustic = gt_acoustic_a7.T  # (7, A-1)
        gt_codes_8a = jnp.concatenate([gt_semantic[None, :], gt_acoustic], axis=0)
        gt_audio_ct = model.mimi.decode(gt_codes_8a)

        return audio_ct[0], gt_audio_ct[0]

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
        """Creates the final dataset that we use to train the model.

        The training dataset has the following columns:
         - text_tokens: Transcribed audio tokens, which we cross-attend to when
           generating the semantic tokens autoregressively, shape (T).
         - text_mask: Binary mask for the text tokens, shape (T).
         - semantic_bpe: BPE-encoded semantic tokens that we generate
           autoregressively, shape (S).
         - semantic_bpe_mask: Binary mask for the semantic tokens, shape (S).
         - bpe_spans: Number of original tokens each BPE token represents, shape (S).
         - dilated_semantic: The semantic tokens expanded from BPE,
           with 1:1 correspondence to audio frames, shape (A).
         - acoustic_tokens: The Mimi acoustic codec tokens Q1-Q7, shape (A, 7).
         - audio_mask: Binary mask for the audio tokens, shape (A).

        Each column is right-padded so that the length of each row in a column
        is consistent (so that they can be easily collated).

        Returns:
            Dataset with the above columns.
        """
        ds = self.load_dataset("bpe")

        max_text_len = self.config.max_text_length
        max_bpe_len = self.config.max_bpe_length
        max_audio_frames = self.config.max_audio_frames

        def prepare_sample(example: dict) -> dict:
            # Text tokens for cross-attention context
            text_tokens = np.array(example["text_tokens"], dtype=np.int32)
            text_len = min(len(text_tokens), max_text_len)
            text_padded = np.zeros(max_text_len, dtype=np.int32)
            text_padded[:text_len] = text_tokens[:text_len]
            text_mask = np.zeros(max_text_len, dtype=np.bool_)
            text_mask[:text_len] = True

            # Semantic BPE tokens (T2S target) - prepend BOS, append EOS
            semantic_bpe_raw = np.array(example["semantic_bpe"], dtype=np.int32)
            bpe_spans_raw = np.array(example["bpe_spans"], dtype=np.int32)

            # Add BOS at start and EOS at end
            # BOS and EOS have span 0 (they don't map to any semantic tokens)
            semantic_bpe = np.concatenate([[BPE_BOS_TOKEN], semantic_bpe_raw, [BPE_EOS_TOKEN]])
            bpe_spans = np.concatenate([[0], bpe_spans_raw, [0]])

            # Truncate to max length (keeping BOS, may lose EOS if too long)
            bpe_len = min(len(semantic_bpe), max_bpe_len)
            bpe_padded = np.zeros(max_bpe_len, dtype=np.int32)
            bpe_padded[:bpe_len] = semantic_bpe[:bpe_len]
            spans_padded = np.zeros(max_bpe_len, dtype=np.int32)
            spans_padded[:bpe_len] = bpe_spans[:bpe_len]
            bpe_mask = np.zeros(max_bpe_len, dtype=np.bool_)
            bpe_mask[:bpe_len] = True

            # Audio codes: (T_audio, 8) where column 0 is Q0 (semantic), 1-7 are acoustic
            audio_codes_raw = np.asarray(example["audio_codes"])
            audio_codes = np.pad(audio_codes_raw, ((1, 1), (0, 0)), mode="empty")
            audio_codes[0] = ACOUSTIC_BOS_TOKEN
            audio_codes[-1] = ACOUSTIC_EOS_TOKEN
            semantic_tokens = audio_codes[:, 0]  # Q0: shape (T_audio,)
            acoustic_tokens = audio_codes[:, 1:8]  # Q1-Q7: shape (T_audio, 7)

            # Add padding tokens to the audio codes.
            audio_len = min(len(semantic_tokens), max_audio_frames)
            semantic_padded = np.pad(
                semantic_tokens[:max_audio_frames],
                ((0, max_audio_frames - audio_len),),
                mode="constant",
                constant_values=ACOUSTIC_PAD_TOKEN,
            )
            acoustic_padded = np.pad(
                acoustic_tokens[:max_audio_frames],
                ((0, max_audio_frames - audio_len), (0, 0)),
                mode="constant",
                constant_values=ACOUSTIC_PAD_TOKEN,
            )

            # Remove BOS token from the semantic tokens.
            semantic_padded = semantic_padded[1:]

            # Mask includes positions 0 to audio_len (inclusive) for EOS prediction
            audio_mask = np.zeros(max_audio_frames, dtype=np.bool_)
            audio_mask[: audio_len + 1] = True  # +1 for EOS position

            return {
                "text_tokens": text_padded,
                "text_mask": text_mask,
                "semantic_bpe": bpe_padded,
                "semantic_bpe_mask": bpe_mask,
                "bpe_spans": spans_padded,
                "dilated_semantic": semantic_padded,
                "audio_codes": acoustic_padded,
                "audio_mask": audio_mask,
            }

        # Map and remove columns that shouldn't be in final dataset (esp. 2D arrays)
        result = ds.map(prepare_sample, desc="Preparing training data")
        # Remove columns that aren't used in training (and would cause JAX dtype issues)
        cols_to_remove = [c for c in result.column_names if c not in prepare_sample.__code__.co_consts]
        cols_to_keep = [
            "text_tokens",
            "text_mask",
            "semantic_bpe",
            "semantic_bpe_mask",
            "bpe_spans",
            "dilated_semantic",
            "audio_codes",
            "audio_mask",
        ]
        cols_to_remove = [c for c in result.column_names if c not in cols_to_keep]
        if cols_to_remove:
            result = result.remove_columns(cols_to_remove)
        return result

    @xax.dataset_fn("bpe", dependencies=["tokenized"], use_hash=False)
    def bpe_dataset(self) -> Dataset:
        """Learns a BPE tokenizer on the semantic tokens and applies it to the dataset.

        The semantic tokens (Q0, first quantizer) capture high-level speech content.
        BPE compression groups frequent token patterns, reducing sequence length
        while preserving semantic information.

        Each semantic token (0-2047) is mapped to a unique unicode character so that
        BPE can properly merge adjacent tokens as atomic units.

        Returns dataset with columns:
        - text_tokens: shape (T_text,) - original text tokens
        - audio_codes: shape (T_audio, C) - original audio codes (all quantizers)
        - semantic_bpe: shape (T_bpe,) - BPE-encoded semantic tokens
        - bpe_spans: shape (T_bpe,) - number of original tokens each BPE token represents
        """
        ds = self.load_dataset("tokenized")

        # Map semantic tokens (0-2047) to unique unicode characters.
        # Use Private Use Area starting at U+E000 to avoid conflicts.
        base_char = 0xE000

        def tokens_to_chars(tokens: np.ndarray) -> str:
            """Convert token IDs to a string of unique unicode characters."""
            return "".join(chr(base_char + int(t)) for t in tokens)

        # Step 1: Extract semantic tokens and convert to character strings
        logger.info("Extracting semantic tokens for BPE training...")

        def get_semantic_chars(example: dict) -> dict:
            """Convert semantic tokens to unicode character string for BPE training."""
            audio_codes = np.asarray(example["audio_codes"])
            semantic_tokens = audio_codes[:, 0]
            semantic_chars = tokens_to_chars(semantic_tokens)
            return {"semantic_chars": semantic_chars}

        ds_with_chars = ds.map(get_semantic_chars, desc="Extracting semantic tokens")

        # Step 2: Train BPE tokenizer on character sequences
        logger.info("Training BPE tokenizer on semantic tokens...")

        # Create initial alphabet from all possible semantic tokens
        initial_alphabet = [chr(base_char + i) for i in range(xax.MIMI_CODEBOOK_SIZE)]

        # Initialize BPE tokenizer with no pre-tokenizer (treat entire string as one sequence)
        tokenizer = Tokenizer(models.BPE())
        # Don't use any pre-tokenizer - we want BPE to operate on the raw character sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])

        # Configure BPE trainer with the semantic token alphabet
        bpe_vocab_size = self.config.bpe_vocab_size
        trainer = trainers.BpeTrainer(
            vocab_size=bpe_vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )

        # Train on semantic token character strings
        def batch_iterator(batch_size: int = 1000) -> Iterator[list[str]]:
            for idx in range(0, len(ds_with_chars), batch_size):
                yield ds_with_chars[idx : idx + batch_size]["semantic_chars"]

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        logger.info("BPE tokenizer trained with vocab size %d", tokenizer.get_vocab_size())

        # Save tokenizer for later use
        tokenizer_path = self.dataset_cache_dir / "bpe_tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        logger.info("Saved BPE tokenizer to %s", tokenizer_path)

        # Step 3: Apply BPE encoding to all samples
        logger.info("Applying BPE encoding to dataset...")

        def apply_bpe(example: dict) -> dict:
            """Apply BPE encoding to semantic tokens and compute spans."""
            audio_codes = np.asarray(example["audio_codes"])
            semantic_tokens = audio_codes[:, 0]
            semantic_chars = tokens_to_chars(semantic_tokens)

            encoding = tokenizer.encode(semantic_chars)
            semantic_bpe = np.array(encoding.ids, dtype=np.int32)

            # Compute spans: how many original tokens each BPE token represents
            # Each token in the encoding corresponds to a substring of the original
            bpe_spans = np.array([len(token) for token in encoding.tokens], dtype=np.int32)

            return {"semantic_bpe": semantic_bpe, "bpe_spans": bpe_spans}

        ds_bpe = ds.map(
            apply_bpe,
            remove_columns=["semantic_chars"] if "semantic_chars" in ds.column_names else [],
            desc="Applying BPE",
        )

        # Log compression statistics
        total_original = sum(len(np.asarray(ex["audio_codes"])) for ex in ds_bpe)
        total_bpe = sum(len(ex["semantic_bpe"]) for ex in ds_bpe)
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
            batch_size=128,
            max_grad_norm=5.0,
            gradient_accumulation_steps=2,
            log_heavy_every_n_seconds=60,
            # max_steps=60 * 60,  # 1 hour
            step_kind="second",
        ),
    )
