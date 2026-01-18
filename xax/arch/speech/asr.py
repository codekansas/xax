"""Whisper ASR implementation for speech recognition.

This module implements OpenAI's Whisper model for automatic speech recognition.
The architecture consists of:
- Audio encoder: Conv layers + sinusoidal positional embeddings + transformer
- Text decoder: Token embeddings + learned positional embeddings + transformer
- Cross-attention from decoder to encoder

Supports loading pretrained weights from HuggingFace (openai/whisper-large-v3-turbo).
"""

import functools
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from xax.utils.jax import jit as xax_jit

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Please install huggingface_hub: pip install huggingface-hub") from e

try:
    from safetensors import safe_open
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Please install safetensors: pip install safetensors") from e

logger = logging.getLogger(__name__)


# Whisper audio processing constants
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_N_MELS = 128
WHISPER_CHUNK_LENGTH = 30  # seconds


@dataclass(frozen=True)
class WhisperConfig:
    """Configuration for the Whisper ASR model."""

    # Model dimensions
    d_model: int = field(default=1280)
    vocab_size: int = field(default=51866)
    num_mel_bins: int = field(default=128)

    # Encoder architecture
    encoder_layers: int = field(default=32)
    encoder_attention_heads: int = field(default=20)
    encoder_ffn_dim: int = field(default=5120)

    # Decoder architecture
    decoder_layers: int = field(default=4)
    decoder_attention_heads: int = field(default=20)
    decoder_ffn_dim: int = field(default=5120)

    # Sequence lengths
    max_source_positions: int = field(default=1500)
    max_target_positions: int = field(default=448)

    # Special tokens
    bos_token_id: int = field(default=50257)
    eos_token_id: int = field(default=50257)
    pad_token_id: int = field(default=50257)
    decoder_start_token_id: int = field(default=50258)

    # Generation
    suppress_tokens: tuple[int, ...] = field(default=(220, 50256))

    @property
    def encoder_head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    @property
    def decoder_head_dim(self) -> int:
        return self.d_model // self.decoder_attention_heads


def get_sinusoidal_embeddings(max_len: int, d_model: int) -> Array:
    """Generate sinusoidal positional embeddings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        Positional embeddings of shape (max_len, d_model)
    """
    positions = jnp.arange(max_len)[:, None]
    dims = jnp.arange(0, d_model, 2)[None, :]

    # Compute angles: pos / (10000 ^ (2i / d_model))
    angles = positions / jnp.power(10000.0, dims / d_model)

    # Interleave sin and cos
    embeddings = jnp.zeros((max_len, d_model))
    embeddings = embeddings.at[:, 0::2].set(jnp.sin(angles))
    embeddings = embeddings.at[:, 1::2].set(jnp.cos(angles))

    return embeddings


class WhisperConv1d(eqx.Module):
    """1D convolution for Whisper encoder input processing."""

    weight_oik: Array
    bias_o: Array
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        key: PRNGKeyArray,
        stride: int = 1,
        padding: int = 1,
    ) -> "WhisperConv1d":
        fan_in = in_channels * kernel_size
        std = 1.0 / math.sqrt(fan_in)

        k1, k2 = jax.random.split(key)
        weight_oik = jax.random.uniform(
            k1,
            (out_channels, in_channels, kernel_size),
            minval=-std,
            maxval=std,
        )
        bias_o = jax.random.uniform(k2, (out_channels,), minval=-std, maxval=std)

        return cls(
            weight_oik=weight_oik,
            bias_o=bias_o,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, x_ct: Array) -> Array:
        """Apply convolution.

        Args:
            x_ct: Input tensor of shape (in_channels, time)

        Returns:
            Output tensor of shape (out_channels, time')
        """
        chex.assert_rank(x_ct, 2)

        # Apply padding
        if self.padding > 0:
            x_ct = jnp.pad(x_ct, ((0, 0), (self.padding, self.padding)), mode="constant")

        # Add batch dimension for conv
        x_1ct = x_ct[None, :, :]

        out_1ot = jax.lax.conv_general_dilated(
            x_1ct,
            self.weight_oik,
            window_strides=(self.stride,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )

        out_ot = out_1ot[0]

        if self.bias_o is not None:
            out_ot = out_ot + self.bias_o[:, None]

        return out_ot


class WhisperMLP(eqx.Module):
    """Feed-forward MLP for transformer layers."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    @classmethod
    def build(
        cls,
        d_model: int,
        ffn_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "WhisperMLP":
        k1, k2 = jax.random.split(key)
        fc1 = eqx.nn.Linear(d_model, ffn_dim, use_bias=True, key=k1)
        fc2 = eqx.nn.Linear(ffn_dim, d_model, use_bias=True, key=k2)
        return cls(fc1=fc1, fc2=fc2)

    def __call__(self, x_d: Array) -> Array:
        x_d = self.fc1(x_d)
        x_d = jax.nn.gelu(x_d)
        x_d = self.fc2(x_d)
        return x_d


class WhisperAttention(eqx.Module):
    """Multi-head attention for Whisper.

    Supports both self-attention and cross-attention.
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        d_model: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ) -> "WhisperAttention":
        keys = jax.random.split(key, 4)
        head_dim = d_model // num_heads

        q_proj = eqx.nn.Linear(d_model, d_model, use_bias=True, key=keys[0])
        k_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[1])
        v_proj = eqx.nn.Linear(d_model, d_model, use_bias=True, key=keys[2])
        out_proj = eqx.nn.Linear(d_model, d_model, use_bias=True, key=keys[3])

        return cls(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
        )

    def __call__(
        self,
        x_td: Array,
        kv_td: Array | None = None,
        is_causal: bool = False,
    ) -> Array:
        """Apply multi-head attention.

        Args:
            x_td: Query input of shape (time, d_model)
            kv_td: Key/value input of shape (time_kv, d_model), None for self-attention
            is_causal: Whether to apply causal masking

        Returns:
            Output of shape (time, d_model)
        """
        if kv_td is None:
            kv_td = x_td

        tsz = x_td.shape[0]
        tsz_kv = kv_td.shape[0]

        # Project Q, K, V
        q_td = jax.vmap(self.q_proj)(x_td)
        k_td = jax.vmap(self.k_proj)(kv_td)
        v_td = jax.vmap(self.v_proj)(kv_td)

        # Reshape to (time, num_heads, head_dim)
        q_thd = q_td.reshape(tsz, self.num_heads, self.head_dim)
        k_thd = k_td.reshape(tsz_kv, self.num_heads, self.head_dim)
        v_thd = v_td.reshape(tsz_kv, self.num_heads, self.head_dim)

        # Compute attention
        ctx_thd = jax.nn.dot_product_attention(
            q_thd,
            k_thd,
            v_thd,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Combine heads
        ctx_td = ctx_thd.reshape(tsz, -1)

        # Output projection
        out_td = jax.vmap(self.out_proj)(ctx_td)

        return out_td


class WhisperEncoderLayer(eqx.Module):
    """Single encoder transformer layer."""

    self_attn: WhisperAttention
    self_attn_layer_norm: eqx.nn.LayerNorm
    mlp: WhisperMLP
    final_layer_norm: eqx.nn.LayerNorm

    @classmethod
    def build(
        cls,
        config: WhisperConfig,
        *,
        key: PRNGKeyArray,
    ) -> "WhisperEncoderLayer":
        k1, k2 = jax.random.split(key)

        self_attn = WhisperAttention.build(
            config.d_model,
            config.encoder_attention_heads,
            key=k1,
        )
        mlp = WhisperMLP.build(
            config.d_model,
            config.encoder_ffn_dim,
            key=k2,
        )
        self_attn_layer_norm = eqx.nn.LayerNorm(config.d_model)
        final_layer_norm = eqx.nn.LayerNorm(config.d_model)

        return cls(
            self_attn=self_attn,
            self_attn_layer_norm=self_attn_layer_norm,
            mlp=mlp,
            final_layer_norm=final_layer_norm,
        )

    def __call__(self, x_td: Array) -> Array:
        """Apply encoder layer.

        Args:
            x_td: Input of shape (time, d_model)

        Returns:
            Output of shape (time, d_model)
        """
        # Self-attention with pre-norm
        residual_td = x_td
        x_td = jax.vmap(self.self_attn_layer_norm)(x_td)
        x_td = self.self_attn(x_td, is_causal=False)
        x_td = residual_td + x_td

        # MLP with pre-norm
        residual_td = x_td
        x_td = jax.vmap(self.final_layer_norm)(x_td)
        x_td = jax.vmap(self.mlp)(x_td)
        x_td = residual_td + x_td

        return x_td


class WhisperEncoder(eqx.Module):
    """Whisper audio encoder.

    Processes mel spectrograms through conv layers and transformer.
    """

    conv1: WhisperConv1d
    conv2: WhisperConv1d
    embed_positions: Array  # Sinusoidal positional embeddings (frozen)
    layers: tuple[WhisperEncoderLayer, ...]
    layer_norm: eqx.nn.LayerNorm
    config: WhisperConfig = eqx.field(static=True)

    @classmethod
    def build(cls, config: WhisperConfig, *, key: PRNGKeyArray) -> "WhisperEncoder":
        keys = jax.random.split(key, config.encoder_layers + 2)

        # Two conv layers: first with stride 1, second with stride 2
        conv1 = WhisperConv1d.build(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[0],
        )
        conv2 = WhisperConv1d.build(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
            key=keys[1],
        )

        # Sinusoidal positional embeddings
        embed_positions = get_sinusoidal_embeddings(config.max_source_positions, config.d_model)

        # Transformer layers
        layers = tuple(WhisperEncoderLayer.build(config, key=keys[idx + 2]) for idx in range(config.encoder_layers))

        layer_norm = eqx.nn.LayerNorm(config.d_model)

        return cls(
            conv1=conv1,
            conv2=conv2,
            embed_positions=embed_positions,
            layers=layers,
            layer_norm=layer_norm,
            config=config,
        )

    def __call__(self, mel_mt: Array) -> Array:
        """Encode mel spectrogram.

        Args:
            mel_mt: Mel spectrogram of shape (n_mels, time)

        Returns:
            Encoded features of shape (time', d_model)
        """
        # Apply conv layers with GELU activation
        x_dt = self.conv1(mel_mt)
        x_dt = jax.nn.gelu(x_dt)
        x_dt = self.conv2(x_dt)
        x_dt = jax.nn.gelu(x_dt)

        # Transpose to (time, d_model)
        x_td = x_dt.T

        # Truncate to max_source_positions if needed
        tsz = x_td.shape[0]
        if tsz > self.config.max_source_positions:
            x_td = x_td[: self.config.max_source_positions]
            tsz = self.config.max_source_positions

        # Add sinusoidal positional embeddings
        positions = self.embed_positions[:tsz]
        x_td = x_td + positions

        # Apply transformer layers
        for layer in self.layers:
            x_td = layer(x_td)

        # Final layer norm
        x_td = jax.vmap(self.layer_norm)(x_td)

        return x_td


class WhisperDecoderLayer(eqx.Module):
    """Single decoder transformer layer with cross-attention."""

    self_attn: WhisperAttention
    self_attn_layer_norm: eqx.nn.LayerNorm
    encoder_attn: WhisperAttention
    encoder_attn_layer_norm: eqx.nn.LayerNorm
    mlp: WhisperMLP
    final_layer_norm: eqx.nn.LayerNorm

    @classmethod
    def build(
        cls,
        config: WhisperConfig,
        *,
        key: PRNGKeyArray,
    ) -> "WhisperDecoderLayer":
        k1, k2, k3 = jax.random.split(key, 3)

        self_attn = WhisperAttention.build(
            config.d_model,
            config.decoder_attention_heads,
            key=k1,
        )
        encoder_attn = WhisperAttention.build(
            config.d_model,
            config.decoder_attention_heads,
            key=k2,
        )
        mlp = WhisperMLP.build(
            config.d_model,
            config.decoder_ffn_dim,
            key=k3,
        )

        self_attn_layer_norm = eqx.nn.LayerNorm(config.d_model)
        encoder_attn_layer_norm = eqx.nn.LayerNorm(config.d_model)
        final_layer_norm = eqx.nn.LayerNorm(config.d_model)

        return cls(
            self_attn=self_attn,
            self_attn_layer_norm=self_attn_layer_norm,
            encoder_attn=encoder_attn,
            encoder_attn_layer_norm=encoder_attn_layer_norm,
            mlp=mlp,
            final_layer_norm=final_layer_norm,
        )

    def __call__(self, x_td: Array, encoder_out_td: Array) -> Array:
        """Apply decoder layer.

        Args:
            x_td: Decoder input of shape (time, d_model)
            encoder_out_td: Encoder output of shape (time_enc, d_model)

        Returns:
            Output of shape (time, d_model)
        """
        # Causal self-attention with pre-norm
        residual_td = x_td
        x_td = jax.vmap(self.self_attn_layer_norm)(x_td)
        x_td = self.self_attn(x_td, is_causal=True)
        x_td = residual_td + x_td

        # Cross-attention with pre-norm
        residual_td = x_td
        x_td = jax.vmap(self.encoder_attn_layer_norm)(x_td)
        x_td = self.encoder_attn(x_td, kv_td=encoder_out_td, is_causal=False)
        x_td = residual_td + x_td

        # MLP with pre-norm
        residual_td = x_td
        x_td = jax.vmap(self.final_layer_norm)(x_td)
        x_td = jax.vmap(self.mlp)(x_td)
        x_td = residual_td + x_td

        return x_td


class WhisperDecoder(eqx.Module):
    """Whisper text decoder.

    Generates text tokens from encoder features.
    """

    embed_tokens: eqx.nn.Embedding
    embed_positions: eqx.nn.Embedding  # Learned positional embeddings
    layers: tuple[WhisperDecoderLayer, ...]
    layer_norm: eqx.nn.LayerNorm
    config: WhisperConfig = eqx.field(static=True)

    @classmethod
    def build(cls, config: WhisperConfig, *, key: PRNGKeyArray) -> "WhisperDecoder":
        keys = jax.random.split(key, config.decoder_layers + 2)

        # Token embedding
        embed_tokens = eqx.nn.Embedding(
            config.vocab_size,
            config.d_model,
            key=keys[0],
        )

        # Learned positional embedding
        embed_positions = eqx.nn.Embedding(
            config.max_target_positions,
            config.d_model,
            key=keys[1],
        )

        # Transformer layers
        layers = tuple(WhisperDecoderLayer.build(config, key=keys[idx + 2]) for idx in range(config.decoder_layers))

        layer_norm = eqx.nn.LayerNorm(config.d_model)

        return cls(
            embed_tokens=embed_tokens,
            embed_positions=embed_positions,
            layers=layers,
            layer_norm=layer_norm,
            config=config,
        )

    def __call__(self, token_ids_t: Array, encoder_out_td: Array) -> Array:
        """Decode tokens using encoder features.

        Args:
            token_ids_t: Token IDs of shape (time,)
            encoder_out_td: Encoder output of shape (time_enc, d_model)

        Returns:
            Hidden states of shape (time, d_model)
        """
        tsz = token_ids_t.shape[0]

        # Token embeddings
        x_td = jax.vmap(self.embed_tokens)(token_ids_t)

        # Add learned positional embeddings
        position_ids = jnp.arange(tsz)
        positions_td = jax.vmap(self.embed_positions)(position_ids)
        x_td = x_td + positions_td

        # Apply transformer layers
        for layer in self.layers:
            x_td = layer(x_td, encoder_out_td)

        # Final layer norm
        x_td = jax.vmap(self.layer_norm)(x_td)

        return x_td


class WhisperModel(eqx.Module):
    """Complete Whisper ASR model.

    Encoder-decoder transformer for speech recognition.
    """

    encoder: WhisperEncoder
    decoder: WhisperDecoder
    proj_out: eqx.nn.Linear  # Output projection to vocab
    config: WhisperConfig = eqx.field(static=True)

    @classmethod
    def build(cls, config: WhisperConfig, *, key: PRNGKeyArray) -> "WhisperModel":
        k1, k2, k3 = jax.random.split(key, 3)

        encoder = WhisperEncoder.build(config, key=k1)
        decoder = WhisperDecoder.build(config, key=k2)

        # Output projection (typically tied to embedding weights)
        proj_out = eqx.nn.Linear(config.d_model, config.vocab_size, use_bias=False, key=k3)

        return cls(
            encoder=encoder,
            decoder=decoder,
            proj_out=proj_out,
            config=config,
        )

    def encode(self, mel_mt: Array) -> Array:
        """Encode mel spectrogram.

        Args:
            mel_mt: Mel spectrogram of shape (n_mels, time)

        Returns:
            Encoded features of shape (time', d_model)
        """
        return self.encoder(mel_mt)

    def decode(self, token_ids_t: Array, encoder_out_td: Array) -> Array:
        """Decode tokens to logits.

        Args:
            token_ids_t: Token IDs of shape (time,)
            encoder_out_td: Encoder output of shape (time_enc, d_model)

        Returns:
            Logits of shape (time, vocab_size)
        """
        hidden_td = self.decoder(token_ids_t, encoder_out_td)
        logits_tv = jax.vmap(self.proj_out)(hidden_td)
        return logits_tv

    def __call__(self, mel_mt: Array, token_ids_t: Array) -> Array:
        """Forward pass for training.

        Args:
            mel_mt: Mel spectrogram of shape (n_mels, time)
            token_ids_t: Token IDs of shape (time,)

        Returns:
            Logits of shape (time, vocab_size)
        """
        encoder_out_td = self.encode(mel_mt)
        return self.decode(token_ids_t, encoder_out_td)


@functools.lru_cache(maxsize=1)
def mel_filters(
    n_mels: int = WHISPER_N_MELS,
    n_fft: int = WHISPER_N_FFT,
    sample_rate: int = WHISPER_SAMPLE_RATE,
) -> Array:
    """Create mel filterbank matching HuggingFace Whisper implementation.

    Uses Slaney-style mel scale and normalization with max frequency of 8000 Hz.

    Returns filterbank of shape (n_mels, n_fft // 2 + 1)
    """
    try:
        # Use HuggingFace's mel_filter_bank if available for exact match
        from transformers.audio_utils import mel_filter_bank  # noqa: PLC0415

        filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=n_mels,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        return jnp.array(filters.T, dtype=jnp.float32)  # Transpose to (n_mels, n_freqs)
    except ImportError:
        pass

    # Fallback to custom implementation
    low_freq = 0.0
    high_freq = 8000.0

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)

    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    filterbank = np.zeros((n_mels, n_freqs))

    for idx in range(n_mels):
        left = hz_points[idx]
        center = hz_points[idx + 1]
        right = hz_points[idx + 2]

        for j, freq in enumerate(fft_freqs):
            if left <= freq < center:
                filterbank[idx, j] = (freq - left) / (center - left)
            elif center <= freq < right:
                filterbank[idx, j] = (right - freq) / (right - center)

    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, None]

    return jnp.array(filterbank, dtype=jnp.float32)


def log_mel_spectrogram(
    audio_t: Array,
    n_mels: int = WHISPER_N_MELS,
    n_fft: int = WHISPER_N_FFT,
    hop_length: int = WHISPER_HOP_LENGTH,
    padding: int = 0,
) -> Array:
    """Compute log mel spectrogram matching HuggingFace Whisper.

    Args:
        audio_t: Audio waveform of shape (time,)
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length
        padding: Padding to add

    Returns:
        Log mel spectrogram of shape (n_mels, time')
    """
    # Pad audio
    if padding > 0:
        audio_t = jnp.pad(audio_t, (0, padding), mode="constant")

    # Create Hann window (periodic, like torch.hann_window with periodic=True)
    window = jnp.hanning(n_fft + 1)[:-1]

    # Pad audio for STFT with reflection padding
    pad_len = n_fft // 2
    audio_padded = jnp.pad(audio_t, (pad_len, pad_len), mode="reflect")

    # Compute number of frames
    n_samples = audio_padded.shape[0]
    n_frames = 1 + (n_samples - n_fft) // hop_length

    # Extract frames and apply window
    indices = jnp.arange(n_fft)[None, :] + jnp.arange(n_frames)[:, None] * hop_length
    frames_tf = audio_padded[indices] * window

    # Compute FFT
    fft_tf = jnp.fft.rfft(frames_tf, n=n_fft)
    magnitudes_tf = jnp.abs(fft_tf) ** 2

    # Discard last frame (HuggingFace does this)
    magnitudes_tf = magnitudes_tf[:-1]

    # Apply mel filterbank
    filters_mf = mel_filters(n_mels, n_fft)
    mel_tf = magnitudes_tf @ filters_mf.T  # (time, n_mels)

    # Convert to log scale with clamping
    log_mel_tf = jnp.log10(jnp.maximum(mel_tf, 1e-10))

    # Normalize (Whisper-style): clip to max - 8, then scale
    log_mel_tf = jnp.maximum(log_mel_tf, log_mel_tf.max() - 8.0)
    log_mel_tf = (log_mel_tf + 4.0) / 4.0

    return log_mel_tf.T  # (n_mels, time)


@functools.lru_cache(maxsize=16)
def download_whisper_repo(repo_id: str = "openai/whisper-large-v3-turbo") -> Path:
    """Download Whisper model from HuggingFace Hub."""
    return Path(snapshot_download(repo_id=repo_id))


def load_whisper_config(repo_id: str = "openai/whisper-large-v3-turbo") -> WhisperConfig:
    """Load Whisper configuration from HuggingFace Hub."""
    path = download_whisper_repo(repo_id)
    config_path = path / "config.json"

    with open(config_path, "r") as f:
        cfg = json.load(f)

    return WhisperConfig(
        d_model=cfg.get("d_model", 1280),
        vocab_size=cfg.get("vocab_size", 51866),
        num_mel_bins=cfg.get("num_mel_bins", 128),
        encoder_layers=cfg.get("encoder_layers", 32),
        encoder_attention_heads=cfg.get("encoder_attention_heads", 20),
        encoder_ffn_dim=cfg.get("encoder_ffn_dim", 5120),
        decoder_layers=cfg.get("decoder_layers", 4),
        decoder_attention_heads=cfg.get("decoder_attention_heads", 20),
        decoder_ffn_dim=cfg.get("decoder_ffn_dim", 5120),
        max_source_positions=cfg.get("max_source_positions", 1500),
        max_target_positions=cfg.get("max_target_positions", 448),
        bos_token_id=cfg.get("bos_token_id", 50257),
        eos_token_id=cfg.get("eos_token_id", 50257),
        pad_token_id=cfg.get("pad_token_id", 50257),
        decoder_start_token_id=cfg.get("decoder_start_token_id", 50258),
        suppress_tokens=tuple(cfg.get("begin_suppress_tokens", [220, 50256])),
    )


def build_pretrained_whisper(
    repo_id: str = "openai/whisper-large-v3-turbo",
    dtype: jnp.dtype | None = None,
) -> WhisperModel:
    """Load pretrained Whisper model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        dtype: Optional dtype for model parameters

    Returns:
        Loaded WhisperModel with pretrained weights
    """
    if dtype is None:
        dtype = jnp.float32

    config = load_whisper_config(repo_id)
    path = download_whisper_repo(repo_id)

    # Build model
    model = WhisperModel.build(config, key=jax.random.key(0))

    # Load weights from safetensors
    safetensor_files = list(path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")

    state: dict[str, np.ndarray] = {}
    for sf in safetensor_files:
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)

    # Map weights to model
    model = _load_weights_into_whisper(model, state, dtype)

    return model


def _load_weights_into_whisper(
    model: WhisperModel,
    state: dict[str, np.ndarray],
    dtype: jnp.dtype,
) -> WhisperModel:
    """Map HuggingFace weights into Whisper model structure."""

    def get_weight(key: str) -> jnp.ndarray | None:
        if key in state:
            return jnp.array(state[key], dtype=dtype)
        return None

    loaded_count = 0
    total_keys = len(state)

    def set_weight(get_leaf: Callable[[WhisperModel], Array], weight: jnp.ndarray) -> None:
        nonlocal model, loaded_count
        model = eqx.tree_at(get_leaf, model, weight)
        loaded_count += 1

    # Load encoder conv layers
    w = get_weight("model.encoder.conv1.weight")
    b = get_weight("model.encoder.conv1.bias")
    if w is not None:
        set_weight(lambda m: m.encoder.conv1.weight_oik, w)
    if b is not None:
        set_weight(lambda m: m.encoder.conv1.bias_o, b)

    w = get_weight("model.encoder.conv2.weight")
    b = get_weight("model.encoder.conv2.bias")
    if w is not None:
        set_weight(lambda m: m.encoder.conv2.weight_oik, w)
    if b is not None:
        set_weight(lambda m: m.encoder.conv2.bias_o, b)

    # Load encoder positional embeddings
    w = get_weight("model.encoder.embed_positions.weight")
    if w is not None:
        set_weight(lambda m: m.encoder.embed_positions, w)

    # Load encoder transformer layers
    for layer_idx in range(model.config.encoder_layers):
        prefix = f"model.encoder.layers.{layer_idx}"

        # Self-attention
        attn_projs = ["q_proj", "k_proj", "v_proj", "out_proj"]
        for proj in attn_projs:
            w = get_weight(f"{prefix}.self_attn.{proj}.weight")
            b = get_weight(f"{prefix}.self_attn.{proj}.bias")
            if w is not None:
                set_weight(
                    lambda m, li=layer_idx, p=proj: getattr(m.encoder.layers[li].self_attn, p).weight,
                    w,
                )
            if b is not None:
                set_weight(
                    lambda m, li=layer_idx, p=proj: getattr(m.encoder.layers[li].self_attn, p).bias,
                    b,
                )

        # Layer norms
        w = get_weight(f"{prefix}.self_attn_layer_norm.weight")
        b = get_weight(f"{prefix}.self_attn_layer_norm.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].self_attn_layer_norm.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].self_attn_layer_norm.bias, b)

        w = get_weight(f"{prefix}.final_layer_norm.weight")
        b = get_weight(f"{prefix}.final_layer_norm.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].final_layer_norm.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].final_layer_norm.bias, b)

        # MLP
        w = get_weight(f"{prefix}.fc1.weight")
        b = get_weight(f"{prefix}.fc1.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].mlp.fc1.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].mlp.fc1.bias, b)

        w = get_weight(f"{prefix}.fc2.weight")
        b = get_weight(f"{prefix}.fc2.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].mlp.fc2.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.encoder.layers[li].mlp.fc2.bias, b)

    # Encoder final layer norm
    w = get_weight("model.encoder.layer_norm.weight")
    b = get_weight("model.encoder.layer_norm.bias")
    if w is not None:
        set_weight(lambda m: m.encoder.layer_norm.weight, w)
    if b is not None:
        set_weight(lambda m: m.encoder.layer_norm.bias, b)

    # Load decoder embeddings
    w = get_weight("model.decoder.embed_tokens.weight")
    if w is not None:
        set_weight(lambda m: m.decoder.embed_tokens.weight, w)

    w = get_weight("model.decoder.embed_positions.weight")
    if w is not None:
        set_weight(lambda m: m.decoder.embed_positions.weight, w)

    # Load decoder transformer layers
    for layer_idx in range(model.config.decoder_layers):
        prefix = f"model.decoder.layers.{layer_idx}"

        # Self-attention
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            w = get_weight(f"{prefix}.self_attn.{proj}.weight")
            b = get_weight(f"{prefix}.self_attn.{proj}.bias")
            if w is not None:
                set_weight(
                    lambda m, li=layer_idx, p=proj: getattr(m.decoder.layers[li].self_attn, p).weight,
                    w,
                )
            if b is not None:
                set_weight(
                    lambda m, li=layer_idx, p=proj: getattr(m.decoder.layers[li].self_attn, p).bias,
                    b,
                )

        # Cross-attention
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            w = get_weight(f"{prefix}.encoder_attn.{proj}.weight")
            b = get_weight(f"{prefix}.encoder_attn.{proj}.bias")
            if w is not None:
                set_weight(
                    lambda m, li=layer_idx, p=proj: getattr(m.decoder.layers[li].encoder_attn, p).weight,
                    w,
                )
            if b is not None:
                set_weight(
                    lambda m, li=layer_idx, p=proj: getattr(m.decoder.layers[li].encoder_attn, p).bias,
                    b,
                )

        # Layer norms
        w = get_weight(f"{prefix}.self_attn_layer_norm.weight")
        b = get_weight(f"{prefix}.self_attn_layer_norm.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].self_attn_layer_norm.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].self_attn_layer_norm.bias, b)

        w = get_weight(f"{prefix}.encoder_attn_layer_norm.weight")
        b = get_weight(f"{prefix}.encoder_attn_layer_norm.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].encoder_attn_layer_norm.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].encoder_attn_layer_norm.bias, b)

        w = get_weight(f"{prefix}.final_layer_norm.weight")
        b = get_weight(f"{prefix}.final_layer_norm.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].final_layer_norm.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].final_layer_norm.bias, b)

        # MLP
        w = get_weight(f"{prefix}.fc1.weight")
        b = get_weight(f"{prefix}.fc1.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].mlp.fc1.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].mlp.fc1.bias, b)

        w = get_weight(f"{prefix}.fc2.weight")
        b = get_weight(f"{prefix}.fc2.bias")
        if w is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].mlp.fc2.weight, w)
        if b is not None:
            set_weight(lambda m, li=layer_idx: m.decoder.layers[li].mlp.fc2.bias, b)

    # Decoder final layer norm
    w = get_weight("model.decoder.layer_norm.weight")
    b = get_weight("model.decoder.layer_norm.bias")
    if w is not None:
        set_weight(lambda m: m.decoder.layer_norm.weight, w)
    if b is not None:
        set_weight(lambda m: m.decoder.layer_norm.bias, b)

    # Output projection (often tied to embedding weights)
    w = get_weight("proj_out.weight")
    if w is not None:
        set_weight(lambda m: m.proj_out.weight, w)
    else:
        # Try to use decoder embedding weights (tied weights)
        w = get_weight("model.decoder.embed_tokens.weight")
        if w is not None:
            set_weight(lambda m: m.proj_out.weight, w)

    logger.info("Loaded %d/%d weight tensors", loaded_count, total_keys)
    return model


@xax_jit(static_argnames=("eos_token_id", "max_tokens"))
def transcribe_with_whisper(
    model: WhisperModel,
    audio_t: Array,
    eos_token_id: int,
    max_tokens: int,
) -> tuple[Array, Array, Array]:
    """Transcribe audio using greedy decoding.

    Args:
        model: Whisper model
        audio_t: Audio waveform of shape (time,)
        eos_token_id: EOS token ID
        max_tokens: Maximum number of tokens to generate

    Returns:
        Tuple of (token_ids, encoder_output)
    """
    # Pad or trim audio to 30 seconds
    target_len = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH
    if audio_t.shape[0] < target_len:
        audio_t = jnp.pad(audio_t, (0, target_len - audio_t.shape[0]))
    else:
        audio_t = audio_t[:target_len]

    # Compute mel spectrogram
    mel_mt = log_mel_spectrogram(audio_t)

    # Encode
    encoder_out_td = model.encode(mel_mt)

    # Start tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    # These are Whisper's special tokens for English transcription
    sot_token = 50258  # <|startoftranscript|>
    lang_token = 50259  # <|en|>
    task_token = 50360  # <|transcribe|>
    notimestamps_token = 50364  # <|notimestamps|>

    tokens_t = jnp.array([sot_token, lang_token, task_token, notimestamps_token], dtype=jnp.int32)

    # Greedy decoding using lax.scan
    max_len = tokens_t.shape[0] + max_tokens
    tokens_full_t = jnp.full((max_len,), eos_token_id, dtype=tokens_t.dtype)
    tokens_full_t = tokens_full_t.at[: tokens_t.shape[0]].set(tokens_t)
    init_len_s = jnp.array(tokens_t.shape[0], dtype=jnp.int32)
    init_done_s = jnp.array(False)

    def decode_step(
        carry: tuple[Array, Array, Array],
        _: None,
    ) -> tuple[tuple[Array, Array, Array], None]:
        tokens_full_t, cur_len_s, done_s = carry
        logits_tv = model.decode(tokens_full_t, encoder_out_td)
        next_token_s = jnp.argmax(logits_tv[cur_len_s - 1]).astype(tokens_full_t.dtype)
        is_eos_s = next_token_s == eos_token_id
        should_write_s = ~done_s

        def write_token(tokens_full_t: Array) -> Array:
            return tokens_full_t.at[cur_len_s].set(next_token_s)

        tokens_full_t = jax.lax.cond(should_write_s, write_token, lambda t: t, tokens_full_t)
        new_len_s = jnp.minimum(cur_len_s + should_write_s.astype(cur_len_s.dtype), max_len)
        new_done_s = done_s | is_eos_s
        return (tokens_full_t, new_len_s, new_done_s), None

    (tokens_full_t, final_len_s, _), _ = jax.lax.scan(
        decode_step,
        (tokens_full_t, init_len_s, init_done_s),
        length=max_tokens,
    )

    return tokens_full_t, final_len_s, encoder_out_td


def main() -> None:
    """CLI for Whisper ASR transcription."""
    import argparse  # noqa: PLC0415

    from xax.utils.logging import configure_logging  # noqa: PLC0415

    configure_logging()

    parser = argparse.ArgumentParser(description="Whisper ASR - transcribe audio to text")
    parser.add_argument("input", type=str, help="Input audio file (WAV, MP3, etc.)")
    parser.add_argument(
        "--repo",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace repo for pretrained model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=224,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()

    try:
        from transformers import WhisperTokenizerFast  # noqa: PLC0415
    except ImportError:
        logger.error("Please install transformers: pip install transformers")
        sys.exit(1)

    # Import audio libraries
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError:
        logger.error("Please install soundfile: pip install soundfile")
        sys.exit(1)

    # Load audio
    logger.info("Loading audio from %s", args.input)
    try:
        import soundfile as sf  # noqa: PLC0415

        audio, sr = sf.read(args.input)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Resample if needed
        if sr != WHISPER_SAMPLE_RATE:
            ratio = WHISPER_SAMPLE_RATE / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            )
    except ImportError as e:
        logger.error("Please install librosa or soundfile: pip install librosa soundfile")
        raise e

    audio_t = jnp.array(audio, dtype=jnp.float32)
    logger.info("Audio loaded: %.2f seconds", len(audio) / WHISPER_SAMPLE_RATE)

    # Load model
    logger.info("Loading Whisper model from %s", args.repo)
    config = load_whisper_config(args.repo)
    model = build_pretrained_whisper(args.repo)
    logger.info("Model loaded: %d encoder layers, %d decoder layers", config.encoder_layers, config.decoder_layers)

    # Load tokenizer
    logger.info("Loading tokenizer")
    path = download_whisper_repo(args.repo)
    tokenizer: WhisperTokenizerFast = WhisperTokenizerFast.from_pretrained(str(path))

    # Transcribe
    logger.info("Transcribing...")
    tokens, token_len, _ = transcribe_with_whisper(
        model,
        audio_t,
        config.eos_token_id,
        max_tokens=args.max_tokens,
    )

    # Decode tokens
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    print(f"\nTranscription:\n{text}")


if __name__ == "__main__":
    main()
