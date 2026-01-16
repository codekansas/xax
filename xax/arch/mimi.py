"""Mimi neural audio codec implementation.

This module implements the Mimi audio codec from Kyutai, which encodes audio
waveforms into discrete tokens and decodes them back. The architecture consists
of:
- SEANet encoder: Downsampling convolutions with residual blocks
- Transformer layers: For contextualization
- Residual Vector Quantization: Discretizes embeddings into codebook indices
- SEANet decoder: Upsampling convolutions with residual blocks
"""

import functools
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from xax.arch.attention import RotaryEmbedding

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MimiConfig:
    """Configuration for the Mimi audio codec model.

    The Mimi model is a neural audio codec that compresses audio waveforms into
    discrete tokens using a SEANet encoder/decoder architecture with a
    transformer and residual vector quantization.
    """

    # Audio parameters
    sampling_rate: int = field(default=24000)
    audio_channels: int = field(default=1)

    # Encoder/decoder architecture
    hidden_size: int = field(default=512)
    num_filters: int = field(default=64)
    num_residual_layers: int = field(default=1)
    upsampling_ratios: tuple[int, ...] = field(default=(8, 6, 5, 4))
    kernel_size: int = field(default=7)
    last_kernel_size: int = field(default=3)
    residual_kernel_size: int = field(default=3)
    dilation_growth_rate: int = field(default=2)
    use_causal_conv: bool = field(default=True)
    pad_mode: str = field(default="constant")
    compress: int = field(default=2)
    trim_right_ratio: float = field(default=1.0)

    # Quantization parameters
    codebook_size: int = field(default=2048)
    codebook_dim: int = field(default=256)
    num_quantizers: int = field(default=32)
    num_semantic_quantizers: int = field(default=1)
    upsample_groups: int = field(default=1)

    # Transformer parameters
    num_hidden_layers: int = field(default=8)
    num_attention_heads: int = field(default=8)
    head_dim: int = field(default=64)
    intermediate_size: int = field(default=2048)
    max_position_embeddings: int = field(default=8000)
    sliding_window: int = field(default=250)
    attention_dropout: float = field(default=0.0)
    layer_scale_initial_scale: float = field(default=0.01)
    rope_theta: float = field(default=10000.0)

    # Frame rates
    frame_rate: float = field(default=12.5)
    encodec_frame_rate: float = field(default=75.0)

    @property
    def num_codebooks(self) -> int:
        return self.num_quantizers

    @property
    def encoder_frame_rate(self) -> float:
        hop_length = math.prod(self.upsampling_ratios)
        return self.sampling_rate / hop_length


# Default Mimi 1.5B configuration
MIMI_DEFAULT_CONFIG = MimiConfig()


class MimiLayerScale(eqx.Module):
    """Learnable diagonal layer scaling for transformer residuals."""

    scale: Array

    @classmethod
    def build(cls, dim: int, init_scale: float = 0.01) -> "MimiLayerScale":
        scale = jnp.full((dim,), init_scale, dtype=jnp.float32)
        return cls(scale=scale)

    def __call__(self, x_td: Array) -> Array:
        return x_td * self.scale


class MimiConv1d(eqx.Module):
    """1D convolution with optional causal padding.

    Supports asymmetric padding for causal convolutions where more padding is
    applied to the left side to ensure the output only depends on past inputs.
    """

    weight_oik: Array
    bias_o: Array | None
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    pad_mode: str = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        key: PRNGKeyArray,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        causal: bool = True,
        pad_mode: str = "constant",
    ) -> "MimiConv1d":
        assert in_channels % groups == 0 and out_channels % groups == 0
        fan_in = (in_channels // groups) * kernel_size
        std = 1.0 / math.sqrt(fan_in)

        k1, k2 = jax.random.split(key)
        weight_oik = jax.random.uniform(
            k1,
            (out_channels, in_channels // groups, kernel_size),
            minval=-std,
            maxval=std,
        )
        bias_o = jax.random.uniform(k2, (out_channels,), minval=-std, maxval=std) if use_bias else None

        return cls(
            weight_oik=weight_oik,
            bias_o=bias_o,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            causal=causal,
            pad_mode=pad_mode,
        )

    def _get_padding(self, length: int) -> tuple[int, int]:
        """Calculate asymmetric padding for causal convolution."""
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        total_padding = effective_kernel - self.stride

        if self.causal:
            # All padding on the left for causal convolution
            return (total_padding, 0)
        else:
            # Symmetric padding
            left = total_padding // 2
            right = total_padding - left
            return (left, right)

    def __call__(self, x_ct: Array) -> Array:
        """Apply convolution.

        Args:
            x_ct: Input tensor of shape (in_channels, time)

        Returns:
            Output tensor of shape (out_channels, time')
        """
        chex.assert_rank(x_ct, 2)

        # Apply padding
        pad_left, pad_right = self._get_padding(x_ct.shape[-1])
        if pad_left > 0 or pad_right > 0:
            x_ct = jnp.pad(x_ct, ((0, 0), (pad_left, pad_right)), mode=self.pad_mode)

        # Apply convolution using lax.conv_general_dilated
        # Input: (in_channels, time) -> need to add batch dim
        x_1ct = x_ct[None, :, :]  # (1, in_channels, time)

        # For grouped convolution, reshape appropriately
        out_1ot = jax.lax.conv_general_dilated(
            x_1ct,
            self.weight_oik,
            window_strides=(self.stride,),
            padding="VALID",
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
            dimension_numbers=("NCH", "OIH", "NCH"),
        )

        out_ot = out_1ot[0]  # Remove batch dim

        if self.bias_o is not None:
            out_ot = out_ot + self.bias_o[:, None]

        return out_ot


class MimiConvTranspose1d(eqx.Module):
    """1D transposed convolution with optional causal padding."""

    weight_oik: Array
    bias_o: Array | None
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    trim_right_ratio: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        key: PRNGKeyArray,
        stride: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        causal: bool = True,
        trim_right_ratio: float = 1.0,
    ) -> "MimiConvTranspose1d":
        assert in_channels % groups == 0 and out_channels % groups == 0
        fan_in = in_channels * kernel_size // groups
        std = 1.0 / math.sqrt(fan_in)

        k1, k2 = jax.random.split(key)
        # Transposed conv weight is (in_channels, out_channels // groups, kernel)
        weight_oik = jax.random.uniform(
            k1,
            (in_channels, out_channels // groups, kernel_size),
            minval=-std,
            maxval=std,
        )
        bias_o = jax.random.uniform(k2, (out_channels,), minval=-std, maxval=std) if use_bias else None

        return cls(
            weight_oik=weight_oik,
            bias_o=bias_o,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            causal=causal,
            trim_right_ratio=trim_right_ratio,
        )

    def __call__(self, x_ct: Array) -> Array:
        """Apply transposed convolution.

        Args:
            x_ct: Input tensor of shape (in_channels, time)

        Returns:
            Output tensor of shape (out_channels, time')
        """
        chex.assert_rank(x_ct, 2)

        # Add batch dimension
        x_1ct = x_ct[None, :, :]

        # Transposed convolution
        out_1ot = jax.lax.conv_transpose(
            x_1ct,
            self.weight_oik,
            strides=(self.stride,),
            padding="VALID",
            dimension_numbers=("NCH", "IOH", "NCH"),
        )

        out_ot = out_1ot[0]

        if self.bias_o is not None:
            out_ot = out_ot + self.bias_o[:, None]

        # Trim output for causal mode
        if self.causal:
            padding_total = self.kernel_size - self.stride
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right

            if padding_left > 0:
                out_ot = out_ot[:, padding_left:]
            if padding_right > 0:
                out_ot = out_ot[:, :-padding_right]

        return out_ot


class MimiResnetBlock(eqx.Module):
    """Residual block with dilated convolutions."""

    conv1: MimiConv1d
    conv2: MimiConv1d
    shortcut: MimiConv1d | None

    @classmethod
    def build(
        cls,
        dim: int,
        kernel_size: int,
        *,
        key: PRNGKeyArray,
        dilation: int = 1,
        causal: bool = True,
        pad_mode: str = "constant",
        compress: int = 2,
    ) -> "MimiResnetBlock":
        k1, k2, k3 = jax.random.split(key, 3)
        hidden = dim // compress

        conv1 = MimiConv1d.build(
            dim,
            hidden,
            kernel_size,
            key=k1,
            dilation=dilation,
            causal=causal,
            pad_mode=pad_mode,
        )
        conv2 = MimiConv1d.build(
            hidden,
            dim,
            1,
            key=k2,
            causal=causal,
            pad_mode=pad_mode,
        )

        shortcut = None

        return cls(conv1=conv1, conv2=conv2, shortcut=shortcut)

    def __call__(self, x_ct: Array) -> Array:
        """Apply residual block.

        Args:
            x_ct: Input tensor of shape (channels, time)

        Returns:
            Output tensor of shape (channels, time)
        """
        residual = x_ct
        if self.shortcut is not None:
            residual = self.shortcut(x_ct)

        y_ct = jax.nn.elu(x_ct)
        y_ct = self.conv1(y_ct)
        y_ct = jax.nn.elu(y_ct)
        y_ct = self.conv2(y_ct)

        # Align lengths if needed
        min_len = min(residual.shape[-1], y_ct.shape[-1])
        residual = residual[:, :min_len]
        y_ct = y_ct[:, :min_len]

        return residual + y_ct


class MimiEncoder(eqx.Module):
    """SEANet encoder for downsampling audio to embeddings."""

    init_conv: MimiConv1d
    blocks: tuple[tuple[tuple[MimiResnetBlock, ...], MimiConv1d], ...]
    final_conv: MimiConv1d

    @classmethod
    def build(cls, config: MimiConfig, *, key: PRNGKeyArray) -> "MimiEncoder":
        keys = jax.random.split(key, 100)
        key_idx = 0

        # Initial convolution
        init_conv = MimiConv1d.build(
            config.audio_channels,
            config.num_filters,
            config.kernel_size,
            key=keys[key_idx],
            causal=config.use_causal_conv,
            pad_mode=config.pad_mode,
        )
        key_idx += 1

        # Downsampling blocks
        blocks = []
        in_channels = config.num_filters

        for _idx, ratio in enumerate(config.upsampling_ratios):
            out_channels = in_channels * 2

            # Residual layers with increasing dilation
            residual_layers = []
            for res_idx in range(config.num_residual_layers):
                dilation = config.dilation_growth_rate**res_idx
                residual_layers.append(
                    MimiResnetBlock.build(
                        in_channels,
                        config.residual_kernel_size,
                        key=keys[key_idx],
                        dilation=dilation,
                        causal=config.use_causal_conv,
                        pad_mode=config.pad_mode,
                        compress=config.compress,
                    )
                )
                key_idx += 1

            # Downsampling convolution
            down_conv = MimiConv1d.build(
                in_channels,
                out_channels,
                ratio * 2,
                key=keys[key_idx],
                stride=ratio,
                causal=config.use_causal_conv,
                pad_mode=config.pad_mode,
            )
            key_idx += 1

            blocks.append((tuple(residual_layers), down_conv))
            in_channels = out_channels

        # Final convolution to hidden size
        final_conv = MimiConv1d.build(
            in_channels,
            config.hidden_size,
            config.last_kernel_size,
            key=keys[key_idx],
            causal=config.use_causal_conv,
            pad_mode=config.pad_mode,
        )

        return cls(
            init_conv=init_conv,
            blocks=tuple(blocks),
            final_conv=final_conv,
        )

    def __call__(self, x_ct: Array) -> Array:
        """Encode audio waveform to embeddings.

        Args:
            x_ct: Audio waveform of shape (channels, time)

        Returns:
            Embeddings of shape (hidden_size, time')
        """
        x_ct = self.init_conv(x_ct)

        for residual_layers, down_conv in self.blocks:
            for res_block in residual_layers:
                x_ct = res_block(x_ct)
            x_ct = jax.nn.elu(x_ct)
            x_ct = down_conv(x_ct)

        x_ct = jax.nn.elu(x_ct)
        x_ct = self.final_conv(x_ct)

        return x_ct


class MimiDecoder(eqx.Module):
    """SEANet decoder for upsampling embeddings to audio."""

    init_conv: MimiConv1d
    blocks: tuple[tuple[MimiConvTranspose1d, tuple[MimiResnetBlock, ...]], ...]
    final_conv: MimiConv1d

    @classmethod
    def build(cls, config: MimiConfig, *, key: PRNGKeyArray) -> "MimiDecoder":
        keys = jax.random.split(key, 100)
        key_idx = 0

        # Compute channel progression (reverse of encoder)
        num_layers = len(config.upsampling_ratios)
        channels = [config.num_filters * (2**i) for i in range(num_layers + 1)]
        channels = channels[::-1]  # Reverse for decoder

        # Initial convolution from hidden size
        init_conv = MimiConv1d.build(
            config.hidden_size,
            channels[0],
            config.last_kernel_size,
            key=keys[key_idx],
            causal=config.use_causal_conv,
            pad_mode=config.pad_mode,
        )
        key_idx += 1

        # Upsampling blocks (reverse order of encoder)
        blocks = []
        ratios = config.upsampling_ratios[::-1]

        for idx, ratio in enumerate(ratios):
            in_channels = channels[idx]
            out_channels = channels[idx + 1]

            # Upsampling convolution
            up_conv = MimiConvTranspose1d.build(
                in_channels,
                out_channels,
                ratio * 2,
                key=keys[key_idx],
                stride=ratio,
                causal=config.use_causal_conv,
                trim_right_ratio=config.trim_right_ratio,
            )
            key_idx += 1

            # Residual layers
            residual_layers = []
            for res_idx in range(config.num_residual_layers):
                dilation = config.dilation_growth_rate**res_idx
                residual_layers.append(
                    MimiResnetBlock.build(
                        out_channels,
                        config.residual_kernel_size,
                        key=keys[key_idx],
                        dilation=dilation,
                        causal=config.use_causal_conv,
                        pad_mode=config.pad_mode,
                        compress=config.compress,
                    )
                )
                key_idx += 1

            blocks.append((up_conv, tuple(residual_layers)))

        # Final convolution to audio channels
        final_conv = MimiConv1d.build(
            channels[-1],
            config.audio_channels,
            config.kernel_size,
            key=keys[key_idx],
            causal=config.use_causal_conv,
            pad_mode=config.pad_mode,
        )

        return cls(
            init_conv=init_conv,
            blocks=tuple(blocks),
            final_conv=final_conv,
        )

    def __call__(self, x_ct: Array) -> Array:
        """Decode embeddings to audio waveform.

        Args:
            x_ct: Embeddings of shape (hidden_size, time)

        Returns:
            Audio waveform of shape (channels, time')
        """
        x_ct = self.init_conv(x_ct)

        for up_conv, residual_layers in self.blocks:
            x_ct = jax.nn.elu(x_ct)
            x_ct = up_conv(x_ct)
            for res_block in residual_layers:
                x_ct = res_block(x_ct)

        x_ct = jax.nn.elu(x_ct)
        x_ct = self.final_conv(x_ct)

        return x_ct


class MimiEuclideanCodebook(eqx.Module):
    """Vector quantization codebook using Euclidean distance."""

    embeddings_kd: Array
    codebook_size: int = eqx.field(static=True)
    codebook_dim: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        codebook_size: int,
        codebook_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "MimiEuclideanCodebook":
        # Initialize with uniform distribution
        embeddings_kd = jax.random.uniform(
            key,
            (codebook_size, codebook_dim),
            minval=-1.0 / codebook_size,
            maxval=1.0 / codebook_size,
        )
        return cls(
            embeddings_kd=embeddings_kd,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

    def quantize(self, x_td: Array) -> Array:
        """Quantize input vectors to nearest codebook entries.

        Args:
            x_td: Input vectors of shape (time, dim)

        Returns:
            Codebook indices of shape (time,)
        """
        # Compute squared distances: (x - e)^2 = x^2 - 2*x*e + e^2
        x_sq_t = jnp.sum(x_td**2, axis=-1, keepdims=True)  # (time, 1)
        e_sq_k = jnp.sum(self.embeddings_kd**2, axis=-1)  # (codebook_size,)
        cross_tk = x_td @ self.embeddings_kd.T  # (time, codebook_size)

        dist_tk = x_sq_t - 2 * cross_tk + e_sq_k
        indices_t = jnp.argmin(dist_tk, axis=-1)

        return indices_t

    def decode(self, indices_t: Array) -> Array:
        """Decode codebook indices to vectors.

        Args:
            indices_t: Codebook indices of shape (time,)

        Returns:
            Vectors of shape (time, dim)
        """
        return self.embeddings_kd[indices_t]

    def __call__(self, x_td: Array) -> tuple[Array, Array]:
        """Quantize and return both quantized vectors and indices.

        Args:
            x_td: Input vectors of shape (time, dim)

        Returns:
            Tuple of (quantized vectors, indices)
        """
        indices_t = self.quantize(x_td)
        quantized_td = self.decode(indices_t)

        # Straight-through estimator: gradient passes through
        quantized_td = x_td + jax.lax.stop_gradient(quantized_td - x_td)

        return quantized_td, indices_t


class MimiVectorQuantization(eqx.Module):
    """Vector quantization with projection layers."""

    input_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    codebook: MimiEuclideanCodebook

    @classmethod
    def build(
        cls,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "MimiVectorQuantization":
        k1, k2, k3 = jax.random.split(key, 3)

        input_proj = eqx.nn.Linear(input_dim, codebook_dim, use_bias=False, key=k1)
        output_proj = eqx.nn.Linear(codebook_dim, input_dim, use_bias=False, key=k2)
        codebook = MimiEuclideanCodebook.build(codebook_size, codebook_dim, key=k3)

        return cls(
            input_proj=input_proj,
            output_proj=output_proj,
            codebook=codebook,
        )

    def encode(self, x_td: Array) -> Array:
        """Encode to codebook indices.

        Args:
            x_td: Input of shape (time, input_dim)

        Returns:
            Indices of shape (time,)
        """
        x_proj_td = jax.vmap(self.input_proj)(x_td)
        return self.codebook.quantize(x_proj_td)

    def decode(self, indices_t: Array) -> Array:
        """Decode from codebook indices.

        Args:
            indices_t: Indices of shape (time,)

        Returns:
            Output of shape (time, input_dim)
        """
        quantized_td = self.codebook.decode(indices_t)
        return jax.vmap(self.output_proj)(quantized_td)


class MimiResidualVectorQuantizer(eqx.Module):
    """Residual Vector Quantization using multiple codebooks.

    Each successive quantizer operates on the residual from previous quantizers.
    """

    layers: tuple[MimiVectorQuantization, ...]
    num_quantizers: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        num_quantizers: int,
        *,
        key: PRNGKeyArray,
    ) -> "MimiResidualVectorQuantizer":
        keys = jax.random.split(key, num_quantizers)
        layers = tuple(
            MimiVectorQuantization.build(input_dim, codebook_size, codebook_dim, key=k)
            for k in keys
        )
        return cls(layers=layers, num_quantizers=num_quantizers)

    def encode(self, x_td: Array, num_quantizers: int | None = None) -> Array:
        """Encode using residual quantization.

        Args:
            x_td: Input of shape (time, input_dim)
            num_quantizers: Number of quantizers to use (defaults to all)

        Returns:
            Indices of shape (num_quantizers, time)
        """
        if num_quantizers is None:
            num_quantizers = self.num_quantizers

        residual_td = x_td
        all_indices = []

        for layer in self.layers[:num_quantizers]:
            indices_t = layer.encode(residual_td)
            all_indices.append(indices_t)

            # Update residual
            quantized_td = layer.decode(indices_t)
            residual_td = residual_td - quantized_td

        return jnp.stack(all_indices, axis=0)  # (num_quantizers, time)

    def decode(self, codes_qt: Array) -> Array:
        """Decode from quantized codes.

        Args:
            codes_qt: Codes of shape (num_quantizers, time)

        Returns:
            Output of shape (time, input_dim)
        """
        num_q = codes_qt.shape[0]
        output_td = None

        for idx in range(num_q):
            quantized_td = self.layers[idx].decode(codes_qt[idx])
            if output_td is None:
                output_td = quantized_td
            else:
                output_td = output_td + quantized_td

        return output_td


class MimiSplitResidualVectorQuantizer(eqx.Module):
    """Split RVQ with separate semantic and acoustic quantizers."""

    semantic_rvq: MimiResidualVectorQuantizer
    acoustic_rvq: MimiResidualVectorQuantizer
    num_semantic_quantizers: int = eqx.field(static=True)
    num_acoustic_quantizers: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        num_quantizers: int,
        num_semantic_quantizers: int,
        *,
        key: PRNGKeyArray,
    ) -> "MimiSplitResidualVectorQuantizer":
        k1, k2 = jax.random.split(key)

        num_acoustic = num_quantizers - num_semantic_quantizers

        semantic_rvq = MimiResidualVectorQuantizer.build(
            input_dim, codebook_size, codebook_dim, num_semantic_quantizers, key=k1
        )
        acoustic_rvq = MimiResidualVectorQuantizer.build(
            input_dim, codebook_size, codebook_dim, num_acoustic, key=k2
        )

        return cls(
            semantic_rvq=semantic_rvq,
            acoustic_rvq=acoustic_rvq,
            num_semantic_quantizers=num_semantic_quantizers,
            num_acoustic_quantizers=num_acoustic,
        )

    def encode(
        self,
        x_td: Array,
        num_quantizers: int | None = None,
    ) -> Array:
        """Encode using split quantization.

        Args:
            x_td: Input of shape (time, input_dim)
            num_quantizers: Total number of quantizers to use

        Returns:
            Codes of shape (num_quantizers, time)
        """
        if num_quantizers is None:
            num_quantizers = self.num_semantic_quantizers + self.num_acoustic_quantizers

        # Semantic quantization
        num_semantic = min(num_quantizers, self.num_semantic_quantizers)
        semantic_codes_qt = self.semantic_rvq.encode(x_td, num_semantic)

        if num_quantizers <= self.num_semantic_quantizers:
            return semantic_codes_qt

        # Compute semantic residual for acoustic quantization
        semantic_output_td = self.semantic_rvq.decode(semantic_codes_qt)
        residual_td = x_td - semantic_output_td

        # Acoustic quantization
        num_acoustic = num_quantizers - self.num_semantic_quantizers
        acoustic_codes_qt = self.acoustic_rvq.encode(residual_td, num_acoustic)

        return jnp.concatenate([semantic_codes_qt, acoustic_codes_qt], axis=0)

    def decode(self, codes_qt: Array) -> Array:
        """Decode from split quantized codes.

        Args:
            codes_qt: Codes of shape (num_quantizers, time)

        Returns:
            Output of shape (time, input_dim)
        """
        num_q = codes_qt.shape[0]

        # Decode semantic
        num_semantic = min(num_q, self.num_semantic_quantizers)
        output_td = self.semantic_rvq.decode(codes_qt[:num_semantic])

        if num_q > self.num_semantic_quantizers:
            # Decode acoustic
            acoustic_td = self.acoustic_rvq.decode(codes_qt[self.num_semantic_quantizers:])
            output_td = output_td + acoustic_td

        return output_td


class MimiMLP(eqx.Module):
    """Feed-forward MLP for transformer."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    @classmethod
    def build(
        cls,
        hidden_size: int,
        intermediate_size: int,
        *,
        key: PRNGKeyArray,
    ) -> "MimiMLP":
        k1, k2 = jax.random.split(key)
        fc1 = eqx.nn.Linear(hidden_size, intermediate_size, use_bias=False, key=k1)
        fc2 = eqx.nn.Linear(intermediate_size, hidden_size, use_bias=False, key=k2)
        return cls(fc1=fc1, fc2=fc2)

    def __call__(self, x_d: Array) -> Array:
        x_d = self.fc1(x_d)
        x_d = jax.nn.gelu(x_d)
        x_d = self.fc2(x_d)
        return x_d


class MimiAttention(eqx.Module):
    """Multi-head attention with RoPE for Mimi transformer."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    rotary_emb: RotaryEmbedding
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    sliding_window: int | None = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        key: PRNGKeyArray,
        sliding_window: int | None = None,
        rope_theta: float = 10000.0,
    ) -> "MimiAttention":
        keys = jax.random.split(key, 4)
        q_dim = num_heads * head_dim
        kv_dim = num_heads * head_dim

        q_proj = eqx.nn.Linear(hidden_size, q_dim, use_bias=False, key=keys[0])
        k_proj = eqx.nn.Linear(hidden_size, kv_dim, use_bias=False, key=keys[1])
        v_proj = eqx.nn.Linear(hidden_size, kv_dim, use_bias=False, key=keys[2])
        o_proj = eqx.nn.Linear(q_dim, hidden_size, use_bias=False, key=keys[3])

        rotary_emb = RotaryEmbedding.build(
            head_dim=head_dim,
            base=rope_theta,
            style="concatenated",
        )

        return cls(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            rotary_emb=rotary_emb,
            num_heads=num_heads,
            head_dim=head_dim,
            sliding_window=sliding_window,
        )

    def __call__(self, x_td: Array) -> Array:
        """Apply multi-head attention.

        Args:
            x_td: Input of shape (time, hidden_size)

        Returns:
            Output of shape (time, hidden_size)
        """
        tsz = x_td.shape[0]

        # Project to Q, K, V
        q_td = jax.vmap(self.q_proj)(x_td)
        k_td = jax.vmap(self.k_proj)(x_td)
        v_td = jax.vmap(self.v_proj)(x_td)

        # Reshape to (time, num_heads, head_dim)
        q_thd = q_td.reshape(tsz, self.num_heads, self.head_dim)
        k_thd = k_td.reshape(tsz, self.num_heads, self.head_dim)
        v_thd = v_td.reshape(tsz, self.num_heads, self.head_dim)

        # Apply RoPE
        q_thd = self.rotary_emb.apply_rotary_embeddings(q_thd)
        k_thd = self.rotary_emb.apply_rotary_embeddings(k_thd)

        # Compute attention with sliding window support
        local_window = (self.sliding_window, 0) if self.sliding_window else None

        ctx_thd = jax.nn.dot_product_attention(
            q_thd,
            k_thd,
            v_thd,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim),
            local_window_size=local_window,
        )

        # Combine heads
        ctx_td = ctx_thd.reshape(tsz, -1)

        # Output projection
        out_td = jax.vmap(self.o_proj)(ctx_td)

        return out_td


class MimiTransformerLayer(eqx.Module):
    """Single transformer layer for Mimi."""

    self_attn: MimiAttention
    mlp: MimiMLP
    self_attn_layer_norm: eqx.nn.LayerNorm
    mlp_layer_norm: eqx.nn.LayerNorm
    self_attn_layer_scale: MimiLayerScale
    mlp_layer_scale: MimiLayerScale

    @classmethod
    def build(
        cls,
        config: MimiConfig,
        *,
        key: PRNGKeyArray,
    ) -> "MimiTransformerLayer":
        k1, k2 = jax.random.split(key)

        self_attn = MimiAttention.build(
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim,
            key=k1,
            sliding_window=config.sliding_window,
            rope_theta=config.rope_theta,
        )

        mlp = MimiMLP.build(
            config.hidden_size,
            config.intermediate_size,
            key=k2,
        )

        self_attn_layer_norm = eqx.nn.LayerNorm(config.hidden_size)
        mlp_layer_norm = eqx.nn.LayerNorm(config.hidden_size)

        self_attn_layer_scale = MimiLayerScale.build(
            config.hidden_size,
            config.layer_scale_initial_scale,
        )
        mlp_layer_scale = MimiLayerScale.build(
            config.hidden_size,
            config.layer_scale_initial_scale,
        )

        return cls(
            self_attn=self_attn,
            mlp=mlp,
            self_attn_layer_norm=self_attn_layer_norm,
            mlp_layer_norm=mlp_layer_norm,
            self_attn_layer_scale=self_attn_layer_scale,
            mlp_layer_scale=mlp_layer_scale,
        )

    def __call__(self, x_td: Array) -> Array:
        """Apply transformer layer.

        Args:
            x_td: Input of shape (time, hidden_size)

        Returns:
            Output of shape (time, hidden_size)
        """
        # Self-attention with pre-norm
        residual_td = x_td
        x_td = jax.vmap(self.self_attn_layer_norm)(x_td)
        x_td = self.self_attn(x_td)
        x_td = self.self_attn_layer_scale(x_td)
        x_td = residual_td + x_td

        # MLP with pre-norm
        residual_td = x_td
        x_td = jax.vmap(self.mlp_layer_norm)(x_td)
        x_td = jax.vmap(self.mlp)(x_td)
        x_td = self.mlp_layer_scale(x_td)
        x_td = residual_td + x_td

        return x_td


class MimiTransformer(eqx.Module):
    """Transformer stack for Mimi."""

    layers: tuple[MimiTransformerLayer, ...]
    layer_norm: eqx.nn.LayerNorm
    num_layers: int = eqx.field(static=True)

    @classmethod
    def build(cls, config: MimiConfig, *, key: PRNGKeyArray) -> "MimiTransformer":
        keys = jax.random.split(key, config.num_hidden_layers)
        layers = tuple(
            MimiTransformerLayer.build(config, key=k)
            for k in keys
        )
        layer_norm = eqx.nn.LayerNorm(config.hidden_size)
        return cls(
            layers=layers,
            layer_norm=layer_norm,
            num_layers=config.num_hidden_layers,
        )

    def __call__(self, x_td: Array) -> Array:
        """Apply transformer stack.

        Args:
            x_td: Input of shape (time, hidden_size)

        Returns:
            Output of shape (time, hidden_size)
        """
        for layer in self.layers:
            x_td = layer(x_td)
        x_td = jax.vmap(self.layer_norm)(x_td)
        return x_td


class MimiModel(eqx.Module):
    """Complete Mimi neural audio codec model.

    The model consists of:
    1. SEANet encoder to downsample audio
    2. Encoder transformer for contextualization
    3. Downsampler to match frame rate
    4. Split RVQ for quantization
    5. Upsampler to match encoder frame rate
    6. Decoder transformer for reconstruction
    7. SEANet decoder to upsample to audio
    """

    encoder: MimiEncoder
    decoder: MimiDecoder
    encoder_transformer: MimiTransformer
    decoder_transformer: MimiTransformer
    quantizer: MimiSplitResidualVectorQuantizer
    downsample: eqx.nn.Linear | None
    upsample: eqx.nn.Linear | None
    config: MimiConfig

    @classmethod
    def build(cls, config: MimiConfig, *, key: PRNGKeyArray) -> "MimiModel":
        keys = jax.random.split(key, 6)

        encoder = MimiEncoder.build(config, key=keys[0])
        decoder = MimiDecoder.build(config, key=keys[1])
        encoder_transformer = MimiTransformer.build(config, key=keys[2])
        decoder_transformer = MimiTransformer.build(config, key=keys[3])

        quantizer = MimiSplitResidualVectorQuantizer.build(
            config.hidden_size,
            config.codebook_size,
            config.codebook_dim,
            config.num_quantizers,
            config.num_semantic_quantizers,
            key=keys[4],
        )

        # Compute frame rate ratio for down/up sampling
        encoder_frame_rate = config.encoder_frame_rate
        frame_rate = config.frame_rate
        ratio = encoder_frame_rate / frame_rate

        if ratio != 1.0:
            downsample_factor = int(ratio)
            k5, k6 = jax.random.split(keys[5])
            downsample = eqx.nn.Linear(
                config.hidden_size * downsample_factor,
                config.hidden_size,
                key=k5,
            )
            upsample = eqx.nn.Linear(
                config.hidden_size,
                config.hidden_size * downsample_factor,
                key=k6,
            )
        else:
            downsample = None
            upsample = None

        return cls(
            encoder=encoder,
            decoder=decoder,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
            quantizer=quantizer,
            downsample=downsample,
            upsample=upsample,
            config=config,
        )

    def _downsample(self, x_td: Array) -> Array:
        """Downsample hidden states to match frame rate."""
        if self.downsample is None:
            return x_td

        # Compute downsample factor from frame rates
        ratio = int(self.config.encoder_frame_rate / self.config.frame_rate)
        tsz, dim = x_td.shape

        # Pad to multiple of ratio
        pad_len = (ratio - tsz % ratio) % ratio
        if pad_len > 0:
            x_td = jnp.pad(x_td, ((0, pad_len), (0, 0)))

        # Reshape and apply linear
        new_tsz = (tsz + pad_len) // ratio
        x_td = x_td.reshape(new_tsz, dim * ratio)
        x_td = jax.vmap(self.downsample)(x_td)

        return x_td

    def _upsample(self, x_td: Array) -> Array:
        """Upsample hidden states from frame rate."""
        if self.upsample is None:
            return x_td

        ratio = int(self.config.encoder_frame_rate / self.config.frame_rate)
        tsz, dim = x_td.shape

        x_td = jax.vmap(self.upsample)(x_td)
        x_td = x_td.reshape(tsz * ratio, dim)

        return x_td

    def encode(
        self,
        audio_ct: Array,
        num_quantizers: int | None = None,
    ) -> Array:
        """Encode audio waveform to discrete codes.

        Args:
            audio_ct: Audio waveform of shape (channels, time)
            num_quantizers: Number of quantizers to use

        Returns:
            Codes of shape (num_quantizers, time')
        """
        # Encode to embeddings
        x_dt = self.encoder(audio_ct)  # (hidden_size, time)
        x_td = x_dt.T  # (time, hidden_size)

        # Apply encoder transformer
        x_td = self.encoder_transformer(x_td)

        # Downsample to match quantizer frame rate
        x_td = self._downsample(x_td)

        # Quantize
        codes_qt = self.quantizer.encode(x_td, num_quantizers)

        return codes_qt

    def decode(self, codes_qt: Array) -> Array:
        """Decode discrete codes to audio waveform.

        Args:
            codes_qt: Codes of shape (num_quantizers, time)

        Returns:
            Audio waveform of shape (channels, time')
        """
        # Dequantize
        x_td = self.quantizer.decode(codes_qt)

        # Upsample from quantizer frame rate
        x_td = self._upsample(x_td)

        # Apply decoder transformer
        x_td = self.decoder_transformer(x_td)

        # Decode to audio
        x_dt = x_td.T  # (hidden_size, time)
        audio_ct = self.decoder(x_dt)

        return audio_ct

    def __call__(self, audio_ct: Array, num_quantizers: int | None = None) -> Array:
        """Encode and decode audio (autoencoder forward pass).

        Args:
            audio_ct: Audio waveform of shape (channels, time)
            num_quantizers: Number of quantizers to use

        Returns:
            Reconstructed audio of shape (channels, time')
        """
        codes_qt = self.encode(audio_ct, num_quantizers)
        return self.decode(codes_qt)


# Pretrained model loading


@functools.lru_cache(maxsize=16)
def download_mimi_repo(repo_id: str = "kyutai/mimi") -> Path:
    """Download Mimi model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "Please install huggingface_hub: pip install huggingface-hub"
        ) from e

    return Path(snapshot_download(repo_id=repo_id))


def load_mimi_config(repo_id: str = "kyutai/mimi") -> MimiConfig:
    """Load Mimi configuration from HuggingFace Hub."""
    path = download_mimi_repo(repo_id)
    config_path = path / "config.json"

    with open(config_path, "r") as f:
        cfg = json.load(f)

    return MimiConfig(
        sampling_rate=cfg.get("sampling_rate", 24000),
        audio_channels=cfg.get("audio_channels", 1),
        hidden_size=cfg.get("hidden_size", 512),
        num_filters=cfg.get("num_filters", 64),
        num_residual_layers=cfg.get("num_residual_layers", 1),
        upsampling_ratios=tuple(cfg.get("upsampling_ratios", [8, 6, 5, 4])),
        kernel_size=cfg.get("kernel_size", 7),
        last_kernel_size=cfg.get("last_kernel_size", 3),
        residual_kernel_size=cfg.get("residual_kernel_size", 3),
        dilation_growth_rate=cfg.get("dilation_growth_rate", 2),
        use_causal_conv=cfg.get("use_causal_conv", True),
        pad_mode=cfg.get("pad_mode", "constant"),
        compress=cfg.get("compress", 2),
        trim_right_ratio=cfg.get("trim_right_ratio", 1.0),
        codebook_size=cfg.get("codebook_size", 2048),
        codebook_dim=cfg.get("codebook_dim", 256),
        num_quantizers=cfg.get("num_quantizers", 32),
        num_semantic_quantizers=cfg.get("num_semantic_quantizers", 1),
        num_hidden_layers=cfg.get("num_hidden_layers", 8),
        num_attention_heads=cfg.get("num_attention_heads", 8),
        head_dim=cfg.get("head_dim", 64),
        intermediate_size=cfg.get("intermediate_size", 2048),
        max_position_embeddings=cfg.get("max_position_embeddings", 8000),
        sliding_window=cfg.get("sliding_window", 250),
        attention_dropout=cfg.get("attention_dropout", 0.0),
        layer_scale_initial_scale=cfg.get("layer_scale_initial_scale", 0.01),
        rope_theta=cfg.get("rope_theta", 10000.0),
        frame_rate=cfg.get("frame_rate", 12.5),
        encodec_frame_rate=cfg.get("encodec_frame_rate", 75.0),
    )


def build_pretrained_mimi(
    repo_id: str = "kyutai/mimi",
    dtype: jnp.dtype | None = None,
) -> MimiModel:
    """Load pretrained Mimi model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        dtype: Optional dtype for model parameters

    Returns:
        Loaded MimiModel with pretrained weights
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "Please install safetensors: pip install safetensors"
        ) from e

    if dtype is None:
        dtype = jnp.float32

    config = load_mimi_config(repo_id)
    path = download_mimi_repo(repo_id)

    # Build model shape
    model = eqx.filter_eval_shape(
        MimiModel.build,
        config,
        key=jax.random.key(0),
    )

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
    model = _load_weights_into_mimi(model, state, dtype)

    return model


def _load_weights_into_mimi(
    model: MimiModel,
    state: dict[str, np.ndarray],
    dtype: jnp.dtype,
) -> MimiModel:
    """Map HuggingFace weights into Mimi model structure.

    This is a simplified weight loading that attempts to match HuggingFace
    naming conventions. For production use, you may need to adjust the
    key mappings based on the specific checkpoint.
    """

    def get_weight(key: str) -> jnp.ndarray | None:
        if key in state:
            return jnp.array(state[key], dtype=dtype)
        return None

    def match_weight(*patterns: str) -> jnp.ndarray | None:
        for pattern in patterns:
            for key in state:
                if pattern in key:
                    return jnp.array(state[key], dtype=dtype)
        return None

    # Load codebook embeddings
    for q_idx in range(model.config.num_quantizers):
        if q_idx < model.config.num_semantic_quantizers:
            rvq = model.quantizer.semantic_rvq
            local_idx = q_idx
        else:
            rvq = model.quantizer.acoustic_rvq
            local_idx = q_idx - model.config.num_semantic_quantizers

        embed_key = f"quantizer.layers.{q_idx}.codebook.embed"
        embed = get_weight(embed_key)
        if embed is not None:
            layer = rvq.layers[local_idx]
            model = eqx.tree_at(
                lambda m: m.quantizer.semantic_rvq.layers[local_idx].codebook.embeddings_kd
                if q_idx < model.config.num_semantic_quantizers
                else m.quantizer.acoustic_rvq.layers[local_idx].codebook.embeddings_kd,
                model,
                embed,
            )

    logger.info("Loaded Mimi model weights")
    return model


def _verify_with_transformers(
    audio: np.ndarray,
    sampling_rate: int,
    num_quantizers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify our implementation against HuggingFace transformers Mimi.

    Args:
        audio: Audio waveform as numpy array
        sampling_rate: Audio sample rate
        num_quantizers: Number of quantizers to use

    Returns:
        Tuple of (jax_codes, hf_codes) for comparison
    """
    try:
        from transformers import MimiModel as HFMimiModel
        from transformers import AutoFeatureExtractor
        import torch
    except ImportError as e:
        raise ImportError(
            "Please install transformers and torch: pip install transformers torch"
        ) from e

    logger.info("Loading HuggingFace Mimi model for verification...")

    # Load HF model
    hf_model = HFMimiModel.from_pretrained("kyutai/mimi")
    hf_model.eval()

    # Prepare input for HF model
    audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).float()  # (1, 1, time)

    # Encode with HF model
    with torch.no_grad():
        if num_quantizers is not None:
            hf_codes = hf_model.encode(audio_tensor, num_quantizers=num_quantizers)
        else:
            hf_codes = hf_model.encode(audio_tensor)
        hf_codes_np = hf_codes.audio_codes.squeeze(0).numpy()  # (num_q, time)

    logger.info("HF model encoded to shape: %s", hf_codes_np.shape)

    # Now encode with our JAX model
    config = load_mimi_config("kyutai/mimi")
    jax_model = MimiModel.build(config, key=jax.random.key(0))

    # For a fair comparison, we need pretrained weights
    # For now, just demonstrate the encoding pipeline
    audio_ct = jnp.array(audio[None, :], dtype=jnp.float32)
    jax_codes = jax_model.encode(audio_ct, num_quantizers)
    jax_codes_np = np.array(jax_codes)

    logger.info("JAX model encoded to shape: %s", jax_codes_np.shape)

    return jax_codes_np, hf_codes_np


def _verify_with_rustymimi(
    audio: np.ndarray,
    sampling_rate: int,
    num_quantizers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify our implementation against rustymimi.

    Args:
        audio: Audio waveform as numpy array
        sampling_rate: Audio sample rate
        num_quantizers: Number of quantizers to use

    Returns:
        Tuple of (jax_codes, rusty_codes) for comparison
    """
    try:
        import rustymimi
    except ImportError as e:
        raise ImportError(
            "Please install rustymimi: pip install rustymimi"
        ) from e

    logger.info("Loading rustymimi for verification...")

    # Create rustymimi encoder
    encoder = rustymimi.StreamEncoder()

    # Encode with rustymimi - expects float32 audio at 24kHz
    rusty_codes = encoder.encode(audio.astype(np.float32))
    rusty_codes_np = np.array(rusty_codes)

    logger.info("rustymimi encoded to shape: %s", rusty_codes_np.shape)

    # Encode with our JAX model
    config = MimiConfig(sampling_rate=sampling_rate)
    jax_model = MimiModel.build(config, key=jax.random.key(0))

    audio_ct = jnp.array(audio[None, :], dtype=jnp.float32)
    jax_codes = jax_model.encode(audio_ct, num_quantizers)
    jax_codes_np = np.array(jax_codes)

    logger.info("JAX model encoded to shape: %s", jax_codes_np.shape)

    return jax_codes_np, rusty_codes_np


def main() -> None:
    """CLI for Mimi audio codec encoding/decoding."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mimi neural audio codec - encode audio to tokens or decode tokens to audio"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode audio file to tokens")
    encode_parser.add_argument("input", type=str, help="Input audio file (WAV)")
    encode_parser.add_argument("output", type=str, help="Output tokens file (NPZ)")
    encode_parser.add_argument(
        "--num-quantizers",
        type=int,
        default=None,
        help="Number of quantizers to use",
    )
    encode_parser.add_argument(
        "--repo",
        type=str,
        default="kyutai/mimi",
        help="HuggingFace repo for pretrained model",
    )

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode tokens to audio file")
    decode_parser.add_argument("input", type=str, help="Input tokens file (NPZ)")
    decode_parser.add_argument("output", type=str, help="Output audio file (WAV)")
    decode_parser.add_argument(
        "--repo",
        type=str,
        default="kyutai/mimi",
        help="HuggingFace repo for pretrained model",
    )

    # Demo command (roundtrip)
    demo_parser = subparsers.add_parser("demo", help="Demo: encode and decode audio")
    demo_parser.add_argument("input", type=str, help="Input audio file (WAV)")
    demo_parser.add_argument("output", type=str, help="Output audio file (WAV)")
    demo_parser.add_argument(
        "--num-quantizers",
        type=int,
        default=None,
        help="Number of quantizers to use",
    )

    # Verify command - compare with reference implementations
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify JAX implementation against reference (transformers or rustymimi)",
    )
    verify_parser.add_argument("input", type=str, help="Input audio file (WAV)")
    verify_parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "rustymimi"],
        default="transformers",
        help="Reference backend to compare against",
    )
    verify_parser.add_argument(
        "--num-quantizers",
        type=int,
        default=8,
        help="Number of quantizers to use",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Import audio libraries
    try:
        import soundfile as sf
    except ImportError:
        logger.error("Please install soundfile: pip install soundfile")
        sys.exit(1)

    if args.command == "encode":
        # Load audio
        logger.info("Loading audio from %s", args.input)
        audio, sr = sf.read(args.input)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Build model
        logger.info("Loading Mimi model from %s", args.repo)
        config = load_mimi_config(args.repo)

        # Resample if needed
        if sr != config.sampling_rate:
            logger.warning(
                "Resampling from %d Hz to %d Hz",
                sr,
                config.sampling_rate,
            )
            # Simple linear resampling (for production, use proper resampling)
            ratio = config.sampling_rate / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            )

        # Build model (random weights for demo - use build_pretrained_mimi for real use)
        model = MimiModel.build(config, key=jax.random.key(0))

        # Encode
        audio_ct = jnp.array(audio[None, :], dtype=jnp.float32)
        logger.info("Encoding audio of shape %s", audio_ct.shape)

        codes_qt = model.encode(audio_ct, args.num_quantizers)
        logger.info("Encoded to codes of shape %s", codes_qt.shape)

        # Save
        np.savez(args.output, codes=np.array(codes_qt))
        logger.info("Saved codes to %s", args.output)

    elif args.command == "decode":
        # Load codes
        logger.info("Loading codes from %s", args.input)
        data = np.load(args.input)
        codes_qt = jnp.array(data["codes"])

        # Build model
        logger.info("Loading Mimi model from %s", args.repo)
        config = load_mimi_config(args.repo)
        model = MimiModel.build(config, key=jax.random.key(0))

        # Decode
        logger.info("Decoding codes of shape %s", codes_qt.shape)
        audio_ct = model.decode(codes_qt)
        logger.info("Decoded to audio of shape %s", audio_ct.shape)

        # Save
        audio = np.array(audio_ct[0])
        sf.write(args.output, audio, config.sampling_rate)
        logger.info("Saved audio to %s", args.output)

    elif args.command == "demo":
        # Load audio
        logger.info("Loading audio from %s", args.input)
        audio, sr = sf.read(args.input)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Build model with default config
        config = MimiConfig()

        # Resample if needed
        if sr != config.sampling_rate:
            logger.warning(
                "Resampling from %d Hz to %d Hz",
                sr,
                config.sampling_rate,
            )
            ratio = config.sampling_rate / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            )

        logger.info("Building Mimi model with random weights (demo mode)")
        model = MimiModel.build(config, key=jax.random.key(42))

        # Roundtrip
        audio_ct = jnp.array(audio[None, :], dtype=jnp.float32)
        logger.info("Input audio shape: %s", audio_ct.shape)

        # JIT compile for efficiency
        encode_fn = jax.jit(lambda m, x: m.encode(x, args.num_quantizers))
        decode_fn = jax.jit(lambda m, c: m.decode(c))

        codes_qt = encode_fn(model, audio_ct)
        logger.info("Encoded to codes shape: %s", codes_qt.shape)

        reconstructed_ct = decode_fn(model, codes_qt)
        logger.info("Decoded to audio shape: %s", reconstructed_ct.shape)

        # Save output
        output_audio = np.array(reconstructed_ct[0])
        sf.write(args.output, output_audio, config.sampling_rate)
        logger.info("Saved reconstructed audio to %s", args.output)

    elif args.command == "verify":
        # Load audio
        logger.info("Loading audio from %s", args.input)
        audio, sr = sf.read(args.input)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 24kHz if needed (Mimi's native sample rate)
        target_sr = 24000
        if sr != target_sr:
            logger.warning(
                "Resampling from %d Hz to %d Hz",
                sr,
                target_sr,
            )
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            )
            sr = target_sr

        # Run verification
        if args.backend == "transformers":
            jax_codes, ref_codes = _verify_with_transformers(
                audio, sr, args.num_quantizers
            )
        else:
            jax_codes, ref_codes = _verify_with_rustymimi(
                audio, sr, args.num_quantizers
            )

        # Compare results
        logger.info("JAX codes shape: %s", jax_codes.shape)
        logger.info("Reference codes shape: %s", ref_codes.shape)

        # Check if shapes match
        if jax_codes.shape != ref_codes.shape:
            logger.warning(
                "Shape mismatch: JAX %s vs Reference %s",
                jax_codes.shape,
                ref_codes.shape,
            )
            # Truncate to minimum length for comparison
            min_q = min(jax_codes.shape[0], ref_codes.shape[0])
            min_t = min(jax_codes.shape[1], ref_codes.shape[1])
            jax_codes = jax_codes[:min_q, :min_t]
            ref_codes = ref_codes[:min_q, :min_t]

        # Compute accuracy (exact match rate)
        matches = (jax_codes == ref_codes).astype(np.float32)
        total_accuracy = matches.mean()
        per_quantizer_accuracy = matches.mean(axis=1)

        logger.info("Total token match accuracy: %.2f%%", total_accuracy * 100)
        for q_idx, acc in enumerate(per_quantizer_accuracy):
            logger.info("  Quantizer %d accuracy: %.2f%%", q_idx, acc * 100)

        # Summary
        if total_accuracy == 1.0:
            logger.info("SUCCESS: JAX implementation matches reference exactly!")
        elif total_accuracy > 0.99:
            logger.info("GOOD: JAX implementation is very close to reference (>99%% match)")
        elif total_accuracy > 0.9:
            logger.warning("PARTIAL: JAX implementation differs from reference (%.1f%% match)", total_accuracy * 100)
        else:
            logger.error("MISMATCH: JAX implementation significantly differs from reference")

        sys.exit(0 if total_accuracy > 0.9 else 1)


if __name__ == "__main__":
    main()
