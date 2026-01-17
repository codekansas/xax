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
from typing import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from xax.arch.attention import RotaryEmbedding

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Please install huggingface_hub: pip install huggingface-hub") from e

try:
    from safetensors import safe_open
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Please install safetensors: pip install safetensors") from e

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
    upsample_groups: int = field(default=512)
    use_conv_shortcut: bool = field(default=False)

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
    encodec_frame_rate: float = field(default=25.0)  # HF default is 25

    @property
    def num_codebooks(self) -> int:
        return self.num_quantizers

    @property
    def encoder_frame_rate(self) -> float:
        hop_length = math.prod(self.upsampling_ratios)
        return self.sampling_rate / hop_length


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
        # Note: JAX conv_transpose requires kernel flipped to match PyTorch ConvTranspose1d
        weight_flipped = self.weight_oik[:, :, ::-1]
        out_1ot = jax.lax.conv_transpose(
            x_1ct,
            weight_flipped,
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
        # Encoder uses reversed upsampling_ratios (from smallest to largest ratio)
        blocks = []
        in_channels = config.num_filters
        encoder_ratios = config.upsampling_ratios[::-1]

        for _idx, ratio in enumerate(encoder_ratios):
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
        # Note: decoder init_conv uses kernel_size (7), not last_kernel_size (3)
        init_conv = MimiConv1d.build(
            config.hidden_size,
            channels[0],
            config.kernel_size,
            key=keys[key_idx],
            causal=config.use_causal_conv,
            pad_mode=config.pad_mode,
        )
        key_idx += 1

        # Upsampling blocks - use upsampling_ratios directly (largest to smallest)
        blocks = []
        decoder_ratios = config.upsampling_ratios  # (8, 6, 5, 4)

        for idx, ratio in enumerate(decoder_ratios):
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
        # Note: decoder final_conv uses last_kernel_size (3), not kernel_size (7)
        final_conv = MimiConv1d.build(
            channels[-1],
            config.audio_channels,
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


class MimiConv1dProj(eqx.Module):
    """1D convolution with kernel_size=1 used for projection in quantizer."""

    weight_oik: Array  # (out_channels, in_channels, 1)

    @classmethod
    def build(cls, in_channels: int, out_channels: int, *, key: PRNGKeyArray) -> "MimiConv1dProj":
        fan_in = in_channels
        std = 1.0 / math.sqrt(fan_in)
        weight_oik = jax.random.uniform(key, (out_channels, in_channels, 1), minval=-std, maxval=std)
        return cls(weight_oik=weight_oik)

    def __call__(self, x_td: Array) -> Array:
        """Apply 1x1 convolution.

        Args:
            x_td: Input of shape (time, in_channels)

        Returns:
            Output of shape (time, out_channels)
        """
        # Transpose to (in_channels, time) for conv
        x_ct = x_td.T
        x_1ct = x_ct[None, :, :]
        out_1ot = jax.lax.conv_general_dilated(
            x_1ct,
            self.weight_oik,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        out_ot = out_1ot[0]
        return out_ot.T  # Back to (time, out_channels)


class MimiResidualVectorQuantizer(eqx.Module):
    """Residual Vector Quantization using multiple codebooks.

    Has shared input/output projections and multiple codebook layers.
    Each successive quantizer operates on the residual from previous quantizers.
    """

    input_proj: MimiConv1dProj
    output_proj: MimiConv1dProj
    layers: tuple[MimiEuclideanCodebook, ...]
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
        keys = jax.random.split(key, num_quantizers + 2)

        input_proj = MimiConv1dProj.build(input_dim, codebook_dim, key=keys[0])
        output_proj = MimiConv1dProj.build(codebook_dim, input_dim, key=keys[1])
        layers = tuple(
            MimiEuclideanCodebook.build(codebook_size, codebook_dim, key=keys[i + 2]) for i in range(num_quantizers)
        )

        return cls(
            input_proj=input_proj,
            output_proj=output_proj,
            layers=layers,
            num_quantizers=num_quantizers,
        )

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

        # Project to codebook dimension
        x_proj_td = self.input_proj(x_td)

        residual_td = x_proj_td
        all_indices = []

        for codebook in self.layers[:num_quantizers]:
            indices_t = codebook.quantize(residual_td)
            all_indices.append(indices_t)

            # Update residual
            quantized_td = codebook.decode(indices_t)
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

        # Project back to input dimension
        assert output_td is not None, "No quantizer layers to decode"
        return self.output_proj(output_td)


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
        acoustic_rvq = MimiResidualVectorQuantizer.build(input_dim, codebook_size, codebook_dim, num_acoustic, key=k2)

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

        Note: Both semantic and acoustic RVQs receive the same input. Each RVQ
        internally computes residuals layer-by-layer. The semantic and acoustic
        outputs are added during decoding (they represent different "subspaces"
        of the signal due to different input/output projections).

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

        # Acoustic quantization - uses same input, not residual
        # (each RVQ has its own input/output projections)
        num_acoustic = num_quantizers - self.num_semantic_quantizers
        acoustic_codes_qt = self.acoustic_rvq.encode(x_td, num_acoustic)

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
            acoustic_td = self.acoustic_rvq.decode(codes_qt[self.num_semantic_quantizers :])
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
    """Transformer stack for Mimi.

    Note: HF Mimi does NOT have a final layer norm, unlike some other transformers.
    """

    layers: tuple[MimiTransformerLayer, ...]
    num_layers: int = eqx.field(static=True)

    @classmethod
    def build(cls, config: MimiConfig, *, key: PRNGKeyArray) -> "MimiTransformer":
        keys = jax.random.split(key, config.num_hidden_layers)
        layers = tuple(MimiTransformerLayer.build(config, key=k) for k in keys)
        return cls(
            layers=layers,
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
        return x_td


class MimiDownsampleConv(eqx.Module):
    """Strided 1D convolution for downsampling with causal padding.

    HF uses kernel_size = 2 * stride, with stride = int(encodec_frame_rate / frame_rate).
    Matches HuggingFace's dynamic padding for proper frame alignment.
    """

    weight_oik: Array  # (out_channels, in_channels, kernel_size)
    stride: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    padding_total: int = eqx.field(static=True)

    @classmethod
    def build(cls, channels: int, stride: int, *, key: PRNGKeyArray) -> "MimiDownsampleConv":
        # HF uses kernel_size = 2 * stride
        kernel_size = 2 * stride
        fan_in = channels * kernel_size
        std = 1.0 / math.sqrt(fan_in)
        weight_oik = jax.random.uniform(key, (channels, channels, kernel_size), minval=-std, maxval=std)
        # padding_total for "same" output size behavior
        padding_total = kernel_size - stride
        return cls(weight_oik=weight_oik, stride=stride, kernel_size=kernel_size, padding_total=padding_total)

    def _get_extra_padding(self, length: int) -> int:
        """Calculate extra padding needed for proper frame alignment (matches HF)."""
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        return max(0, ideal_length - length)

    def __call__(self, x_td: Array) -> Array:
        """Apply strided convolution for downsampling.

        Args:
            x_td: Input of shape (time, channels)

        Returns:
            Output of shape (ceil(time / stride), channels)
        """
        tsz = x_td.shape[0]

        # Calculate extra padding for frame alignment (HF's _get_extra_padding_for_conv1d)
        extra_padding = self._get_extra_padding(tsz)

        # Causal padding: all on left + extra on right
        pad_left = self.padding_total
        pad_right = extra_padding

        x_ct = x_td.T  # (channels, time)
        if pad_left > 0 or pad_right > 0:
            # HF uses "replicate" padding for downsample
            x_ct = jnp.pad(x_ct, ((0, 0), (pad_left, pad_right)), mode="edge")

        x_1ct = x_ct[None, :, :]
        out_1ot = jax.lax.conv_general_dilated(
            x_1ct,
            self.weight_oik,
            window_strides=(self.stride,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        return out_1ot[0].T


class MimiUpsampleConv(eqx.Module):
    """Transposed 1D convolution for upsampling using depthwise convolution.

    HF uses kernel_size = 2 * stride, with stride = int(encodec_frame_rate / frame_rate).
    Includes causal trimming to match HuggingFace output length.
    """

    weight_oik: Array  # (channels, 1, kernel_size)
    stride: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    padding_right: int = eqx.field(static=True)

    @classmethod
    def build(cls, channels: int, stride: int, *, key: PRNGKeyArray) -> "MimiUpsampleConv":
        # HF uses kernel_size = 2 * stride
        kernel_size = 2 * stride
        fan_in = kernel_size
        std = 1.0 / math.sqrt(fan_in)
        weight_oik = jax.random.uniform(key, (channels, 1, kernel_size), minval=-std, maxval=std)
        # Causal trimming: trim_right_ratio=1.0 means all padding trimmed from right
        padding_total = kernel_size - stride
        padding_right = math.ceil(padding_total * 1.0)  # trim_right_ratio=1.0
        return cls(
            weight_oik=weight_oik,
            stride=stride,
            kernel_size=kernel_size,
            channels=channels,
            padding_right=padding_right,
        )

    def __call__(self, x_td: Array) -> Array:
        """Apply transposed convolution for upsampling.

        Args:
            x_td: Input of shape (time, channels)

        Returns:
            Output of shape (time * stride, channels)
        """

        # Depthwise transpose conv: each channel independently
        def conv_single_channel(x_t: Array, w_k: Array) -> Array:
            x_11t = x_t[None, None, :]  # (1, 1, time)
            # Note: JAX conv_transpose requires kernel flipped to match PyTorch ConvTranspose1d
            w_11k = w_k[None, None, ::-1]  # (1, 1, kernel) flipped
            out_11t = jax.lax.conv_transpose(
                x_11t,
                w_11k,
                strides=(self.stride,),
                padding="VALID",
                dimension_numbers=("NCH", "OIH", "NCH"),
            )
            return out_11t[0, 0]  # (time',)

        # x_td.T is (channels, time), weight_oik[:, 0, :] is (channels, kernel)
        out_ct = jax.vmap(conv_single_channel)(x_td.T, self.weight_oik[:, 0, :])  # (channels, time')

        # Apply causal trimming (trim from right side)
        if self.padding_right > 0:
            out_ct = out_ct[:, : -self.padding_right]

        return out_ct.T


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
    downsample: MimiDownsampleConv | None
    upsample: MimiUpsampleConv | None
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
        # HF formula: stride = int(encodec_frame_rate / frame_rate)
        # kernel_size = 2 * stride is handled in the conv modules
        downsample_stride = int(config.encodec_frame_rate / config.frame_rate)

        if config.frame_rate != config.encodec_frame_rate:
            k5, k6 = jax.random.split(keys[5])
            downsample = MimiDownsampleConv.build(config.hidden_size, downsample_stride, key=k5)
            upsample = MimiUpsampleConv.build(config.hidden_size, downsample_stride, key=k6)
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
        return self.downsample(x_td)

    def _upsample(self, x_td: Array) -> Array:
        """Upsample hidden states from frame rate."""
        if self.upsample is None:
            return x_td
        return self.upsample(x_td)

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
        upsample_groups=cfg.get("upsample_groups", 512),
        use_conv_shortcut=cfg.get("use_conv_shortcut", False),
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
        encodec_frame_rate=cfg.get("encodec_frame_rate", 25.0),
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
    if dtype is None:
        dtype = jnp.float32

    config = load_mimi_config(repo_id)
    path = download_mimi_repo(repo_id)

    # Build model with actual values (not just shapes)
    model = MimiModel.build(config, key=jax.random.key(0))

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
    """Map HuggingFace weights into Mimi model structure."""

    def get_weight(key: str) -> jnp.ndarray | None:
        if key in state:
            return jnp.array(state[key], dtype=dtype)
        return None

    loaded_count = 0
    total_keys = len(state)

    # Helper to set weights using tree_at
    def set_weight(get_leaf: Callable[[MimiModel], Array], weight: jnp.ndarray) -> None:
        nonlocal model, loaded_count
        model = eqx.tree_at(get_leaf, model, weight)
        loaded_count += 1

    # Load encoder weights
    # Encoder structure: init_conv, then blocks of (resnet_layers, down_conv), then final_conv
    # HF structure: encoder.layers.0 = init_conv, then flat indexing

    # encoder.layers.0 = init_conv
    w = get_weight("encoder.layers.0.conv.weight")
    b = get_weight("encoder.layers.0.conv.bias")
    if w is not None:
        set_weight(lambda m: m.encoder.init_conv.weight_oik, w)
    if b is not None:
        set_weight(lambda m: m.encoder.init_conv.bias_o, b)

    # Encoder blocks: layer indices 1, 3, 4, 6, 7, 9, 10, 12 for resnet, 3, 6, 9, 12 for down_conv
    # Pattern: for each upsampling ratio, there's residual blocks then down conv
    # HF indices: 1 (resnet), 3 (down), 4 (resnet), 6 (down), 7 (resnet), 9 (down), 10 (resnet), 12 (down), 14 (final)
    enc_hf_indices = [
        (1, 3),  # Block 0: resnet at 1, down at 3
        (4, 6),  # Block 1: resnet at 4, down at 6
        (7, 9),  # Block 2: resnet at 7, down at 9
        (10, 12),  # Block 3: resnet at 10, down at 12
    ]

    for block_idx, (res_idx, down_idx) in enumerate(enc_hf_indices):
        # Resnet block weights
        w1 = get_weight(f"encoder.layers.{res_idx}.block.1.conv.weight")
        b1 = get_weight(f"encoder.layers.{res_idx}.block.1.conv.bias")
        w2 = get_weight(f"encoder.layers.{res_idx}.block.3.conv.weight")
        b2 = get_weight(f"encoder.layers.{res_idx}.block.3.conv.bias")

        if w1 is not None:
            set_weight(lambda m, bi=block_idx: m.encoder.blocks[bi][0][0].conv1.weight_oik, w1)
        if b1 is not None:
            set_weight(lambda m, bi=block_idx: m.encoder.blocks[bi][0][0].conv1.bias_o, b1)
        if w2 is not None:
            set_weight(lambda m, bi=block_idx: m.encoder.blocks[bi][0][0].conv2.weight_oik, w2)
        if b2 is not None:
            set_weight(lambda m, bi=block_idx: m.encoder.blocks[bi][0][0].conv2.bias_o, b2)

        # Down conv weights
        dw = get_weight(f"encoder.layers.{down_idx}.conv.weight")
        db = get_weight(f"encoder.layers.{down_idx}.conv.bias")
        if dw is not None:
            set_weight(lambda m, bi=block_idx: m.encoder.blocks[bi][1].weight_oik, dw)
        if db is not None:
            set_weight(lambda m, bi=block_idx: m.encoder.blocks[bi][1].bias_o, db)

    # encoder.layers.14 = final_conv
    w = get_weight("encoder.layers.14.conv.weight")
    b = get_weight("encoder.layers.14.conv.bias")
    if w is not None:
        set_weight(lambda m: m.encoder.final_conv.weight_oik, w)
    if b is not None:
        set_weight(lambda m: m.encoder.final_conv.bias_o, b)

    # Load decoder weights
    # decoder.layers.0 = init_conv, then blocks of (up_conv, resnet_layers), then final_conv
    w = get_weight("decoder.layers.0.conv.weight")
    b = get_weight("decoder.layers.0.conv.bias")
    if w is not None:
        set_weight(lambda m: m.decoder.init_conv.weight_oik, w)
    if b is not None:
        set_weight(lambda m: m.decoder.init_conv.bias_o, b)

    # Decoder blocks: up then resnet
    # HF indices: 2 (up), 3 (resnet), 5 (up), 6 (resnet), 8 (up), 9 (resnet), 11 (up), 12 (resnet), 14 (final)
    dec_hf_indices = [
        (2, 3),  # Block 0: up at 2, resnet at 3
        (5, 6),  # Block 1: up at 5, resnet at 6
        (8, 9),  # Block 2: up at 8, resnet at 9
        (11, 12),  # Block 3: up at 11, resnet at 12
    ]

    for block_idx, (up_idx, res_idx) in enumerate(dec_hf_indices):
        # Up conv weights (transposed conv)
        uw = get_weight(f"decoder.layers.{up_idx}.conv.weight")
        ub = get_weight(f"decoder.layers.{up_idx}.conv.bias")
        if uw is not None:
            set_weight(lambda m, bi=block_idx: m.decoder.blocks[bi][0].weight_oik, uw)
        if ub is not None:
            set_weight(lambda m, bi=block_idx: m.decoder.blocks[bi][0].bias_o, ub)

        # Resnet block weights
        w1 = get_weight(f"decoder.layers.{res_idx}.block.1.conv.weight")
        b1 = get_weight(f"decoder.layers.{res_idx}.block.1.conv.bias")
        w2 = get_weight(f"decoder.layers.{res_idx}.block.3.conv.weight")
        b2 = get_weight(f"decoder.layers.{res_idx}.block.3.conv.bias")

        if w1 is not None:
            set_weight(lambda m, bi=block_idx: m.decoder.blocks[bi][1][0].conv1.weight_oik, w1)
        if b1 is not None:
            set_weight(lambda m, bi=block_idx: m.decoder.blocks[bi][1][0].conv1.bias_o, b1)
        if w2 is not None:
            set_weight(lambda m, bi=block_idx: m.decoder.blocks[bi][1][0].conv2.weight_oik, w2)
        if b2 is not None:
            set_weight(lambda m, bi=block_idx: m.decoder.blocks[bi][1][0].conv2.bias_o, b2)

    # decoder.layers.14 = final_conv
    w = get_weight("decoder.layers.14.conv.weight")
    b = get_weight("decoder.layers.14.conv.bias")
    if w is not None:
        set_weight(lambda m: m.decoder.final_conv.weight_oik, w)
    if b is not None:
        set_weight(lambda m: m.decoder.final_conv.bias_o, b)

    # Load transformer weights (encoder and decoder)
    for prefix, _get_transformer in [
        ("encoder_transformer", lambda m: m.encoder_transformer),
        ("decoder_transformer", lambda m: m.decoder_transformer),
    ]:
        for layer_idx in range(model.config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{layer_idx}"

            # Self-attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                w = get_weight(f"{layer_prefix}.self_attn.{proj}.weight")
                if w is not None:
                    # JAX Linear uses (out, in), HF also uses (out, in)
                    set_weight(
                        lambda m, p=prefix, li=layer_idx, pj=proj: getattr(
                            getattr(m, p.split("_")[0] + "_transformer").layers[li].self_attn, pj
                        ).weight,
                        w,
                    )

            # Layer norms (HF: input_layernorm, post_attention_layernorm)
            for hf_name, our_name in [
                ("input_layernorm", "self_attn_layer_norm"),
                ("post_attention_layernorm", "mlp_layer_norm"),
            ]:
                w = get_weight(f"{layer_prefix}.{hf_name}.weight")
                b = get_weight(f"{layer_prefix}.{hf_name}.bias")
                if w is not None:
                    set_weight(
                        lambda m, p=prefix, li=layer_idx, on=our_name: getattr(
                            getattr(m, p.split("_")[0] + "_transformer").layers[li], on
                        ).weight,
                        w,
                    )
                if b is not None:
                    set_weight(
                        lambda m, p=prefix, li=layer_idx, on=our_name: getattr(
                            getattr(m, p.split("_")[0] + "_transformer").layers[li], on
                        ).bias,
                        b,
                    )

            # MLP
            for fc in ["fc1", "fc2"]:
                w = get_weight(f"{layer_prefix}.mlp.{fc}.weight")
                if w is not None:
                    set_weight(
                        lambda m, p=prefix, li=layer_idx, f=fc: getattr(
                            getattr(m, p.split("_")[0] + "_transformer").layers[li].mlp, f
                        ).weight,
                        w,
                    )

            # Layer scales
            for scale_name in ["self_attn_layer_scale", "mlp_layer_scale"]:
                s = get_weight(f"{layer_prefix}.{scale_name}.scale")
                if s is not None:
                    set_weight(
                        lambda m, p=prefix, li=layer_idx, sn=scale_name: getattr(
                            getattr(m, p.split("_")[0] + "_transformer").layers[li], sn
                        ).scale,
                        s,
                    )

    # Load downsample/upsample conv weights
    dw = get_weight("downsample.conv.weight")
    if dw is not None and model.downsample is not None:
        set_weight(lambda m: m.downsample.weight_oik, dw)

    uw = get_weight("upsample.conv.weight")
    if uw is not None and model.upsample is not None:
        set_weight(lambda m: m.upsample.weight_oik, uw)

    # Load quantizer weights
    # Semantic RVQ
    w = get_weight("quantizer.semantic_residual_vector_quantizer.input_proj.weight")
    if w is not None:
        set_weight(lambda m: m.quantizer.semantic_rvq.input_proj.weight_oik, w)

    w = get_weight("quantizer.semantic_residual_vector_quantizer.output_proj.weight")
    if w is not None:
        set_weight(lambda m: m.quantizer.semantic_rvq.output_proj.weight_oik, w)

    # Acoustic RVQ
    w = get_weight("quantizer.acoustic_residual_vector_quantizer.input_proj.weight")
    if w is not None:
        set_weight(lambda m: m.quantizer.acoustic_rvq.input_proj.weight_oik, w)

    w = get_weight("quantizer.acoustic_residual_vector_quantizer.output_proj.weight")
    if w is not None:
        set_weight(lambda m: m.quantizer.acoustic_rvq.output_proj.weight_oik, w)

    # Codebook embeddings - HF uses embed = embed_sum / cluster_usage for inference
    sem_prefix = "quantizer.semantic_residual_vector_quantizer.layers"
    for layer_idx in range(model.config.num_semantic_quantizers):
        embed_sum = get_weight(f"{sem_prefix}.{layer_idx}.codebook.embed_sum")
        cluster_usage = get_weight(f"{sem_prefix}.{layer_idx}.codebook.cluster_usage")
        if embed_sum is not None and cluster_usage is not None:
            embed = embed_sum / cluster_usage[:, None]
            set_weight(lambda m, li=layer_idx: m.quantizer.semantic_rvq.layers[li].embeddings_kd, embed)

    acou_prefix = "quantizer.acoustic_residual_vector_quantizer.layers"
    num_acoustic = model.config.num_quantizers - model.config.num_semantic_quantizers
    for layer_idx in range(num_acoustic):
        embed_sum = get_weight(f"{acou_prefix}.{layer_idx}.codebook.embed_sum")
        cluster_usage = get_weight(f"{acou_prefix}.{layer_idx}.codebook.cluster_usage")
        if embed_sum is not None and cluster_usage is not None:
            embed = embed_sum / cluster_usage[:, None]
            set_weight(lambda m, li=layer_idx: m.quantizer.acoustic_rvq.layers[li].embeddings_kd, embed)

    logger.info("Loaded %d/%d weight tensors", loaded_count, total_keys)
    return model


def main() -> None:
    """CLI for Mimi audio codec encoding/decoding."""
    import argparse  # noqa: PLC0415

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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Import audio libraries
    try:
        import soundfile as sf  # noqa: PLC0415
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

        # Build model.
        model = build_pretrained_mimi()

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
        model = build_pretrained_mimi(args.repo)

        # Decode
        logger.info("Decoding codes of shape %s", codes_qt.shape)
        audio_ct = model.decode(codes_qt)
        logger.info("Decoded to audio of shape %s", audio_ct.shape)

        # Save
        audio = np.array(audio_ct[0])
        sf.write(args.output, audio, config.sampling_rate)
        logger.info("Saved audio to %s", args.output)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
