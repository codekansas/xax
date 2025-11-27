"""Defines a general-purpose UNet model."""

import math
from collections.abc import Callable
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from xax.nn.activation import ActivationType, get_activation
from xax.nn.norm import Norm2D, NormType, get_norm_2d


class _MLP(eqx.Module):
    """Simple MLP for embedding projection."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    act_fn: Callable[[Array], Array]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act_fn: Callable[[Array], Array],
        key1: PRNGKeyArray,
        key2: PRNGKeyArray,
    ) -> None:
        self.linear1 = eqx.nn.Linear(in_dim, hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_dim, key=key2)
        self.act_fn = act_fn

    def __call__(self, x: Array) -> Array:
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class PositionalEmbedding(eqx.Module):
    """Positional embedding using sinusoidal encoding."""

    dim: int
    max_length: int
    embedding: Array

    def __init__(self, dim: int, max_length: int = 10000, *, key: PRNGKeyArray | None = None) -> None:
        self.dim = dim
        self.max_length = max_length
        self.embedding = self.make_embedding(dim, max_length)

    @staticmethod
    def make_embedding(dim: int, max_length: int = 10000) -> Array:
        """Create sinusoidal positional embeddings."""
        embedding = jnp.zeros((max_length, dim))
        position = jnp.arange(0, max_length)[:, None].astype(jnp.float32)
        div_term = jnp.exp(jnp.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        embedding = embedding.at[:, 0::2].set(jnp.sin(position * div_term))
        embedding = embedding.at[:, 1::2].set(jnp.cos(position * div_term))
        return embedding

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Integer indices of shape (...,) or (..., 1)

        Returns:
            Embeddings of shape (..., dim)
        """
        # Handle both scalar and array inputs
        if x.ndim == 0:
            x = x[None]
        return self.embedding[x]


class BasicBlock(eqx.Module):
    """Basic residual block with optional embedding conditioning."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: Norm2D
    norm2: Norm2D
    mlp_emb: _MLP | None
    shortcut: eqx.nn.Conv2d | eqx.nn.Identity

    def __init__(
        self,
        in_c: int,
        out_c: int,
        embed_c: int | None = None,
        act: ActivationType = "relu",
        norm: NormType = "batch_affine",
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2, key_emb, key_shortcut = jax.random.split(key, 4)

        self.conv1 = eqx.nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, use_bias=False, key=key1)
        self.norm1 = get_norm_2d(norm, dim=out_c)
        self.conv2 = eqx.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, use_bias=False, key=key2)
        self.norm2 = get_norm_2d(norm, dim=out_c)

        # Projects input embedding to embedding space
        if embed_c is not None:
            act_fn = get_activation(act)
            key_emb2 = jax.random.split(key_emb)[0]
            self.mlp_emb = _MLP(embed_c, embed_c, out_c, act_fn, key_emb, key_emb2)
        else:
            self.mlp_emb = None

        # Shortcut for residual connection
        if in_c == out_c:
            self.shortcut = eqx.nn.Identity()
        else:
            self.shortcut = eqx.nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, use_bias=False, key=key_shortcut)

    def __call__(self, x: Array, embedding: Array | None = None) -> Array:
        """Forward pass.

        Args:
            x: Input tensor of shape (C, H, W)
            embedding: Optional embedding tensor of shape (embed_c,)

        Returns:
            Output tensor of shape (out_c, H, W)
        """
        out = self.conv1(x)
        out = self.norm1(out)

        if embedding is not None:
            if self.mlp_emb is None:
                raise ValueError("Embedding was provided but embedding projection is None")
            tx = self.mlp_emb(embedding)
            out = out + tx[:, None, None]
        elif self.mlp_emb is not None:
            raise ValueError("Embedding projection is not None but no embedding was provided")

        out = jax.nn.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = jax.nn.relu(out + self.shortcut(x))
        return out


class SelfAttention2d(eqx.Module):
    """2D self-attention module."""

    dim: int
    num_heads: int
    q_conv: eqx.nn.Conv2d
    k_conv: eqx.nn.Conv2d
    v_conv: eqx.nn.Conv2d
    o_conv: eqx.nn.Conv2d
    dropout: eqx.nn.Dropout

    def __init__(self, dim: int, num_heads: int = 8, dropout_prob: float = 0.1, *, key: PRNGKeyArray) -> None:
        self.dim = dim
        self.num_heads = num_heads
        key_q, key_k, key_v, key_o = jax.random.split(key, 4)
        self.q_conv = eqx.nn.Conv2d(dim, dim, kernel_size=1, use_bias=True, key=key_q)
        self.k_conv = eqx.nn.Conv2d(dim, dim, kernel_size=1, use_bias=True, key=key_k)
        self.v_conv = eqx.nn.Conv2d(dim, dim, kernel_size=1, use_bias=True, key=key_v)
        self.o_conv = eqx.nn.Conv2d(dim, dim, kernel_size=1, use_bias=True, key=key_o)
        self.dropout = eqx.nn.Dropout(dropout_prob)

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """Forward pass.

        Args:
            x: Input tensor of shape (C, H, W)
            key: Optional random key for dropout

        Returns:
            Output tensor of shape (C, H, W)
        """
        _, h, w = x.shape
        head_dim = self.dim // self.num_heads

        q = self.q_conv(x)  # (C, H, W)
        k = self.k_conv(x)  # (C, H, W)
        v = self.v_conv(x)  # (C, H, W)

        # Reshape to (num_heads, head_dim, H*W)
        q = q.reshape(self.num_heads, head_dim, h * w)
        k = k.reshape(self.num_heads, head_dim, h * w)
        v = v.reshape(self.num_heads, head_dim, h * w)

        # Compute attention: (num_heads, H*W, H*W)
        a = jnp.einsum("h d q, h d k -> h q k", q, k) / (self.dim**0.5)
        a = jax.nn.softmax(a, axis=-1)
        if key is not None:
            a = self.dropout(a, key=key)
        else:
            a = self.dropout(a, inference=True)

        # Apply attention to values: (num_heads, head_dim, H*W)
        o = jnp.einsum("h q k, h d k -> h d q", a, v)

        # Reshape back to (C, H, W)
        o = o.reshape(self.dim, h, w)
        o = self.o_conv(o)
        return o


class UNet(eqx.Module):
    """Defines a general-purpose UNet model.

    Parameters:
        in_dim: Number of input dimensions.
        embed_dim: Embedding dimension.
        dim_scales: List of dimension scales.
        input_embedding_dim: The input embedding dimension, if an input
            embedding is used (for example, when conditioning on time, or some
            class embedding).

    Inputs:
        x: Input tensor of shape ``(batch_size, in_dim, height, width)``.
        embedding: Optional embedding tensor of shape ``(batch_size, input_embedding_dim)``.

    Outputs:
        x: Output tensor of shape ``(batch_size, in_dim, height, width)``.
    """

    init_embed: eqx.nn.Conv2d
    down_blocks: tuple[BasicBlock | eqx.nn.Conv2d, ...]
    mid_blocks: tuple[BasicBlock | SelfAttention2d, ...]
    up_blocks: tuple[BasicBlock | eqx.nn.ConvTranspose2d | eqx.nn.Conv2d, ...]
    out_blocks: tuple[BasicBlock | eqx.nn.Conv2d, ...]

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        dim_scales: Sequence[int],
        input_embedding_dim: int | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key_init, key = jax.random.split(key)
        self.init_embed = eqx.nn.Conv2d(in_dim, embed_dim, kernel_size=1, key=key_init)

        all_dims = (embed_dim, *[embed_dim * s for s in dim_scales])

        # Down blocks
        down_blocks_list: list[BasicBlock | eqx.nn.Conv2d] = []
        for idx, (in_c, out_c) in enumerate(zip(all_dims[:-1], all_dims[1:], strict=True)):
            is_last = idx == len(all_dims) - 2
            block_key1, block_key2, conv_key, key = jax.random.split(key, 4)

            down_blocks_list.append(BasicBlock(in_c, in_c, input_embedding_dim, key=block_key1))
            down_blocks_list.append(BasicBlock(in_c, in_c, input_embedding_dim, key=block_key2))
            if not is_last:
                conv = eqx.nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, key=conv_key)
                down_blocks_list.append(conv)
            else:
                conv = eqx.nn.Conv2d(in_c, out_c, kernel_size=1, key=conv_key)
                down_blocks_list.append(conv)

        self.down_blocks = tuple(down_blocks_list)

        # Mid blocks
        mid_key1, mid_key2, attn_key, key = jax.random.split(key, 4)
        self.mid_blocks = (
            BasicBlock(all_dims[-1], all_dims[-1], input_embedding_dim, key=mid_key1),
            SelfAttention2d(all_dims[-1], key=attn_key),
            BasicBlock(all_dims[-1], all_dims[-1], input_embedding_dim, key=mid_key2),
        )

        # Up blocks
        up_blocks_list: list[BasicBlock | eqx.nn.ConvTranspose2d | eqx.nn.Conv2d] = []
        for idx, (in_c, out_c, skip_c) in enumerate(
            zip(all_dims[::-1][:-1], all_dims[::-1][1:], all_dims[:-1][::-1], strict=True)
        ):
            is_last = idx == len(all_dims) - 2
            block_key1, block_key2, conv_key, key = jax.random.split(key, 4)

            up_blocks_list.append(BasicBlock(in_c + skip_c, in_c, input_embedding_dim, key=block_key1))
            up_blocks_list.append(BasicBlock(in_c + skip_c, in_c, input_embedding_dim, key=block_key2))
            if not is_last:
                conv_t = eqx.nn.ConvTranspose2d(in_c, out_c, kernel_size=(2, 2), stride=2, key=conv_key)
                up_blocks_list.append(conv_t)
            else:
                conv = eqx.nn.Conv2d(in_c, out_c, kernel_size=1, key=conv_key)
                up_blocks_list.append(conv)

        self.up_blocks = tuple(up_blocks_list)

        # Output blocks
        out_key1, out_key2, key = jax.random.split(key, 3)
        self.out_blocks = (
            BasicBlock(embed_dim, embed_dim, input_embedding_dim, key=out_key1),
            eqx.nn.Conv2d(embed_dim, in_dim, kernel_size=1, use_bias=True, key=out_key2),
        )

    def __call__(self, x: Array, embedding: Array | None = None, *, key: PRNGKeyArray | None = None) -> Array:
        """Forward pass.

        Args:
            x: Input tensor of shape (C, H, W)
            embedding: Optional embedding tensor of shape (embed_dim)
            key: Optional random key for dropout in attention

        Returns:
            Output tensor of shape (C, H, W)
        """
        x = self.init_embed(x)
        skip_conns: list[Array] = []
        residual = x

        # Down blocks
        for down_block in self.down_blocks:
            if isinstance(down_block, BasicBlock):
                x = down_block(x, embedding)
                skip_conns.append(x)
            else:
                x = down_block(x)

        # Mid blocks
        for mid_block in self.mid_blocks:
            if isinstance(mid_block, BasicBlock):
                x = mid_block(x, embedding)
            elif isinstance(mid_block, SelfAttention2d):
                x = mid_block(x, key=key)

        # Up blocks
        for up_block in self.up_blocks:
            if isinstance(up_block, BasicBlock):
                skip = skip_conns.pop()
                x = jax.image.resize(x, (x.shape[0], skip.shape[1], skip.shape[2]), method="linear")
                x = jnp.concatenate([x, skip], axis=0)  # Concatenate along channel dimension
                x = up_block(x, embedding)
            else:
                x = up_block(x)

        # Residual connection
        x = jax.image.resize(x, residual.shape, method="linear") + residual

        # Output blocks
        for out_block in self.out_blocks:
            if isinstance(out_block, BasicBlock):
                x = out_block(x, embedding)
            else:
                x = out_block(x)

        return x
