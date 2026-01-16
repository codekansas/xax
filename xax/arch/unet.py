"""Defines a general-purpose UNet model."""

import math
from typing import Callable, Sequence

import chex
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
        chex.assert_shape(x, (self.linear1.in_features,))

        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


def fourier_embeddings_2d(h: int, w: int, dim: int, max_period: int) -> Array:
    """Generate 2D Fourier embeddings for a spatial grid.

    Args:
        h: Height of the spatial grid
        w: Width of the spatial grid
        dim: Dimension of the embeddings
        max_period: Maximum period for the embeddings

    Returns:
        Embeddings of shape (h, w, dim)
    """
    y_coords = jnp.arange(h, dtype=jnp.float32)
    x_coords = jnp.arange(w, dtype=jnp.float32)
    y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing="ij")

    half_dim = dim // 4
    freqs = jnp.exp(-jnp.arange(0, half_dim, dtype=jnp.float32) * math.log(max_period) / half_dim)

    y_args = y_grid[..., None] * freqs[None, None, :]
    x_args = x_grid[..., None] * freqs[None, None, :]

    y_emb = jnp.concatenate([jnp.sin(y_args), jnp.cos(y_args)], axis=-1)
    x_emb = jnp.concatenate([jnp.sin(x_args), jnp.cos(x_args)], axis=-1)

    embedding = jnp.concatenate([y_emb, x_emb], axis=-1)

    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros((h, w, 1))], axis=-1)

    return embedding


class BasicBlock(eqx.Module):
    """Basic residual block with optional embedding conditioning."""

    in_c: int
    out_c: int
    embed_c: int | None
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
        self.in_c = in_c
        self.out_c = out_c
        self.embed_c = embed_c

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
        chex.assert_shape(x, (self.in_c, None, None))

        out = self.conv1(x)
        out = self.norm1(out)

        if embedding is not None:
            if self.mlp_emb is None:
                raise ValueError("Embedding was provided but embedding projection is None")
            chex.assert_shape(embedding, (self.embed_c,))
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
    emb_dims: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        emb_dims: int = 128,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.emb_dims = emb_dims
        key_q, key_k, key_v, key_o = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(dim + emb_dims, dim, use_bias=True, key=key_q)
        self.k_proj = eqx.nn.Linear(dim + emb_dims, dim, use_bias=True, key=key_k)
        self.v_proj = eqx.nn.Linear(dim + emb_dims, dim, use_bias=True, key=key_v)
        self.o_proj = eqx.nn.Linear(dim, dim, use_bias=True, key=key_o)

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input tensor of shape (C, H, W)

        Returns:
            Output tensor of shape (C, H, W)
        """
        chex.assert_shape(x, (self.dim, None, None))

        _, h, w = x.shape
        head_dim = self.dim // self.num_heads

        emb = fourier_embeddings_2d(h, w, self.emb_dims, max_period=max(h, w) * 2).astype(x.dtype)
        i = jnp.concatenate([x.transpose(1, 2, 0), emb], axis=-1).reshape(h * w, self.dim + self.emb_dims)

        q = jax.vmap(self.q_proj)(i).reshape(h * w, self.num_heads, head_dim)
        k = jax.vmap(self.k_proj)(i).reshape(h * w, self.num_heads, head_dim)
        v = jax.vmap(self.v_proj)(i).reshape(h * w, self.num_heads, head_dim)

        # Compute attention.
        a = jax.nn.dot_product_attention(q, k, v).reshape(h * w, self.dim)
        o = jax.vmap(self.o_proj)(a).reshape(h, w, self.dim).transpose(2, 0, 1)

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

    in_dim: int
    embed_dim: int
    input_embedding_dim: int | None
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
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.input_embedding_dim = input_embedding_dim

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
            zip(
                all_dims[::-1][:-1],
                all_dims[::-1][1:],
                all_dims[:-1][::-1],
                strict=True,
            )
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
        chex.assert_shape(x, (self.in_dim, None, None))
        if embedding is not None:
            chex.assert_shape(embedding, (self.input_embedding_dim,))

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
                x = mid_block(x)

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
