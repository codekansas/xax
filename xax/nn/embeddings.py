"""Defines embedding layers."""

import math
from typing import Literal, cast, get_args, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, DTypeLike, PRNGKeyArray

EmbeddingKind = Literal["identity", "learned", "sinusoidal", "rotary"]


def cast_embedding_kind(k: str) -> EmbeddingKind:
    args = get_args(EmbeddingKind)
    assert k in args, f"Invalid initialization type: '{k}' Valid options are {args}"
    return cast(EmbeddingKind, k)


class IdentityPositionalEmbeddings(eqx.Module):
    def __call__(self, x: Array, offset: int = 0, times_t: Array | None = None) -> Array:
        return x


class LearnedPositionalEmbeddings(eqx.Module):
    """Defines a learned embeddings module.

    Parameters:
        max_tsz: The maximum sequence length.
        embed_dim: The embedding dimension.
        weight_init: The initialization type for the embedding weight.
        learnable: Whether the embeddings are learnable.
    """

    max_tsz: int = eqx.field()
    embed_dim: int = eqx.field()
    learnable: bool = eqx.field()
    embeddings_tc: Array = eqx.field()

    def __init__(
        self,
        max_tsz: int,
        embed_dim: int,
        learnable: bool = True,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.max_tsz = max_tsz
        self.embed_dim = embed_dim
        self.learnable = learnable

        self.embeddings_tc = jrandom.normal(key, (max_tsz, embed_dim))

    def __call__(self, x_tc: Array, offset: int = 0, times_t: Array | None = None) -> Array:
        if times_t is None:
            emb_tc = self.embeddings_tc[offset : offset + x_tc.shape[-2]]
        else:
            emb_tc = self.embeddings_tc[times_t]
        if not self.learnable:
            emb_tc = jax.lax.stop_gradient(emb_tc)
        return emb_tc


class SinusoidalEmbeddings(eqx.Module):
    """Defines a sinusoidal embeddings module.

    Parameters:
        embed_dim: The embedding dimension.
        max_tsz: The maximum sequence length.
        learnable: Whether the embeddings are learnable.
        base: The base for the sinusoidal embeddings.
    """

    base: int = eqx.field()
    max_tsz: int | None = eqx.field()
    embed_dim: int | None = eqx.field()
    embeddings_tc: Array | None = eqx.field()

    def __init__(
        self,
        embed_dim: int | None = None,
        max_tsz: int | None = None,
        learnable: bool = True,
        base: int = 10_000,
    ) -> None:
        super().__init__()

        self.max_tsz = max_tsz
        self.embed_dim = embed_dim
        self.base = base
        self.embeddings_tc = None

        if learnable:
            assert max_tsz is not None, "Learnable parameters require `max_tsz` to be set"
            assert embed_dim is not None, "Learnable parameters require `embed_dim` to be set"
            self.embeddings_tc = self.get_embeddings(max_tsz, embed_dim)

    def __call__(self, x_tc: Array, offset: int = 0, times_t: Array | None = None) -> Array:
        tsz, dims = x_tc.shape

        # If the embeddings are learnable, use the property.
        if self.embeddings_tc is None:
            if times_t is None:
                embeddings_tc = self.get_embeddings(offset + tsz, dims, x_tc.dtype)
            else:
                embeddings_tc = self.get_embeddings(times_t.max().item() + 1, dims, x_tc.dtype)
        else:
            embeddings_tc = self.embeddings_tc

        # Get only the embeddings for the specified time steps.
        if times_t is None:
            embeddings_tc = embeddings_tc[offset : offset + tsz]
        else:
            embeddings_tc = embeddings_tc[times_t]

        return x_tc + embeddings_tc

    def get_embeddings(
        self,
        tsz: int,
        embed_dim: int,
        dtype: DTypeLike | None = None,
    ) -> Array:
        positions_t = jax.numpy.arange(tsz, dtype=dtype)
        dim_d = jax.numpy.arange(embed_dim, dtype=dtype)
        dim_d = self.base ** (2 * (dim_d // 2) / embed_dim)
        embeddings_td = positions_t[:, None] / dim_d[None, :]
        embeddings_td = jnp.concatenate(
            [jax.numpy.sin(embeddings_td[:, 0::2]), jax.numpy.cos(embeddings_td[:, 1::2])],
            axis=-1,
        )
        return embeddings_td.astype(dtype)


def get_rotary_embeddings(
    tsz: int,
    embed_dim: int,
    dtype: jnp.dtype,
    offset: int = 0,
    base: int = 10_000,
) -> Array:
    assert embed_dim % 4 == 0, f"Embedding dimension must be divisible by 4, got {embed_dim}"
    half_d = embed_dim // 2
    theta = 1.0 / (base ** (jnp.arange(0, half_d, 2, dtype=jnp.float32) / half_d))
    seq_idx = jnp.arange(offset, tsz + offset, dtype=jnp.float32)
    idx_theta_tc = jnp.einsum("t,c->tc", seq_idx, theta)
    idx_theta2_tc = jnp.concatenate([idx_theta_tc, idx_theta_tc], axis=1)
    cos_tc, sin_tc = jnp.cos(idx_theta2_tc), jnp.sin(idx_theta2_tc)
    emb_2tc = jnp.stack((cos_tc, sin_tc), axis=0)
    return emb_2tc.astype(dtype)


def apply_rotary_embeddings(x_tc: Array, embs_2tc: Array, offset: int = 0, times_t: Array | None = None) -> Array:
    cos_tc, sin_tc = embs_2tc[0], embs_2tc[1]
    tsz, embed_dim = x_tc.shape
    half_d = embed_dim // 2
    quarter_d = embed_dim // 4
    x_rope_tc, x_pass_tc = x_tc[..., :half_d], x_tc[..., half_d:]
    neg_half_x_tc = jnp.concatenate([-x_rope_tc[..., quarter_d:], x_rope_tc[..., :quarter_d]], axis=-1)
    cos_part_tc = cos_tc[offset : offset + tsz] if times_t is None else cos_tc[times_t]
    sin_part_tc = sin_tc[offset : offset + tsz] if times_t is None else sin_tc[times_t]
    x_rope_tc = x_rope_tc * cos_part_tc + neg_half_x_tc * sin_part_tc
    return jnp.concatenate((x_rope_tc, x_pass_tc), axis=-1)


def rotary_embeddings(x_tc: Array, offset: int = 0, base: int = 10_000) -> Array:
    """Defines a single function for applying rotary embeddings.

    This is slower than using the module, but it doesn't require
    pre-initializing the embeddings, so it can be used when running online.

    Args:
        x_tc: The input tensor, with shape ``(batch, tsz, embed_dim)``.
        offset: The offset for the first element.
        base: The base for the sinusoidal embeddings.

    Returns:
        The input tensor with rotary embeddings applied.
    """
    (tsz, embed_dim), dtype = x_tc.shape, x_tc.dtype
    emb_2tc = get_rotary_embeddings(tsz + offset, embed_dim, dtype, 0, base)
    return apply_rotary_embeddings(x_tc, emb_2tc, offset)


class RotaryEmbeddings(eqx.Module):
    """Defines a rotary embeddings module.

    Parameters:
        base: The base for the sinusoidal embeddings.
    """

    base: int = eqx.field()

    def __init__(self, base: int = 10_000) -> None:
        """Defines a rotary embeddings module.

        Args:
            base: The base for the sinusoidal embeddings.
        """
        super().__init__()

        self.base = base

    def __call__(self, x_tc: Array, offset: int = 0, times_t: Array | None = None) -> Array:
        tsz, embed_dim = x_tc.shape
        max_tsz = max(tsz, 0 if times_t is None else int(times_t.max().item()) + 1) + offset
        emb_2tc = get_rotary_embeddings(max_tsz, embed_dim, x_tc.dtype, 0, self.base)
        return apply_rotary_embeddings(x_tc, emb_2tc, offset, times_t)


@overload
def get_positional_embeddings(kind: Literal["identity"]) -> IdentityPositionalEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: Literal["learned"],
    *,
    max_tsz: int,
    embed_dim: int,
    learnable: bool | None = None,
    key: PRNGKeyArray,
) -> LearnedPositionalEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: Literal["sinusoidal"],
    *,
    max_tsz: int | None = None,
    embed_dim: int | None = None,
    learnable: bool | None = None,
    base: int = 10_000,
) -> SinusoidalEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: Literal["rotary"],
    *,
    base: int = 10_000,
) -> RotaryEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: EmbeddingKind,
    *,
    max_tsz: int | None = None,
    embed_dim: int | None = None,
    learnable: bool | None = None,
    base: int = 10_000,
    key: PRNGKeyArray | None = None,
) -> IdentityPositionalEmbeddings | LearnedPositionalEmbeddings | SinusoidalEmbeddings | RotaryEmbeddings: ...


def get_positional_embeddings(
    kind: EmbeddingKind,
    *,
    max_tsz: int | None = None,
    embed_dim: int | None = None,
    learnable: bool | None = None,
    base: int = 10_000,
    key: PRNGKeyArray | None = None,
) -> eqx.Module:
    """Defines the common module for adding positional embeddings.

    Args:
        kind: The type of embedding to use.
        max_tsz: The maximum sequence length.
        embed_dim: The embedding dimension.
        learnable: Whether the embeddings are learnable; if not provided,
            uses sensible defaults.
        base: The base for the sinusoidal embeddings.
        key: The PRNG key for initializing learnable embeddings.

    Returns:
        The positional embeddings module.

    Raises:
        ValueError: If an invalid embedding kind is supplied.
    """
    match kind:
        case "identity":
            return IdentityPositionalEmbeddings()

        case "learned":
            assert max_tsz is not None, "Learned embeddings require `max_tsz` to be set"
            assert embed_dim is not None, "Learned embeddings require `embed_dim` to be set"
            assert key is not None, "Learned embeddings require `key` to be set"

            return LearnedPositionalEmbeddings(
                max_tsz=max_tsz,
                embed_dim=embed_dim,
                learnable=True if learnable is None else learnable,
                key=key,
            )

        case "sinusoidal":
            return SinusoidalEmbeddings(
                max_tsz=max_tsz,
                embed_dim=embed_dim,
                learnable=False if learnable is None else learnable,
                base=base,
            )

        case "rotary":
            return RotaryEmbeddings(base=base)

        case _:
            raise ValueError(f"Invalid embedding kind: {kind}")


def fourier_embeddings(t: Array, dim: int, max_period: int = 10000) -> Array:
    if squeeze := t.ndim == 0:
        t = t[None]
    half = dim // 2
    idxs = jnp.arange(start=0, stop=half, dtype=jnp.float32)
    freqs = jnp.exp(-math.log(max_period) * idxs / half)
    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    # Adds an additional row of zeros to match the expected dimension.
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    if squeeze:
        embedding = embedding[0]
    return embedding


class FourierEmbeddings(eqx.Module):
    """Defines a module for applying Fourier embeddings to timesteps.

    This module differs from the other positional embedding modules because it
    expects a continuous time input, rather than a discrete time input.

    Parameters:
        dim: The number of embedding dimensions. This value is used to determine
            how many different frequencies to use, and a higher value means
            higher frequencies.
        max_period: The maximum period for the embeddings. This should roughly
            be in line with the maximum number of timesteps; the default value
            of 10,000 is commonly used in NLP applications, and is derived from
            operating on sequence lengths of 100 to 1000 tokens.
    """

    dim: int
    max_period: int

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()

        self.dim = dim
        self.max_period = max_period

    def __call__(self, t: Array) -> Array:
        return fourier_embeddings(t, self.dim, self.max_period)
