"""Tests the embeddings API."""

from typing import get_args

import jax.numpy as jnp
import jax.random as jrandom
import pytest

import xax


@pytest.mark.parametrize("kind", get_args(xax.EmbeddingKind))
def test_embeddings_api(kind: xax.EmbeddingKind) -> None:
    key = jrandom.key(0)
    xkey, pkey = jrandom.split(key)
    x_tc = jrandom.normal(xkey, (5, 8), dtype=jnp.float32)
    times_t = jnp.arange(1, 6)
    emb = xax.get_positional_embeddings(kind, max_tsz=12, embed_dim=8, key=pkey)
    y1_tc = emb(x_tc, times_t=times_t)
    y2_tc = emb(x_tc, offset=1)
    assert y1_tc.shape == (5, 8)
    assert y2_tc.shape == (5, 8)
    assert jnp.allclose(y1_tc, y2_tc)


@pytest.mark.parametrize("offset", [0, 12])
def test_rotary_embeddings_inference(offset: int) -> None:
    key = jrandom.key(0)
    x_tc = jrandom.normal(key, (5, 8), dtype=jnp.float32)
    emb = xax.get_positional_embeddings("rotary", max_tsz=8 + offset, embed_dim=8, learnable=False)
    y1_tc = emb(x_tc, offset=offset)
    y2_tc = xax.rotary_embeddings(x_tc, offset=offset)
    assert y1_tc.shape == (5, 8)
    assert y2_tc.shape == (5, 8)
    assert jnp.allclose(y1_tc, y2_tc)


def test_fourier_embeddings() -> None:
    emb = xax.FourierEmbeddings(8)
    times_t = jnp.arange(1, 6)
    y_t = emb(times_t)
    assert y_t.shape == (5, 8)
