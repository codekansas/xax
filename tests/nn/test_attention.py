"""Tests for attention mechanisms."""

import jax
import jax.numpy as jnp
import pytest

import xax


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_self_attention_block_loopback(use_rotary_embeddings: bool) -> None:
    """Test that autoregressive (cached) forward matches batched forward with causal mask."""
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    tsz = 5

    block = xax.SelfAttentionBlock.build(
        embed_dim=32,
        num_heads=2,
        key=subkey,
        causal=True,
        context_length=tsz + 1,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    # Generate random input sequence
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (tsz, block.embed_dim))

    # Method 1: Autoregressive with cache - process one token at a time
    cache = block.init_cache(dtype=inputs.dtype)
    autoregressive_outputs = []
    for i in range(tsz):
        out, cache, _ = block.forward(inputs[i : i + 1], cache=cache)
        autoregressive_outputs.append(out[0])
    autoregressive_outputs = jnp.stack(autoregressive_outputs)

    # Method 2: Batched forward with causal mask (no cache)
    mask = block.init_mask(tsz)
    batched_outputs, _, _ = block.forward(inputs, mask=mask)

    assert jnp.allclose(autoregressive_outputs, batched_outputs, atol=1e-4)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_self_attention_block_mask(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    block = xax.SelfAttentionBlock.build(
        embed_dim=32,
        num_heads=2,
        key=subkey,
        causal=True,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    tsz = 10
    mask = block.init_mask(tsz)

    # Checks that the forward pass matches the autoregressive unrolling pass.
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (tsz, block.embed_dim))
    out_b, _, _ = block.forward(x)
    out_a, _, _ = block.forward(x, mask=mask)

    assert jnp.allclose(out_a, out_b, atol=1e-4)
