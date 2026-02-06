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


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_self_attention_block_batched_cache(use_rotary_embeddings: bool) -> None:
    """Test caching with multi-token batches (not just single tokens)."""
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)

    tsz = 8
    block = xax.SelfAttentionBlock.build(
        embed_dim=32,
        num_heads=2,
        key=subkey,
        causal=True,
        context_length=tsz + 1,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (tsz, block.embed_dim))

    # Method 1: Process in batches of varying sizes with cache
    cache = block.init_cache(dtype=inputs.dtype)
    out1, cache, _ = block.forward(inputs[0:3], cache=cache)  # First 3 tokens
    out2, cache, _ = block.forward(inputs[3:5], cache=cache)  # Next 2 tokens
    out3, cache, _ = block.forward(inputs[5:8], cache=cache)  # Last 3 tokens
    batched_with_cache = jnp.concatenate([out1, out2, out3], axis=0)

    # Method 2: Single batched forward with causal mask (no cache)
    mask = block.init_mask(tsz)
    batched_no_cache, _, _ = block.forward(inputs, mask=mask)

    assert jnp.allclose(batched_with_cache, batched_no_cache, atol=1e-4)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_cross_attention_block_cache(use_rotary_embeddings: bool) -> None:
    """Test that cached cross-attention matches fresh computation."""
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    embed_dim = 32
    block = xax.CrossAttentionBlock.build(
        embed_dim=embed_dim,
        num_heads=2,
        key=subkey,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    # Generate random query and key/value sequences
    key, k1, k2 = jax.random.split(key, 3)
    q_seq_len, kv_seq_len = 4, 8
    q_tn = jax.random.normal(k1, (q_seq_len, embed_dim))
    kv_sn = jax.random.normal(k2, (kv_seq_len, embed_dim))

    # Method 1: Fresh computation (no cache)
    out_fresh, _, _ = block.forward(q_tn, kv_sn=kv_sn)

    # Method 2: Using cache - first init, then forward with cache
    cache, _ = block.init_cache(kv_sn)
    out_cached, _, _ = block.forward(q_tn, cache=cache)

    assert jnp.allclose(out_fresh, out_cached, atol=1e-5)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_cross_attention_block_cache_reuse(use_rotary_embeddings: bool) -> None:
    """Test that reusing cache multiple times produces consistent results.

    This specifically tests for the RoPE double-application bug where RoPE
    was being applied to cached keys on each forward pass.
    """
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    embed_dim = 32
    block = xax.CrossAttentionBlock.build(
        embed_dim=embed_dim,
        num_heads=2,
        key=subkey,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    # Generate random sequences
    key, k1, k2, k3 = jax.random.split(key, 4)
    kv_seq_len = 8
    kv_sn = jax.random.normal(k1, (kv_seq_len, embed_dim))
    q1_tn = jax.random.normal(k2, (3, embed_dim))
    q2_tn = jax.random.normal(k3, (2, embed_dim))

    # Initialize cache once
    cache, _ = block.init_cache(kv_sn)

    # First forward with cache
    out1, cache1, _ = block.forward(q1_tn, cache=cache)

    # Second forward reusing the returned cache
    out2, cache2, _ = block.forward(q2_tn, cache=cache1)

    # Third forward - recompute from original cache to verify consistency
    out1_again, _, _ = block.forward(q1_tn, cache=cache)

    # The first query should produce the same output both times
    assert jnp.allclose(out1, out1_again, atol=1e-5)

    # Also verify against fresh computation
    out1_fresh, _, _ = block.forward(q1_tn, kv_sn=kv_sn)
    assert jnp.allclose(out1, out1_fresh, atol=1e-5)
