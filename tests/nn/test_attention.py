"""Tests for attention mechanisms."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import xax


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_self_attention_block_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    block = xax.SelfAttentionBlock(
        embed_dim=32,
        num_heads=2,
        key=subkey,
        causal=True,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    def scan_fn(
        carry: tuple[Array, xax.AttentionCache],
        _: Array,
    ) -> tuple[tuple[Array, xax.AttentionCache], Array]:
        x, cache = carry
        x, cache, _fp8 = block.forward(x, cache=cache)
        return (x, cache), x

    # Gets a random starting vector.
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1, block.embed_dim))

    # Autoregressive unrolling.
    tsz = 10
    cache = block.init_cache(x.dtype)
    _, xs = xax.scan(scan_fn, (x, cache), length=tsz, jit_level=-1)
    xs = xs.squeeze(1)
    prev_xs = jnp.concatenate([x, xs[:-1]], axis=0)

    # Calls the batched forward function.
    mask = block.init_mask(tsz, add_cache=True)
    next_xs, _, _ = block.forward(prev_xs, cache=cache, mask=mask)

    assert jnp.allclose(xs, next_xs, atol=1e-3)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_self_attention_block_mask(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    block = xax.SelfAttentionBlock(
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
def test_transformer_block_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    block = xax.TransformerBlock(
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        key=subkey,
        cross_attention=True,
        causal=True,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    def scan_fn(
        carry: tuple[Array, xax.TransformerBlockCache],
        _: Array,
    ) -> tuple[tuple[Array, xax.TransformerBlockCache], Array]:
        x, cache = carry
        x, cache = block.forward(x, cache=cache)
        return (x, cache), x

    # Gets a random starting vector.
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1, block.embed_dim))

    # Gets a random context vector.
    key, subkey = jax.random.split(key)
    context_sn = jax.random.normal(subkey, (3, block.embed_dim))

    # Autoregressive unrolling.
    cache = block.init_cache(x.dtype, context_sn=context_sn)
    _, xs = xax.scan(scan_fn, (x, cache), length=10)
    xs = xs.squeeze(1)
    prev_xs = jnp.concatenate([x, xs[:-1]], axis=0)

    # Calls the batched forward function.
    mask = block.init_mask(10, add_cache=True)
    next_xs, _ = block.forward(prev_xs, context_sn=context_sn, cache=cache, mask=mask)

    assert jnp.allclose(xs, next_xs, atol=1e-3)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_transformer_stack_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    stack = xax.TransformerStack(
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        num_layers=3,
        key=subkey,
        cross_attention=True,
        causal=True,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    def scan_fn(
        carry: tuple[Array, xax.TransformerCache],
        _: Array,
    ) -> tuple[tuple[Array, xax.TransformerCache], Array]:
        x, cache = carry
        x, cache = stack.forward(x, cache=cache)
        return (x, cache), x

    # Gets a random starting vector.
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1, stack.layers[0].embed_dim))

    # Gets a random context vector.
    key, subkey = jax.random.split(key)
    context_sn = jax.random.normal(subkey, (3, stack.layers[0].embed_dim))

    # Autoregressive unrolling.
    cache = stack.init_cache(x.dtype, x_tn=context_sn)
    _, xs = xax.scan(scan_fn, (x, cache), length=10)
    xs = xs.squeeze(1)
    prev_xs = jnp.concatenate([x, xs[:-1]], axis=0)

    # Calls the batched forward function.
    mask = stack.init_mask(10, add_cache=True)
    next_xs, _ = stack.forward(prev_xs, context_sn=context_sn, cache=cache, mask=mask)

    assert jnp.allclose(xs, next_xs, atol=1e-3)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_transformer_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    transformer = xax.Transformer(
        vocab_size=1000,
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        num_layers=2,
        key=subkey,
        cross_attention=False,
        causal=True,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    # Generates a random sequence.
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (5,), 0, 1000)
    seq = transformer.generate_sequence(x, max_len=10, top_k=1)

    # Checks that the next token prediction matches the generated sequence.
    cache = transformer.init_cache(x.dtype)
    mask = transformer.init_mask(5, add_cache=True)
    _, cache = transformer.encode(x, cache=cache, mask=mask)
    mask = transformer.init_mask(10, add_cache=True)
    next_x, _ = transformer.forward(seq[4:-1], cache=cache, mask=mask)
    pseq = next_x.argmax(axis=-1)

    assert jnp.allclose(pseq, seq[5:])
