"""Smoke tests for the lightweight LLM implementations."""

import jax
import jax.numpy as jnp

import xax

# Small config for testing - minimal size to keep tests fast
TEST_LLM_CONFIG = xax.LLMConfig(
    vocab_size=256,
    embed_dim=64,
    q_heads=4,
    kv_heads=2,
    head_dim=16,
    num_layers=2,
    max_tsz=128,
)


def test_qwen3_forward_shape() -> None:
    config = TEST_LLM_CONFIG
    key = jax.random.key(0)
    tokens_bt = jax.random.randint(key, (2, 5), minval=0, maxval=config.vocab_size)
    model = xax.LLM.build(config, key=key)
    logits_btv = jax.vmap(model.forward)(tokens_bt)
    assert logits_btv.shape == (2, 5, config.vocab_size)


def test_tie_embedding_and_head() -> None:
    model = xax.LLM.build(TEST_LLM_CONFIG, key=jax.random.key(42))
    tied = xax.tie_embedding_and_head(model)
    assert tied.lm_head.weight is tied.embed.weight


def test_generate_tokens_length() -> None:
    model = xax.LLM.build(TEST_LLM_CONFIG, key=jax.random.key(0))
    tokens = [ord("h"), ord("i")]
    eos_id = ord("!")
    max_new = 3
    output_tokens = xax.llm_generate(
        model,
        tokens,
        eos_id,
        max_new_tokens=max_new,
    )
    assert len(output_tokens) == len(tokens) + max_new


def _reference_llm_generate(
    model: xax.LLM,
    tokens_t: jax.Array,
    *,
    eos_id: int,
    max_new_tokens: int,
    context_tn: jax.Array | None,
) -> tuple[jax.Array, int]:
    initial_len = int(tokens_t.shape[0])
    max_len = initial_len + max_new_tokens
    padded_tokens_t = jnp.zeros((max_len,), dtype=jnp.int32).at[:initial_len].set(tokens_t)
    caches = model.init_cache(max_len, dtype=model.embed.weight.dtype)

    prompt_context_tn = None if context_tn is None else context_tn[:initial_len]
    prompt_logits_tv, caches = model.forward(tokens_t, context_tn=prompt_context_tn, caches=caches)
    next_token = int(jnp.argmax(prompt_logits_tv[-1]).item())

    padded_tokens_t = padded_tokens_t.at[initial_len].set(next_token)
    final_len = initial_len + 1
    done = eos_id >= 0 and next_token == eos_id

    while final_len < max_len and not done:
        prev_token_1 = padded_tokens_t[final_len - 1 : final_len]
        step_context_1n = None if context_tn is None else context_tn[final_len - 1 : final_len]
        logits_1v, caches = model.forward(prev_token_1, context_tn=step_context_1n, caches=caches)
        next_token = int(jnp.argmax(logits_1v[-1]).item())
        padded_tokens_t = padded_tokens_t.at[final_len].set(next_token)
        final_len += 1
        done = eos_id >= 0 and next_token == eos_id

    return padded_tokens_t, final_len


def test_llm_generate_jit_matches_reference_cache_decode_with_context() -> None:
    model = xax.LLM.build(TEST_LLM_CONFIG, key=jax.random.key(0))
    tokens_t = jnp.array([11, 17, 23], dtype=jnp.int32)
    max_new_tokens = 6
    context_key = jax.random.key(3)
    context_tn = jax.random.normal(
        context_key,
        (tokens_t.shape[0] + max_new_tokens, TEST_LLM_CONFIG.embed_dim),
        dtype=model.embed.weight.dtype,
    )

    generated_t, generated_len = xax.llm_generate_jit(
        model,
        tokens_t=tokens_t,
        eos_id=-1,
        max_new_tokens=max_new_tokens,
        context_tn=context_tn,
        temperature=0.0,
        top_p=1.0,
        key=jax.random.key(7),
    )
    ref_t, ref_len = _reference_llm_generate(
        model,
        tokens_t=tokens_t,
        eos_id=-1,
        max_new_tokens=max_new_tokens,
        context_tn=context_tn,
    )

    assert int(generated_len) == ref_len
    assert jnp.array_equal(generated_t, ref_t)


def test_llm_generate_jit_respects_allowed_token_range() -> None:
    model = xax.LLM.build(TEST_LLM_CONFIG, key=jax.random.key(0))
    tokens_t = jnp.array([5, 9, 13], dtype=jnp.int32)
    max_new_tokens = 12
    allowed_min_id = 32
    allowed_max_id = 48

    generated_t, generated_len = xax.llm_generate_jit(
        model,
        tokens_t=tokens_t,
        eos_id=-1,
        max_new_tokens=max_new_tokens,
        context_tn=None,
        temperature=0.0,
        top_p=1.0,
        key=jax.random.key(11),
        allowed_token_range=(allowed_min_id, allowed_max_id),
    )

    final_len = int(generated_len)
    new_tokens_t = generated_t[tokens_t.shape[0] : final_len]
    assert new_tokens_t.shape[0] == max_new_tokens
    assert bool(jnp.all((new_tokens_t >= allowed_min_id) & (new_tokens_t < allowed_max_id)))


def test_llm_generate_jit_delays_eos_until_min_tokens_generated() -> None:
    model = xax.LLM.build(TEST_LLM_CONFIG, key=jax.random.key(0))
    tokens_t = jnp.array([5, 9, 13], dtype=jnp.int32)
    max_new_tokens = 4

    max_len = int(tokens_t.shape[0] + max_new_tokens)
    caches = model.init_cache(max_len, dtype=model.embed.weight.dtype)
    prompt_logits_tv, _ = model.forward(tokens_t, context_tn=None, caches=caches)
    eos_id = int(jnp.argmax(prompt_logits_tv[-1]).item())

    no_delay_t, no_delay_len = xax.llm_generate_jit(
        model,
        tokens_t=tokens_t,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        context_tn=None,
        temperature=0.0,
        top_p=1.0,
        key=jax.random.key(12),
    )
    assert int(no_delay_len) == tokens_t.shape[0] + 1
    assert int(no_delay_t[tokens_t.shape[0]]) == eos_id

    delayed_t, delayed_len = xax.llm_generate_jit(
        model,
        tokens_t=tokens_t,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        context_tn=None,
        temperature=0.0,
        top_p=1.0,
        key=jax.random.key(12),
        min_new_tokens_before_eos=2,
    )
    generated_t = delayed_t[tokens_t.shape[0] : int(delayed_len)]
    assert generated_t.shape[0] >= 2
    assert int(generated_t[0]) != eos_id
    assert int(generated_t[1]) != eos_id
