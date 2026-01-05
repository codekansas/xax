"""Smoke tests for the lightweight LLM implementations."""

import jax

import xax


def test_qwen3_forward_shape() -> None:
    config = xax.QWEN3_SMALL
    key = jax.random.key(0)
    tokens_bt = jax.random.randint(key, (2, 5), minval=0, maxval=config.vocab_size)
    model = xax.build_qwen3_model(config, key=key)
    logits_btv = model(tokens_bt)
    assert logits_btv.shape == (2, 5, config.vocab_size)


def test_tie_embedding_and_head() -> None:
    model = xax.build_qwen3_model(xax.QWEN3_SMALL, key=jax.random.key(42))
    tied = xax.tie_embedding_and_head(model)
    assert tied.lm_head.weight is tied.embed.weight


def test_generate_tokens_length() -> None:
    model = xax.build_qwen3_model(xax.QWEN3_SMALL, key=jax.random.key(0))
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
