"""Smoke tests for the lightweight LLM implementations."""

import jax

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
    model = xax.build_pretrained_model(config, key=key)
    logits_btv = model(tokens_bt)
    assert logits_btv.shape == (2, 5, config.vocab_size)


def test_tie_embedding_and_head() -> None:
    model = xax.build_pretrained_model(TEST_LLM_CONFIG, key=jax.random.key(42))
    tied = xax.tie_embedding_and_head(model)
    assert tied.lm_head.weight is tied.embed.weight


def test_generate_tokens_length() -> None:
    model = xax.build_pretrained_model(TEST_LLM_CONFIG, key=jax.random.key(0))
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
