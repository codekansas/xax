"""Smoke tests for the lightweight LLM implementations."""

import jax

import xax


def _forward_and_shape(config: object) -> None:
    key = jax.random.key(0)
    tokens_bt = jax.random.randint(key, (2, 5), minval=0, maxval=config.vocab_size)
    if config is xax.QWEN3_SMALL:
        model = xax.build_qwen3_model(config, key=key)
    elif config is xax.GPT_OSS_SMALL:
        model = xax.build_gpt_oss_model(config, key=key)
    else:
        model = xax.build_deepseek_r1_model(config, key=key)
    logits_btv = model(tokens_bt)
    assert logits_btv.shape == (2, 5, config.vocab_size)


def test_qwen3_forward_shape() -> None:
    _forward_and_shape(xax.QWEN3_SMALL)


def test_gpt_oss_forward_shape() -> None:
    _forward_and_shape(xax.GPT_OSS_SMALL)


def test_deepseek_forward_shape() -> None:
    _forward_and_shape(xax.DEEPSEEK_R1_SMALL)


def test_tie_embedding_and_head() -> None:
    model = xax.build_qwen3_model(xax.QWEN3_SMALL, key=jax.random.key(42))
    tied = xax.tie_embedding_and_head(model)
    assert tied.lm_head.weight is tied.embed.weight


class _DummyTokenizer:
    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, list[list[int]]]:
        ids = [[ord(c) % 100 for c in text]]
        return {"input_ids": ids}

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        return "".join(chr((t % 26) + 97) for t in tokens)


def test_generate_tokens_length() -> None:
    tokenizer = _DummyTokenizer()
    model = xax.build_qwen3_model(xax.QWEN3_SMALL, key=jax.random.key(0))
    prompt = "hi"
    max_new = 3
    text = xax.generate(model, tokenizer, prompt, max_new_tokens=max_new, chat_template=False)
    assert isinstance(text, str)
    assert len(text) == len(prompt) + max_new
