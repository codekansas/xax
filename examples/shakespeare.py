#!/usr/bin/env -S uv run --no-project --script
"""LoRA fine-tuning of a pre-trained LLM on Shakespeare text."""

from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import jax.numpy as jnp
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

import xax
from xax.arch.llm import (
    LLM,
    QWEN3_SMALL,
    build_qwen3_model,
    hf_config_to_llm_config,
    load_hf_config,
    load_hf_weights_into_llm,
)
from xax.nn.lora import loraize


class Batch(TypedDict):
    attention_mask: Array
    input_ids: Array


@dataclass
class Config(xax.SupervisedConfig):
    # Model settings
    model_repo: str = xax.field("Qwen/Qwen3-0.6B", help="HuggingFace model repository")

    # LoRA settings
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(32.0, help="LoRA scaling factor")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")

    # Training settings
    learning_rate: float = xax.field(1e-4, help="Peak learning rate")
    min_learning_rate: float = xax.field(1e-5, help="Minimum learning rate for cosine decay")
    warmup_steps: int = xax.field(100, help="Number of warmup steps")
    sequence_length: int = xax.field(256, help="Maximum sequence length")


class ShakespeareLora(xax.SupervisedTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(config.model_repo)
        self._llm_config = hf_config_to_llm_config(
            load_hf_config(config.model_repo),
            base=QWEN3_SMALL,
        )

    @property
    def vocab_size(self) -> int:
        return self._llm_config.vocab_size

    def get_model(self, params: xax.InitParams) -> LLM:
        # Build model with correct config
        model = build_qwen3_model(self._llm_config, key=params.key)

        # Load pre-trained weights
        model = load_hf_weights_into_llm(model, self.config.model_repo)

        # Apply LoRA to all linear layers (attention and MLP projections)
        # Type assertion: loraize preserves the model structure
        model = loraize(
            model,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout_rate=self.config.lora_dropout,
            key=params.key,
        )
        assert isinstance(model, LLM)

        return model

    def get_optimizer(self) -> xax.Optimizer:
        # Create learning rate schedule with warmup and cosine decay
        if self.config.max_steps is not None:
            warmup_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=self.config.learning_rate,
                transition_steps=self.config.warmup_steps,
            )
            cosine_schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=max(self.config.max_steps - self.config.warmup_steps, 1),
                alpha=self.config.min_learning_rate / self.config.learning_rate,
            )
            learning_rate_schedule = optax.join_schedules(
                schedules=[warmup_schedule, cosine_schedule],
                boundaries=[self.config.warmup_steps],
            )
        else:
            learning_rate_schedule = self.config.learning_rate

        return optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=0.01,
        )

    def get_output(self, model: LLM, batch: Batch, state: xax.State, key: PRNGKeyArray) -> Array:
        # LLM takes (batch, seq_len) and returns (batch, seq_len, vocab_size)
        # Use all but last token as input
        input_ids_bt = batch["input_ids"][:, :-1]
        logits_btv = model(input_ids_bt, key=key, inference=False)
        return logits_btv

    def compute_loss(
        self,
        model: LLM,
        batch: Batch,
        output: Array,
        state: xax.State,
        key: PRNGKeyArray,
    ) -> Array:
        # Targets are shifted by 1 (next-token prediction)
        targets_bt = batch["input_ids"][:, 1:]
        mask_bt = batch["attention_mask"][:, 1:] == 1

        loss_bt = optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=targets_bt)
        return jnp.where(mask_bt, loss_bt, 0.0).sum() / mask_bt.sum()

    def compute_metrics(
        self,
        model: LLM,
        batch: Batch,
        output: Array,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> dict[str, xax.Metric]:
        targets_bt = batch["input_ids"][:, 1:]
        preds_bt = output.argmax(axis=-1)
        mask_bt = batch["attention_mask"][:, 1:] == 1

        metrics: dict[str, xax.Metric] = {}
        correct = (preds_bt == targets_bt) & mask_bt
        metrics["acc"] = xax.Scalar(correct.sum() / mask_bt.sum())

        # Note: llm_generate uses Python control flow (int(), break) which can't be
        # traced by JAX, so text generation is not supported inside compute_metrics.
        # For text generation during training, use a callback outside the JIT context.

        return metrics

    def decode_tokens(self, tokens: Array) -> str:
        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

    def get_dataset(self) -> Dataset:
        ds = load_dataset("Trelis/tiny-shakespeare", split="train")
        tokenize_fn = partial(
            _tokenize_with_tokenizer,
            tokenizer=self.tokenizer,
            max_length=self.config.sequence_length,
        )
        ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="numpy", columns=["input_ids", "attention_mask"])
        return ds


def _tokenize_with_tokenizer(
    examples: dict[str, str],
    tokenizer: Qwen2TokenizerFast,
    max_length: int,
) -> dict[str, list[int]]:
    """Standalone tokenization function that can be properly hashed for dataset fingerprinting."""
    return tokenizer(
        examples["Text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


if __name__ == "__main__":
    ShakespeareLora.launch(
        Config(
            batch_size=4,
            gradient_accumulation_steps=4,
            log_heavy_every_n_seconds=120,
            max_steps=1000,
        ),
    )
