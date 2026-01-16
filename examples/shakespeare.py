#!/usr/bin/env -S uv run --no-project --script
"""LoRA fine-tuning of a pre-trained LLM on Shakespeare text."""

from dataclasses import dataclass
from functools import partial
from typing import TypedDict, override

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

import xax

# Attention (q_proj, v_proj) + MLP (gate, up) layers for best quality
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "gate", "up")


class Batch(TypedDict):
    attention_mask: Array
    input_ids: Array


@dataclass
class Config(xax.SupervisedConfig):
    # Model settings
    llm_repo: xax.LLMRepo = xax.field(xax.LLMRepo.QWEN3_600M, help="Pretrained model")

    # LoRA settings
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(16.0, help="LoRA alpha parameter (actual scaling is alpha/rank)")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")
    lora_targets: tuple[str, ...] | None = xax.field(DEFAULT_LORA_TARGETS, help="Layer name suffixes to apply LoRA to")

    # Training settings
    learning_rate: float = xax.field(5e-4, help="Peak learning rate")
    min_learning_rate: float = xax.field(1e-5, help="Minimum learning rate for cosine decay")
    warmup_steps: int = xax.field(50, help="Number of warmup steps")
    sequence_length: int = xax.field(1025, help="Maximum sequence length (N - 1 should be a multiple of 64 for cuDNN)")
    eval_prompt: str = xax.field("To be or not to be", help="Prompt to use for evaluation")


class ShakespeareLora(xax.SupervisedTask[Config]):
    # Large models that require tensor parallelism
    LARGE_MODELS = (xax.LLMRepo.QWEN3_8B, xax.LLMRepo.QWEN3_14B, xax.LLMRepo.QWEN3_32B)

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(config.llm_repo.value)

        # Pre-encode the generation prompt for use in compute_metrics
        self._generation_prompt_tokens = jnp.array(
            self.tokenizer.encode(
                self.config.eval_prompt,
                add_special_tokens=False,
            ),
            dtype=jnp.int32,
        )

    @override
    def should_do_model_parallel(self) -> bool:
        # Only enable model parallelism for models that won't easily fit on a single GPU.
        return self.config.llm_repo in self.LARGE_MODELS

    @override
    def get_model(self, params: xax.InitParams) -> xax.LLM:
        # Build model with correct config
        model = xax.build_pretrained_llm(self.config.llm_repo)

        # Apply LoRA selectively to specified layers (e.g., q_proj, v_proj)
        return xax.loraize_by_path(
            model,
            rank=self.config.lora_rank,
            include_suffixes=list(self.config.lora_targets) if self.config.lora_targets else None,
            alpha=self.config.lora_alpha,
            dropout_rate=self.config.lora_dropout,
            key=params.key,
        )

    @override
    def get_trainable_filter_spec(self, model: xax.LLM) -> xax.LLM:
        # Only train LoRA parameters (lora_a and lora_b matrices).
        # Base model weights are frozen.
        return xax.lora_filter_spec(model)

    @override
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

    @override
    def compute_loss(
        self,
        model: xax.LLM,
        batch: Batch,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, xax.Metric]]:
        # Use all but last token as input, targets are shifted by 1
        input_ids_bt = batch["input_ids"][:, :-1]
        targets_bt = batch["input_ids"][:, 1:]
        mask_bt = batch["attention_mask"][:, 1:] == 1

        # Memory-efficient forward pass: get hidden states without computing full logits
        hidden_btd = jax.vmap(model.forward_hidden)(input_ids_bt)

        # Compute loss using chunked cross-entropy to avoid materializing full logits
        # This processes the sequence in chunks, computing logits for each chunk
        # and discarding them after computing the loss contribution
        loss = jax.vmap(xax.chunked_cross_entropy_loss, in_axes=(0, 0, None, 0, None))(
            hidden_btd,
            targets_bt,
            model.lm_head.weight,
            mask_bt,
            256,  # Chunk size
        )
        loss = loss.mean()

        metrics: dict[str, xax.Metric] = {}

        # Always log perplexity (exp of loss)
        perplexity = jnp.exp(loss)
        metrics["perplexity"] = xax.Scalar(perplexity)

        if heavy:
            # Compute prediction accuracy.
            accuracy = jax.vmap(xax.chunked_cross_entropy_acc, in_axes=(0, 0, None, 0, None))(
                hidden_btd,
                targets_bt,
                model.lm_head.weight,
                mask_bt,
                256,
            )
            accuracy = accuracy.mean()
            metrics["accuracy"] = xax.Scalar(accuracy)

            # Generate text using JIT-compilable generation
            prompt_tokens = self._generation_prompt_tokens
            eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
            gen_key, key = jax.random.split(key)
            gen_result = xax.llm_generate_jit(
                model,
                prompt_tokens,
                eos_id,
                max_new_tokens=64,
                temperature=0.8,
                top_p=0.9,
                key=gen_key,
            )
            generated_tokens = gen_result[0]
            metrics["generated"] = xax.Tokens(generated_tokens)

        return loss, metrics

    @override
    def decode_tokens(self, tokens: Array | np.ndarray) -> str:
        # Convert to list and strip trailing zeros (padding from generation)
        token_list: list[int] = tokens.tolist()
        last_zero = len(token_list)
        while last_zero > 0 and token_list[last_zero - 1] == 0:
            last_zero -= 1
        token_list = token_list[:last_zero]
        return self.tokenizer.decode(token_list, skip_special_tokens=True)

    @override
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
            batch_size=16,
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,
            log_heavy_every_n_seconds=120,
            max_steps=60 * 10,
            step_kind="second",
        ),
    )
