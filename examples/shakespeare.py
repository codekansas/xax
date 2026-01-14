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

# DEFAULT_MODEL_REPO = "Qwen/Qwen3-1.7B"  # Can also use 0.6B or 4B (4B needs >32GB VRAM)
DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B"
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "gate", "up")


class Batch(TypedDict):
    attention_mask: Array
    input_ids: Array


@dataclass
class Config(xax.SupervisedConfig):
    # Model settings
    model_repo: str = xax.field(DEFAULT_MODEL_REPO, help="HuggingFace model repository")

    # LoRA settings
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(32.0, help="LoRA alpha parameter (actual scaling is alpha/rank)")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")
    lora_targets: tuple[str, ...] | None = xax.field(DEFAULT_LORA_TARGETS, help="Layer name suffixes to apply LoRA to")

    # Training settings
    learning_rate: float = xax.field(5e-4, help="Peak learning rate")
    min_learning_rate: float = xax.field(1e-5, help="Minimum learning rate for cosine decay")
    warmup_steps: int = xax.field(50, help="Number of warmup steps")
    sequence_length: int = xax.field(1025, help="Maximum sequence length (N - 1 should be a multiple of 64 for cuDNN)")
    use_gradient_checkpointing: bool = xax.field(True, help="Recompute activations to save memory")
    eval_prompt: str = xax.field("To be or not to be", help="Prompt to use for evaluation")


class ShakespeareLora(xax.SupervisedTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(config.model_repo)

        # Pre-encode the generation prompt for use in compute_metrics
        self._generation_prompt_tokens = jnp.array(
            self.tokenizer.encode(
                self.config.eval_prompt,
                add_special_tokens=False,
            ),
            dtype=jnp.int32,
        )

    @override
    def get_model(self, params: xax.InitParams) -> xax.LLM:
        # Loads the HF model config, optionally enabling gradient checkpointing.
        llm_config = xax.hf_config_to_llm_config(
            xax.load_hf_config(self.config.model_repo),
            use_remat=self.config.use_gradient_checkpointing,
        )

        # Build model with correct config
        model = xax.build_qwen3_model(llm_config, key=params.key)

        # Load pre-trained weights
        model = xax.load_hf_weights_into_llm(model, self.config.model_repo)

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
        hidden_btd = model.forward_hidden(input_ids_bt, key=key, inference=False)

        # Compute loss using chunked cross-entropy to avoid materializing full logits
        # This processes the sequence in chunks, computing logits for each chunk
        # and discarding them after computing the loss contribution
        loss = xax.chunked_cross_entropy_loss(
            hidden_btd,
            targets_bt,
            model.lm_head.weight,
            mask_bt,
            chunk_size=256,
        )

        metrics: dict[str, xax.Metric] = {}

        # Always log perplexity (exp of loss)
        perplexity = jnp.exp(loss)
        metrics["perplexity"] = xax.Scalar(perplexity)

        if heavy:
            # Compute prediction accuracy.
            accuracy = xax.chunked_cross_entropy_acc(
                hidden_btd,
                targets_bt,
                model.lm_head.weight,
                mask_bt,
                chunk_size=256,
            )
            metrics["accuracy"] = xax.Scalar(accuracy)

            # Generate text using JIT-compilable generation
            prompt_tokens = self._generation_prompt_tokens
            eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
            gen_key, key = jax.random.split(key)
            generated_tokens, _ = xax.llm_generate_jit(
                model,
                prompt_tokens,
                eos_id,
                max_new_tokens=64,
                temperature=0.8,
                top_p=0.9,
                key=gen_key,
            )
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
