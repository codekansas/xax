#!/usr/bin/env -S uv run --no-project --script
"""LoRA fine-tuning of a pre-trained LLM on Shakespeare text."""

from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

import xax

DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B"
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "k_proj", "o_proj")


class Batch(TypedDict):
    attention_mask: Array
    input_ids: Array


@dataclass
class Config(xax.SupervisedConfig):
    # LoRA settings
    lora_rank: int = xax.field(16, help="Rank of LoRA decomposition")
    lora_alpha: float = xax.field(1.0, help="LoRA scaling factor (1.0 = no amplification)")
    lora_dropout: float = xax.field(0.0, help="Dropout rate for LoRA layers")
    lora_targets: tuple[str, ...] | None = xax.field(DEFAULT_LORA_TARGETS, help="Layer name suffixes to apply LoRA to")

    # Training settings
    learning_rate: float = xax.field(1e-4, help="Peak learning rate")
    min_learning_rate: float = xax.field(1e-5, help="Minimum learning rate for cosine decay")
    warmup_steps: int = xax.field(100, help="Number of warmup steps")
    sequence_length: int = xax.field(512, help="Maximum sequence length")
    use_gradient_checkpointing: bool = xax.field(True, help="Recompute activations to save memory")


class ShakespeareLora(xax.SupervisedTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(DEFAULT_MODEL_REPO)

        # Pre-encode the generation prompt for use in compute_metrics
        prompt = "To be or not to be"
        self._generation_prompt_tokens = jnp.array(
            self.tokenizer.encode(prompt, add_special_tokens=False),
            dtype=jnp.int32,
        )

    def get_model(self, params: xax.InitParams) -> xax.LLM:
        # Loads the HF model config, optionally enabling gradient checkpointing.
        llm_config = xax.hf_config_to_llm_config(
            xax.load_hf_config(DEFAULT_MODEL_REPO),
            base=xax.QWEN3_SMALL,
            use_remat=self.config.use_gradient_checkpointing,
        )

        # Build model with correct config
        model = xax.build_qwen3_model(llm_config, key=params.key)

        # Load pre-trained weights
        model = xax.load_hf_weights_into_llm(model, DEFAULT_MODEL_REPO)

        # Apply LoRA selectively to specified layers (e.g., q_proj, v_proj)
        return xax.loraize_by_path(
            model,
            rank=self.config.lora_rank,
            include_suffixes=list(self.config.lora_targets) if self.config.lora_targets else None,
            alpha=self.config.lora_alpha,
            dropout_rate=self.config.lora_dropout,
            key=params.key,
        )

    def get_model_filter_spec(self, model: xax.LLM) -> xax.LLM:
        # Only train LoRA parameters (lora_a and lora_b matrices)
        # Base model weights are frozen
        return xax.lora_filter_spec(model)

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

    def get_output(self, model: xax.LLM, batch: Batch, state: xax.State, key: PRNGKeyArray) -> Array:
        # LLM takes (batch, seq_len) and returns (batch, seq_len, vocab_size)
        # Use all but last token as input
        input_ids_bt = batch["input_ids"][:, :-1]
        logits_btv = model(input_ids_bt, key=key, inference=False)
        return logits_btv

    def compute_loss(
        self,
        model: xax.LLM,
        batch: Batch,
        output: Array,
        state: xax.State,
        key: PRNGKeyArray,
    ) -> Array:
        # Targets are shifted by 1 (next-token prediction)
        targets_bt = batch["input_ids"][:, 1:]
        mask_bt = batch["attention_mask"][:, 1:] == 1
        loss_bt = optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=targets_bt)
        masked_loss = jnp.where(mask_bt, loss_bt, 0.0)
        loss = masked_loss.sum() / (mask_bt.sum() + 1e-8)
        return loss

    def compute_metrics(
        self,
        model: xax.LLM,
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

        if heavy:
            # Generate text using JIT-compilable generation
            # Use a fixed prompt encoded at class initialization time
            prompt_tokens = self._generation_prompt_tokens
            eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
            gen_key, key = jax.random.split(key)
            generated = xax.llm_generate_jit(
                model,
                prompt_tokens,
                eos_id,
                max_new_tokens=64,
                temperature=0.8,
                top_p=0.9,
                key=gen_key,
            )
            metrics["generated"] = xax.Tokens(generated)

        return metrics

    def decode_tokens(self, tokens: Array | np.ndarray) -> str:
        # Convert to list for tokenizer compatibility
        token_list: list[int] = tokens.tolist()
        return self.tokenizer.decode(token_list, skip_special_tokens=True)

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
            batch_size=8,
            max_grad_norm=1.0,
            gradient_accumulation_steps=8,
            log_heavy_every_n_seconds=120,
            max_steps=50_000,
        ),
    )
