#!/usr/bin/env -S uv run --no-project --script
"""Trains a state space model on a character-level tokenized dataset of Shakespeare."""

from dataclasses import dataclass
from functools import partial
from typing import Protocol, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from datasets import Dataset, load_dataset
from jaxtyping import Array, PRNGKeyArray
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

import xax


class Batch(TypedDict):
    attention_mask: Array
    input_ids: Array


@dataclass
class Config(xax.SupervisedConfig):
    num_layers: int = xax.field(4)
    hidden_size: int = xax.field(256)
    learning_rate: float = xax.field(1e-4)
    min_learning_rate: float = xax.field(1e-5, help="Minimum learning rate for cosine decay")
    warmup_steps: int = xax.field(1000, help="Number of warmup steps")
    sequence_length: int = xax.field(1024)
    model_type: str = xax.field("ssm", help="The model to use")


class SequenceModel(Protocol):
    def predict_sequence(self, x_seq: Array) -> Array: ...

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array: ...


class RNN(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: tuple[eqx.nn.GRUCell, ...]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ) -> None:
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(input_size, hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, num_layers)
        self.rnn_cells = tuple(
            eqx.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size, key=keys[i]) for i in range(num_layers)
        )
        self.output_layer = eqx.nn.Linear(hidden_size, output_size, key=keys[-1])

    def __call__(self, hs: list[Array], x: Array) -> tuple[list[Array], Array]:
        new_hs = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            h = rnn_cell(x, hs[i])
            new_hs.append(h)
            x = h  # Pass the output of the current layer as input to the next
        y = self.output_layer(x)
        return new_hs, y

    def predict_sequence(self, x_seq: Array) -> Array:
        hs = [jnp.zeros(cell.hidden_size) for cell in self.rnn_cells]
        x_seq = jax.vmap(self.vocab_embedding)(x_seq)

        def step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        _, y_seq = jax.lax.scan(step, hs, x_seq)
        return y_seq

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        hs = [jnp.zeros(cell.hidden_size) for cell in self.rnn_cells]
        prompt_seq_embedded = jax.vmap(self.vocab_embedding)(prompt_seq)

        def encode_step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        def decode_step(
            carry: tuple[list[Array], Array, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[list[Array], Array, PRNGKeyArray], Array]:
            hs, last_token, rng = carry
            token_embedded = self.vocab_embedding(last_token)
            hs, y = self(hs, token_embedded)
            token = jax.random.categorical(rng, y)
            rng = jax.random.split(rng)[0]
            return (hs, token, rng), token

        hs, _ = jax.lax.scan(encode_step, hs, prompt_seq_embedded)
        _, sequence = jax.lax.scan(decode_step, (hs, prompt_seq[-1], jax.random.key(0)), None, length=max_len)

        return sequence


class LSTM(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: tuple[eqx.nn.LSTMCell, ...]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ) -> None:
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(input_size, hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, num_layers)
        self.rnn_cells = tuple(
            eqx.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, key=keys[i]) for i in range(num_layers)
        )
        self.output_layer = eqx.nn.Linear(hidden_size, output_size, key=keys[-1])

    def __call__(self, hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
        new_hs: list[tuple[Array, Array]] = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            h, c = rnn_cell(x, hs[i])
            new_hs.append((h, c))
            x = h  # Pass the output of the current layer as input to the next
        y = self.output_layer(x)
        return new_hs, y

    def predict_sequence(self, x_seq: Array) -> Array:
        hs = [(jnp.zeros(cell.hidden_size), jnp.zeros(cell.hidden_size)) for cell in self.rnn_cells]
        x_seq = jax.vmap(self.vocab_embedding)(x_seq)

        def step(hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
            hs, y = self(hs, x)
            return hs, y

        _, y_seq = jax.lax.scan(step, hs, x_seq)
        return y_seq

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        hs = [(jnp.zeros(cell.hidden_size), jnp.zeros(cell.hidden_size)) for cell in self.rnn_cells]
        prompt_seq_embedded = jax.vmap(self.vocab_embedding)(prompt_seq)

        def encode_step(hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
            hs, y = self(hs, x)
            return hs, y

        def decode_step(
            carry: tuple[list[tuple[Array, Array]], Array, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[list[tuple[Array, Array]], Array, PRNGKeyArray], Array]:
            hs, last_token, rng = carry
            token_embedded = self.vocab_embedding(last_token)
            hs, y = self(hs, token_embedded)
            token = jax.random.categorical(rng, y)
            rng = jax.random.split(rng)[0]
            return (hs, token, rng), token

        hs, _ = jax.lax.scan(encode_step, hs, prompt_seq_embedded)
        _, sequence = jax.lax.scan(decode_step, (hs, prompt_seq[-1], jax.random.key(0)), None, length=max_len)

        return sequence


class ShakespearePrediction(xax.SupervisedTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def compute_metrics(
        self,
        model: SequenceModel,
        batch: Batch,
        output: Array,
        state: xax.State,
        heavy: bool,
        key: PRNGKeyArray,
    ) -> dict[str, xax.Metric]:
        y, yhat = batch["input_ids"][:, 1:], output.argmax(axis=-1)
        metrics: dict[str, xax.Metric] = {}
        metrics["acc"] = xax.Scalar((yhat == y).astype(float).mean())
        if not heavy:
            return metrics

        # prompt = "To be or not to be, that is the"
        # prompt_seq = jnp.array(self.tokenizer.encode(prompt))
        prompt_seq = batch["input_ids"][0, :16]
        generated_tokens = model.generate_sequence(prompt_seq, max_len=96)
        metrics["generated"] = xax.Tokens(generated_tokens)

        return metrics

    def decode_tokens(self, tokens: Array) -> str:
        return self.tokenizer.decode(tokens.tolist())

    def get_model(self, params: xax.InitParams) -> SequenceModel:
        match self.config.model_type:
            case "rnn":
                return RNN(
                    input_size=self.vocab_size,
                    hidden_size=self.config.hidden_size,
                    output_size=self.vocab_size,
                    num_layers=self.config.num_layers,
                    key=params.key,
                )
            case "lstm":
                return LSTM(
                    input_size=self.vocab_size,
                    hidden_size=self.config.hidden_size,
                    output_size=self.vocab_size,
                    num_layers=self.config.num_layers,
                    key=params.key,
                )
            case "ssm":
                return xax.SSM(
                    input_size=self.vocab_size,
                    hidden_size=self.config.hidden_size,
                    output_size=self.vocab_size,
                    num_layers=self.config.num_layers,
                    block_type="diagonal",
                    skip_connections=True,
                    discretize=False,
                    key=params.key,
                )
            case "transformer":
                return xax.Transformer(
                    vocab_size=self.vocab_size,
                    embed_dim=self.config.hidden_size,
                    num_heads=self.config.hidden_size // 64,
                    ff_dim=self.config.hidden_size * 4,
                    num_layers=self.config.num_layers,
                    output_size=self.vocab_size,
                    context_length=self.config.sequence_length,
                    causal=True,
                    # use_rotary_embeddings=True,
                    key=params.key,
                )
            case _:
                raise ValueError(f"Unknown model type: {self.config.model_type}")

    def get_optimizer(self) -> xax.Optimizer:
        # Create learning rate schedule with warmup and cosine decay
        if self.config.max_steps is not None:
            # Warmup schedule
            warmup_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=self.config.learning_rate,
                transition_steps=self.config.warmup_steps,
            )

            # Cosine decay schedule
            cosine_schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=max(self.config.max_steps - self.config.warmup_steps, 1),
                alpha=self.config.min_learning_rate / self.config.learning_rate,
            )

            # Combine warmup and decay
            learning_rate_schedule = optax.join_schedules(
                schedules=[warmup_schedule, cosine_schedule],
                boundaries=[self.config.warmup_steps],
            )
        else:
            # Fallback to constant learning rate if max_steps not specified
            learning_rate_schedule = self.config.learning_rate

        return optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=0.01,
        )

    def get_output(self, model: SequenceModel, batch: Batch, state: xax.State | None, key: PRNGKeyArray) -> Array:
        return jax.vmap(model.predict_sequence)(batch["input_ids"][:, :-1])

    def compute_loss(
        self,
        model: SequenceModel,
        batch: Batch,
        output: Array,
        state: xax.State | None,
        key: PRNGKeyArray,
    ) -> Array:
        y, yhat, mask = batch["input_ids"][:, 1:], output, batch["attention_mask"][:, 1:] == 1
        return optax.softmax_cross_entropy_with_integer_labels(logits=yhat, labels=y, where=mask).mean()

    def _tokenize(self, examples: dict[str, str]) -> dict[str, list[int]]:
        return self.tokenizer(
            examples["Text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.sequence_length,
        )

    def get_dataset(self) -> Dataset:
        ds = load_dataset("Trelis/tiny-shakespeare", split="train")
        # Use functools.partial to create a serializable function for dataset fingerprinting
        tokenize_fn = partial(_tokenize_with_tokenizer, tokenizer=self.tokenizer)
        ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="numpy", columns=["input_ids", "attention_mask"])
        return ds


def _tokenize_with_tokenizer(examples: dict[str, str], tokenizer: Qwen2TokenizerFast) -> dict[str, list[int]]:
    """Standalone tokenization function that can be properly hashed for dataset fingerprinting."""
    return tokenizer(
        examples["Text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )


if __name__ == "__main__":
    # Launch the training task.
    ShakespearePrediction.launch(
        Config(
            batch_size=8,
            log_heavy_every_n_seconds=120,
            gradient_accumulation_steps=8,
        ),
    )
