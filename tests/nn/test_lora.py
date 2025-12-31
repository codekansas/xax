"""Tests the LoRA utilities."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from xax.nn.lora import LoRALinear, lora_filter_spec, loraize, loraize_linear, merge_lora


def test_lora_linear_matches_manual_delta() -> None:
    key = jrandom.key(0)
    lin_key, lora_key, x_key = jrandom.split(key, 3)
    linear = eqx.nn.Linear(4, 3, key=lin_key)
    lora_linear = loraize_linear(linear, rank=2, alpha=4.0, key=lora_key)
    x_bi = jrandom.normal(x_key, (2, 4))

    y_model_bo = lora_linear(x_bi, inference=True)
    manual_delta_bo = (x_bi @ lora_linear.lora_a_ir) @ lora_linear.lora_b_ro
    expected_bo = jax.vmap(linear)(x_bi) + manual_delta_bo * lora_linear.scaling

    assert jnp.allclose(y_model_bo, expected_bo)


def test_loraize_replaces_all_linears() -> None:
    key = jrandom.key(1)
    mlp = eqx.nn.MLP(in_size=4, out_size=2, width_size=8, depth=2, key=key)

    linear_count = sum(
        isinstance(node, eqx.nn.Linear)
        for node in jax.tree_util.tree_leaves(mlp, is_leaf=lambda x: isinstance(x, eqx.nn.Linear))
    )

    lora_mlp = loraize(mlp, rank=1, key=jrandom.key(7))
    lora_count = sum(
        isinstance(node, LoRALinear)
        for node in jax.tree_util.tree_leaves(lora_mlp, is_leaf=lambda x: isinstance(x, LoRALinear))
    )

    assert lora_count == linear_count


def test_lora_filter_spec_partitions_params() -> None:
    key = jrandom.key(2)
    linear = eqx.nn.Linear(3, 3, key=key)
    lora_linear = loraize_linear(linear, rank=1, key=jrandom.key(5))

    spec = lora_filter_spec(lora_linear)
    trainable, frozen = eqx.partition(lora_linear, spec)

    assert trainable.lora_a_ir is not None
    assert trainable.lora_b_ro is not None
    assert trainable.weight_oi is None
    assert frozen.weight_oi is not None


@pytest.mark.parametrize("rank", [1, 2])
def test_merge_lora_matches_inference(rank: int) -> None:
    key = jrandom.key(3)
    mlp = eqx.nn.MLP(in_size=3, out_size=2, width_size=6, depth=1, key=key)
    lora_mlp = loraize(mlp, rank=rank, alpha=float(rank), key=jrandom.key(9))

    x_bi = jrandom.normal(jrandom.key(4), (4, 3))
    y_lora_bo = jax.vmap(lora_mlp)(x_bi)
    merged = merge_lora(lora_mlp)
    y_merged_bo = jax.vmap(merged)(x_bi)

    assert jnp.allclose(y_lora_bo, y_merged_bo)
