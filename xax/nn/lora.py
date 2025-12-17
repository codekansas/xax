"""LoRA utilities for Equinox modules."""

import math
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class LoRALinear(eqx.Module):
    """Linear layer augmented with a low-rank residual (LoRA).

    The base weight and bias are treated as frozen parameters; only the low-rank
    matrices ``lora_a_ir`` and ``lora_b_ro`` are updated during fine-tuning.
    Scaling follows the convention ``alpha / rank``.
    """

    weight_oi: Array
    bias_o: Array | None
    lora_a_ir: Array
    lora_b_ro: Array
    scaling: float = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    @classmethod
    def from_linear(
        cls,
        linear: eqx.nn.Linear,
        rank: int,
        *,
        alpha: float | None = None,
        dropout_rate: float = 0.0,
        key: PRNGKeyArray | None = None,
    ) -> "LoRALinear":
        """Create a LoRA-augmented linear layer from an ``eqx.nn.Linear``."""
        if rank < 1:
            raise ValueError("rank must be at least 1")
        if alpha is None:
            alpha = float(rank)
        if key is None:
            key = jrandom.key(0)

        key_a, _ = jrandom.split(key)
        in_features = linear.in_features
        out_features = linear.out_features

        lora_a_ir = jrandom.normal(key_a, (in_features, rank), dtype=linear.weight.dtype) / jnp.sqrt(float(in_features))
        lora_b_ro = jnp.zeros((rank, out_features), dtype=linear.weight.dtype)

        return cls(
            weight_oi=jax.lax.stop_gradient(linear.weight),
            bias_o=None if linear.bias is None else jax.lax.stop_gradient(linear.bias),
            lora_a_ir=lora_a_ir,
            lora_b_ro=lora_b_ro,
            scaling=alpha / float(rank),
            dropout_rate=dropout_rate,
        )

    def __call__(self, x_bi: Array, *, key: PRNGKeyArray | None = None, inference: bool = False) -> Array:
        y_bo = x_bi @ self.weight_oi.T
        if self.bias_o is not None:
            y_bo = y_bo + self.bias_o

        x_lora_bi = x_bi
        if self.dropout_rate > 0.0 and not inference:
            if key is None:
                raise ValueError("LoRA dropout requires a PRNG key.")
            keep_prob = 1.0 - self.dropout_rate
            mask_bi = jrandom.bernoulli(key, keep_prob, x_bi.shape)
            x_lora_bi = jnp.where(mask_bi, x_bi / keep_prob, 0.0)

        delta_bo = (x_lora_bi @ self.lora_a_ir) @ self.lora_b_ro
        return y_bo + delta_bo * self.scaling


def loraize_linear(
    linear: eqx.nn.Linear,
    rank: int,
    *,
    alpha: float | None = None,
    dropout_rate: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> LoRALinear:
    """Replace an Equinox linear layer with a ``LoRALinear``."""
    return LoRALinear.from_linear(linear, rank, alpha=alpha, dropout_rate=dropout_rate, key=key)


def loraize(
    module: eqx.Module,
    rank: int,
    *,
    alpha: float | None = None,
    dropout_rate: float = 0.0,
    predicate: Callable[[eqx.nn.Linear], bool] | None = None,
    key: PRNGKeyArray | None = None,
) -> eqx.Module:
    """Recursively replace ``eqx.nn.Linear`` layers with ``LoRALinear``."""
    if predicate is None:
        predicate = lambda _: True
    if key is None:
        key = jrandom.key(0)

    def convert(node: Any) -> Any:
        nonlocal key
        if isinstance(node, eqx.nn.Linear) and predicate(node):
            key, init_key = jrandom.split(key)
            return loraize_linear(node, rank, alpha=alpha, dropout_rate=dropout_rate, key=init_key)
        return node

    return jax.tree_util.tree_map(convert, module, is_leaf=lambda node: isinstance(node, eqx.nn.Linear))


def lora_filter_spec(module: eqx.Module) -> eqx.Module:
    """Build a filter spec marking only LoRA parameters as trainable."""

    def _mark(model_node: Any, spec_node: Any) -> Any:
        if isinstance(model_node, LoRALinear):
            spec_node = eqx.tree_at(lambda l: l.lora_a_ir, spec_node, True)
            spec_node = eqx.tree_at(lambda l: l.lora_b_ro, spec_node, True)
            return spec_node
        if hasattr(model_node, "__dataclass_fields__"):
            for fname in model_node.__dataclass_fields__:
                child_model = getattr(model_node, fname)
                child_spec = getattr(spec_node, fname)
                updated_child = _mark(child_model, child_spec)
                spec_node = eqx.tree_at(lambda m: getattr(m, fname), spec_node, updated_child)
            return spec_node
        if isinstance(model_node, list):
            return [_mark(m, s) for m, s in zip(model_node, spec_node, strict=False)]
        if isinstance(model_node, tuple):
            return tuple(_mark(m, s) for m, s in zip(model_node, spec_node, strict=False))
        if isinstance(model_node, dict):
            return {k: _mark(model_node[k], spec_node[k]) for k in model_node}
        return spec_node

    spec = jax.tree_util.tree_map(lambda _: False, module)
    return _mark(module, spec)


def merge_lora(module: eqx.Module) -> eqx.Module:
    """Merge LoRA weights back into standard ``eqx.nn.Linear`` layers."""

    def convert(node: Any) -> Any:
        if isinstance(node, LoRALinear):
            delta_oi = (node.lora_a_ir @ node.lora_b_ro).T * node.scaling
            merged_weight_oi = node.weight_oi + delta_oi
            merged = eqx.nn.Linear(
                in_features=node.weight_oi.shape[1],
                out_features=node.weight_oi.shape[0],
                use_bias=node.bias_o is not None,
                key=jrandom.key(0),
            )
            merged = eqx.tree_at(lambda l: l.weight, merged, merged_weight_oi)
            if node.bias_o is not None:
                merged = eqx.tree_at(lambda l: l.bias, merged, node.bias_o)
            return merged
        return node

    return jax.tree_util.tree_map(convert, module, is_leaf=lambda node: isinstance(node, LoRALinear))


# TODO: Remove any PyTorch logic from this file, it should just be pure Jax.
try:  # pragma: no cover - optional torch support
    import torch
    import torch.nn as torch_nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional
    torch = None
    torch_nn = None
    _TORCH_AVAILABLE = False

    class TorchLoRALinear:  # type: ignore[too-few-public-methods]
        """Placeholder that raises when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            raise ImportError("PyTorch is required for TorchLoRALinear.")


if _TORCH_AVAILABLE:  # pragma: no cover - exercised in example usage

    class TorchLoRALinear(torch_nn.Module):
        """Torch counterpart of :class:`LoRALinear`."""

        def __init__(
            self,
            linear: torch_nn.Linear,
            rank: int,
            *,
            alpha: float | None = None,
            dropout_rate: float = 0.0,
        ) -> None:
            super().__init__()
            if rank < 1:
                raise ValueError("rank must be at least 1")
            if alpha is None:
                alpha = float(rank)

            self.register_buffer("weight_oi", linear.weight.detach().clone())
            if linear.bias is not None:
                self.register_buffer("bias_o", linear.bias.detach().clone())
            else:
                self.bias_o = None

            self.lora_a_ir = torch_nn.Parameter(
                torch.randn(linear.in_features, rank, device=linear.weight.device, dtype=linear.weight.dtype)
                / math.sqrt(float(linear.in_features))
            )
            self.lora_b_ro = torch_nn.Parameter(
                torch.zeros(rank, linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype)
            )
            self.scaling = alpha / float(rank)
            self.dropout = torch_nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

        def forward(self, x_bi: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            y_bo = x_bi @ self.weight_oi.T
            if self.bias_o is not None:
                y_bo = y_bo + self.bias_o

            x_lora_bi = self.dropout(x_bi) if self.dropout is not None else x_bi
            delta_bo = (x_lora_bi @ self.lora_a_ir) @ self.lora_b_ro
            return y_bo + delta_bo * self.scaling


def torch_loraize(
    module: Any,
    rank: int,
    *,
    alpha: float | None = None,
    dropout_rate: float = 0.0,
    predicate: Callable[[Any], bool] | None = None,
) -> Any:
    """Recursively replace ``torch.nn.Linear`` layers with ``TorchLoRALinear``."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for torch_loraize.")
    if predicate is None:
        predicate = lambda _: True

    for name, child in module.named_children():
        if isinstance(child, torch_nn.Linear) and predicate(child):
            setattr(module, name, TorchLoRALinear(child, rank, alpha=alpha, dropout_rate=dropout_rate))
        else:
            torch_loraize(child, rank, alpha=alpha, dropout_rate=dropout_rate, predicate=predicate)
    return module


def torch_merge_lora(module: Any) -> Any:
    """Merge torch LoRA layers into base linear weights in-place."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for torch_merge_lora.")

    for name, child in module.named_children():
        if isinstance(child, TorchLoRALinear):
            delta_oi = (child.lora_a_ir @ child.lora_b_ro).T * child.scaling
            merged = torch_nn.Linear(
                in_features=child.weight_oi.shape[1],
                out_features=child.weight_oi.shape[0],
                bias=child.bias_o is not None,
                device=child.weight_oi.device,
                dtype=child.weight_oi.dtype,
            )
            merged.weight.data.copy_(child.weight_oi + delta_oi)
            if child.bias_o is not None:
                merged.bias.data.copy_(child.bias_o)
            setattr(module, name, merged)
        else:
            torch_merge_lora(child)
    return module
