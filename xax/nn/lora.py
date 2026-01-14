"""LoRA utilities for Equinox modules."""

from typing import Any, Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

Tmodule = TypeVar("Tmodule", bound=eqx.Module)


class LoRALinear(eqx.Module):
    """Linear layer augmented with a low-rank residual (LoRA).

    The base weight and bias are treated as frozen parameters; only the low-rank
    matrices ``lora_a_ir`` and ``lora_b_ro`` are updated during fine-tuning.

    Following the standard LoRA formulation, the contribution is scaled by
    ``alpha / rank``. This ensures consistent gradient magnitudes regardless of
    rank choice. With the default ``alpha=16``, rank 16 gives a scaling of 1.0,
    while rank 8 gives 2.0, and rank 32 gives 0.5.
    """

    weight_oi: Array
    bias_o: Array | None
    lora_a_ir: Array
    lora_b_ro: Array
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    @classmethod
    def from_linear(
        cls,
        linear: eqx.nn.Linear,
        rank: int,
        *,
        alpha: float = 16.0,
        dropout_rate: float = 0.0,
        key: PRNGKeyArray | None = None,
    ) -> "LoRALinear":
        """Create a LoRA-augmented linear layer from an ``eqx.nn.Linear``.

        Args:
            linear: The base linear layer to augment.
            rank: Rank of the low-rank decomposition.
            alpha: LoRA scaling parameter. The actual scaling is alpha/rank.
                With the default alpha=16 and rank=16, scaling is 1.0.
            dropout_rate: Dropout rate for LoRA layers.
            key: PRNG key for initialization.
        """
        if rank < 1:
            raise ValueError("rank must be at least 1")
        if key is None:
            key = jrandom.key(0)

        key_a, _ = jrandom.split(key)
        in_features = int(linear.in_features)
        out_features = int(linear.out_features)

        lora_a_ir = jrandom.normal(key_a, (in_features, rank), dtype=linear.weight.dtype) / jnp.sqrt(float(in_features))
        lora_b_ro = jnp.zeros((rank, out_features), dtype=linear.weight.dtype)

        return cls(
            weight_oi=jax.lax.stop_gradient(linear.weight),
            bias_o=None if linear.bias is None else jax.lax.stop_gradient(linear.bias),
            lora_a_ir=lora_a_ir,
            lora_b_ro=lora_b_ro,
            rank=rank,
            alpha=alpha,
            dropout_rate=dropout_rate,
        )

    @property
    def scaling(self) -> float:
        """Return the LoRA scaling factor (alpha / rank)."""
        return self.alpha / self.rank

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
    alpha: float = 16.0,
    dropout_rate: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> LoRALinear:
    """Replace an Equinox linear layer with a ``LoRALinear``.

    Args:
        linear: The base linear layer to augment.
        rank: Rank of the low-rank decomposition.
        alpha: LoRA scaling parameter. The actual scaling is alpha/rank.
            With the default alpha=16 and rank=16, scaling is 1.0.
        dropout_rate: Dropout rate for LoRA layers.
        key: PRNG key for initialization.
    """
    return LoRALinear.from_linear(linear, rank, alpha=alpha, dropout_rate=dropout_rate, key=key)


def loraize(
    module: Tmodule,
    rank: int,
    *,
    alpha: float = 16.0,
    dropout_rate: float = 0.0,
    predicate: Callable[[eqx.nn.Linear], bool] | None = None,
    key: PRNGKeyArray | None = None,
) -> Tmodule:
    """Recursively replace ``eqx.nn.Linear`` layers with ``LoRALinear``.

    Args:
        module: The module to transform.
        rank: LoRA rank for the low-rank decomposition.
        alpha: LoRA scaling parameter. The actual scaling is alpha/rank.
            With the default alpha=16 and rank=16, scaling is 1.0.
        dropout_rate: Dropout rate for LoRA layers.
        predicate: Optional function to filter which Linear layers to convert.
        key: PRNG key for initialization.
    """
    if predicate is None:

        def predicate(_: eqx.nn.Linear) -> bool:  # noqa: ARG001
            return True

    if key is None:
        key = jrandom.key(0)

    def convert(node: Any) -> Any:  # noqa: ANN001, ANN401
        nonlocal key
        if isinstance(node, eqx.nn.Linear) and predicate(node):
            assert key is not None  # Checked above
            key, init_key = jrandom.split(key)
            return loraize_linear(node, rank, alpha=alpha, dropout_rate=dropout_rate, key=init_key)
        return node

    return jax.tree_util.tree_map(convert, module, is_leaf=lambda node: isinstance(node, eqx.nn.Linear))


def loraize_by_path(
    module: Tmodule,
    rank: int,
    *,
    include_suffixes: list[str] | None = None,
    exclude_suffixes: list[str] | None = None,
    alpha: float = 16.0,
    dropout_rate: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> Tmodule:
    """Recursively replace ``eqx.nn.Linear`` layers matching path suffixes.

    This function applies LoRA selectively based on the path names in the model tree.
    For example, to only apply LoRA to query and value projections in an LLM:

        loraize_by_path(model, rank=16, include_suffixes=["q_proj", "v_proj"])

    Args:
        module: The module to transform.
        rank: LoRA rank for the low-rank decomposition.
        include_suffixes: List of suffixes to match against layer paths.
            If None, all Linear layers are candidates.
        exclude_suffixes: List of suffixes that exclude a layer if matched.
        alpha: LoRA scaling parameter. The actual scaling is alpha/rank.
            With the default alpha=16 and rank=16, scaling is 1.0.
        dropout_rate: Dropout rate for LoRA layers.
        key: PRNG key for initialization.

    Returns:
        Module with selected Linear layers replaced by LoRALinear.
    """
    # Initialize key if not provided
    current_key: PRNGKeyArray = jrandom.key(0) if key is None else key

    def should_loraize(path: str) -> bool:
        should_include = include_suffixes is None or any(path.endswith(s) for s in include_suffixes)
        should_exclude = exclude_suffixes is not None and any(path.endswith(s) for s in exclude_suffixes)
        return should_include and not should_exclude

    def convert_with_path(path: tuple[Any, ...], node: Any) -> Any:  # noqa: ANN401
        nonlocal current_key
        if isinstance(node, eqx.nn.Linear):
            # Build path string from JAX key path
            path_str = ".".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
            if should_loraize(path_str):
                current_key, init_key = jrandom.split(current_key)
                return loraize_linear(node, rank, alpha=alpha, dropout_rate=dropout_rate, key=init_key)
        return node

    # Use tree_map_with_path to get paths during traversal
    return jax.tree_util.tree_map_with_path(
        convert_with_path,
        module,
        is_leaf=lambda node: isinstance(node, eqx.nn.Linear),
    )


def lora_filter_spec(module: Tmodule) -> Tmodule:
    """Build a filter spec marking only LoRA parameters as trainable.

    Returns a pytree with the same structure as the module, where:
    - LoRA parameters (lora_a_ir, lora_b_ro) are marked True
    - All other leaves are marked False

    Use with eqx.partition or eqx.filter:

        filter_spec = lora_filter_spec(model)
        trainable, frozen = eqx.partition(model, filter_spec)

    Or use for gradient masking:

        grads = jax.tree.map(
            lambda g, f: g if f else None,
            raw_grads,
            filter_spec,
        )
    """

    def make_spec(node: Any) -> Any:  # noqa: ANN401
        if isinstance(node, LoRALinear):
            # Create a spec tree matching LoRALinear structure with all False
            base_spec = jax.tree.map(lambda _: False, node)
            # Mark LoRA params as trainable using eqx.tree_at
            base_spec = eqx.tree_at(lambda x: x.lora_a_ir, base_spec, True)
            base_spec = eqx.tree_at(lambda x: x.lora_b_ro, base_spec, True)
            return base_spec
        # For regular leaves (arrays, scalars), return False
        return False

    # Treat LoRALinear as a leaf so we can create its spec structure
    return jax.tree.map(make_spec, module, is_leaf=lambda x: isinstance(x, LoRALinear))


def merge_lora(module: Tmodule) -> Tmodule:
    """Merge LoRA weights back into standard ``eqx.nn.Linear`` layers."""

    def convert(node: Any) -> Any:  # noqa: ANN001, ANN401
        if isinstance(node, LoRALinear):
            delta_oi = (node.lora_a_ir @ node.lora_b_ro).T * node.scaling
            merged_weight_oi = node.weight_oi + delta_oi
            merged = eqx.nn.Linear(
                in_features=node.weight_oi.shape[1],
                out_features=node.weight_oi.shape[0],
                use_bias=node.bias_o is not None,
                key=jrandom.key(0),
            )

            def get_weight(linear: eqx.nn.Linear) -> Array:
                return linear.weight

            def get_bias(linear: eqx.nn.Linear) -> Array | None:
                return linear.bias

            merged = eqx.tree_at(get_weight, merged, merged_weight_oi)
            if node.bias_o is not None:
                merged = eqx.tree_at(get_bias, merged, node.bias_o)
            return merged
        return node

    return jax.tree_util.tree_map(convert, module, is_leaf=lambda node: isinstance(node, LoRALinear))
