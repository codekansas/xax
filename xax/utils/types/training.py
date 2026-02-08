"""Defines types that are used during training."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, PyTree

from xax.core.state import State
from xax.utils.structured_config import field

S = TypeVar("S")


@runtime_checkable
class Optimizer(Protocol):
    def init(self, params: optax.Params) -> S: ...

    def update(
        self,
        updates: optax.Updates,
        state: S,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, S]: ...


Precision = Literal["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]


def precision_to_jax_dtype(precision: Precision) -> jnp.dtype:
    if precision == "float32":
        return jnp.float32
    if precision == "bfloat16":
        return jnp.bfloat16
    if precision == "float16":
        return jnp.float16
    if precision == "float8_e4m3fn":
        return jnp.float8_e4m3fn
    return jnp.float8_e5m2


@jax.tree_util.register_dataclass
@dataclass
class PrecisionConfig:
    """Configuration for different precision types used during training.

    This allows fine-grained control over precision for different parts of training:
    - data_dtype: Precision for input data (can be lower to save H2D bandwidth)
    - param_dtype: Precision for model weights
    - compute_dtype: Precision for matrix multiplications
    - grad_dtype: Precision for gradient accumulation
    """

    data_dtype: Precision = field("bfloat16", help="Precision for input data")
    param_dtype: Precision = field("bfloat16", help="Precision for model weights")
    compute_dtype: Precision = field("bfloat16", help="Precision for matrix multiplications")
    grad_dtype: Precision = field("float32", help="Precision for gradients")

    @property
    def data_jax_dtype(self) -> jnp.dtype:
        return precision_to_jax_dtype(self.data_dtype)

    @property
    def param_jax_dtype(self) -> jnp.dtype:
        return precision_to_jax_dtype(self.param_dtype)

    @property
    def compute_jax_dtype(self) -> jnp.dtype:
        return precision_to_jax_dtype(self.compute_dtype)

    @property
    def grad_jax_dtype(self) -> jnp.dtype:
        return precision_to_jax_dtype(self.grad_dtype)


def as_shape_dtype(x: Any) -> Any:  # noqa: ANN401
    if isinstance(x, Array):
        return ocp.utils.to_shape_dtype_struct(x)
    return x


@dataclass
class TrainingState:
    models: list[PyTree]
    opt_states: list[optax.OptState]
    state: State
    aux_data: PyTree | None = None
