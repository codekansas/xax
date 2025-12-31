"""Defines types that are used during training."""

import enum
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, PyTree

from xax.core.conf import field
from xax.core.state import State

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


class Precision(enum.Enum):
    """Precision enum for dtype configuration."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"

    def to_jax_dtype(self) -> jnp.dtype:
        """Convert to a JAX dtype."""
        dtype_map = {
            Precision.FLOAT32: jnp.float32,
            Precision.BFLOAT16: jnp.bfloat16,
            Precision.FLOAT16: jnp.float16,
            Precision.FLOAT8_E4M3FN: jnp.float8_e4m3fn,
            Precision.FLOAT8_E5M2: jnp.float8_e5m2,
        }
        return dtype_map[self]


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

    data_dtype: Precision = field(Precision.BFLOAT16, help="Precision for input data")
    param_dtype: Precision = field(Precision.BFLOAT16, help="Precision for model weights")
    compute_dtype: Precision = field(Precision.BFLOAT16, help="Precision for matrix multiplications")
    grad_dtype: Precision = field(Precision.FLOAT32, help="Precision for gradients")

    @property
    def data_jax_dtype(self) -> jnp.dtype:
        return self.data_dtype.to_jax_dtype()

    @property
    def param_jax_dtype(self) -> jnp.dtype:
        return self.param_dtype.to_jax_dtype()

    @property
    def compute_jax_dtype(self) -> jnp.dtype:
        return self.compute_dtype.to_jax_dtype()

    @property
    def grad_jax_dtype(self) -> jnp.dtype:
        return self.grad_dtype.to_jax_dtype()


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
