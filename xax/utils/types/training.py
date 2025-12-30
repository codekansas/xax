"""Defines types that are used during training."""

import enum
from typing import Any, Protocol, runtime_checkable

import optax
import orbax.checkpoint as ocp
from jaxtyping import Array


@runtime_checkable
class Optimizer(Protocol):
    def init(self, params: optax.Params) -> optax.OptState: ...

    def update(
        self,
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.OptState]: ...


class Precision(enum.Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


def as_shape_dtype(x: Any) -> Any:  # noqa: ANN401
    if isinstance(x, Array):
        return ocp.utils.to_shape_dtype_struct(x)
    return x
