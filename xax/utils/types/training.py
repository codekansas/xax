"""Defines types that are used during training."""

import enum
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, PyTree

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
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


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
