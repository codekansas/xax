"""Defines a dataclass for keeping track of the current training state."""

import time
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict, Unpack, cast, get_args

import jax
import jax.numpy as jnp
from jaxtyping import Array
from omegaconf import MISSING

from xax.core.conf import field

# Batch = TypeVar("Batch")
# Output = TypeVar("Output")

Batch = Any
Output = Any

StepKind = Literal["step", "sample", "second"]


def cast_step_kind(s: str) -> StepKind:
    assert s in get_args(StepKind), f"`step_kind` must be one of {get_args(StepKind)}, not {s}"
    return cast(StepKind, s)


class StateDict(TypedDict, total=False):
    num_steps: NotRequired[int | Array]
    num_samples: NotRequired[int | Array]
    start_time_s: NotRequired[float | Array]
    elapsed_time_s: NotRequired[float | Array]
    last_log_time_s: NotRequired[float | Array]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class State:
    _int32_arr: Array = field(MISSING, static=False, help="Internal array for storing int64 values")
    _float32_arr: Array = field(MISSING, static=False, help="Internal array for storing floating-point values")

    @property
    def num_steps(self) -> Array:
        return self._int32_arr[0]

    @property
    def num_samples(self) -> Array:
        return self._float32_arr[0]

    @property
    def start_time_s(self) -> Array:
        return self._float32_arr[1]

    @property
    def elapsed_time_s(self) -> Array:
        return self._float32_arr[2]

    @property
    def last_log_time_s(self) -> Array:
        return self._float32_arr[3]

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            _int32_arr=jnp.array([0], dtype=jnp.int32),
            _float32_arr=jnp.array([0.0, time.time(), 0.0, 0.0], dtype=jnp.float32),
        )

    def replace(self, **kwargs: Unpack[StateDict]) -> "State":
        int32_arr = self._int32_arr
        float32_arr = self._float32_arr

        if "num_steps" in kwargs:
            int32_arr = int32_arr.at[0].set(kwargs["num_steps"])

        if "num_samples" in kwargs:
            float32_arr = float32_arr.at[0].set(kwargs["num_samples"])

        if "start_time_s" in kwargs:
            float32_arr = float32_arr.at[1].set(kwargs["start_time_s"])
        if "elapsed_time_s" in kwargs:
            float32_arr = float32_arr.at[2].set(kwargs["elapsed_time_s"])
        if "last_log_time_s" in kwargs:
            float32_arr = float32_arr.at[3].set(kwargs["last_log_time_s"])

        return State(
            _int32_arr=int32_arr,
            _float32_arr=float32_arr,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "num_steps": int(self.num_steps.item()),
            "num_samples": int(self.num_samples.item()),
            "start_time_s": float(self.start_time_s.item()),
            "elapsed_time_s": float(self.elapsed_time_s.item()),
        }

    @classmethod
    def from_dict(cls, **d: Unpack[StateDict]) -> "State":
        int32_arr = jnp.array(
            [
                d.get("num_steps", 0),
            ],
            dtype=jnp.int32,
        )

        float32_arr = jnp.array(
            [
                d.get("num_samples", 0),
                d.get("start_time_s", time.time()),
                d.get("elapsed_time_s", 0.0),
                d.get("last_log_time_s", 0.0),
            ],
            dtype=jnp.float32,
        )

        return cls(
            _int32_arr=int32_arr,
            _float32_arr=float32_arr,
        )
