"""Defines a mixin for supporting parallel training."""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import (
    Generic,
    TypeVar,
)

import jax
from jax.sharding import PartitionSpec as P

from xax.task.base import BaseConfig, BaseTask

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class ParallelConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=ParallelConfig)


class ParallelMixin(BaseTask[Config], Generic[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get_mesh(self) -> jax.sharding.Mesh:
        ndevices = jax.local_device_count()
        return jax.make_mesh((ndevices,), axis_names=("batch",))

    def get_data_sharding(self, mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
        return jax.sharding.NamedSharding(mesh, P("batch"))

    def get_model_sharding(self, mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
        return jax.sharding.NamedSharding(mesh, P())
