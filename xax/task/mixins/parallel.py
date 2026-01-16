"""Defines a mixin for supporting parallel training."""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import (
    Generic,
    TypeVar,
)

import jax
import numpy as np
from jax.sharding import PartitionSpec as P

from xax.core.conf import field
from xax.task.base import BaseConfig, BaseTask

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class ParallelConfig(BaseConfig):
    model_parallel_size: int | None = field(None, help="Number of GPUs for tensor parallelism (model sharding)")


Config = TypeVar("Config", bound=ParallelConfig)


class ParallelMixin(BaseTask[Config], Generic[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def should_do_model_parallel(self) -> bool:
        """Provides a flag for whether to opt for model parallelism.

        This shorthand can be used for configuring the mesh to support model
        parallelism, instead of having to override the get_mesh method.

        Returns:
            True if model parallelism should be used, False otherwise.
        """
        return False

    def get_mesh(self) -> jax.sharding.Mesh:
        """Create a mesh for parallel training.

        If model_parallel_size > 1, creates a mesh with a 'model' axis for
        model parallelism. Otherwise creates a mesh with a 'batch' axis for
        data parallelism.

        Returns:
            JAX mesh configured for the requested parallelism strategy.
        """
        ndevices = jax.local_device_count()
        mp_size = self.config.model_parallel_size

        if mp_size is None:
            mp_size = ndevices if self.should_do_model_parallel() else 1

        if mp_size > 1:
            if ndevices < mp_size:
                raise ValueError(f"model_parallel_size ({mp_size}) > available devices ({ndevices})")

            # For model parallelism, create a mesh with 'model' axis
            # This shards model weights across GPUs
            devices = np.array(jax.devices()[:mp_size])
            return jax.sharding.Mesh(devices, axis_names=("model",))
        else:
            # Standard data parallelism with 'batch' axis
            return jax.make_mesh((ndevices,), axis_names=("batch",))

    def get_data_sharding(self, mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
        """Get sharding specification for batch data.

        For tensor parallelism (mesh has 'model' axis), data is replicated.
        For data parallelism (mesh has 'batch' axis), data is sharded.
        """
        if "model" in mesh.axis_names:
            # Tensor parallelism: replicate data across all model-parallel ranks
            return jax.sharding.NamedSharding(mesh, P())
        else:
            # Data parallelism: shard along batch dimension
            return jax.sharding.NamedSharding(mesh, P("batch"))

    def get_model_sharding(self, mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
        """Get default model sharding (replicated).

        For tensor parallelism, individual weights will be sharded differently
        based on their role (Q/K/V vs O projection, etc). This returns the
        default replicated sharding for weights that should not be sharded.
        """
        return jax.sharding.NamedSharding(mesh, P())
