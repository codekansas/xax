"""Defines a mixin for instantiating dataloaders."""

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeVar

import jax
import tensorflow as tf
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets.utils.tf_utils import minimal_tf_collate_fn
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import Array
from omegaconf import II

from xax.core.conf import field
from xax.core.state import Batch, Phase
from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins.process import ProcessConfig, ProcessMixin

logger = logging.getLogger(__name__)

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)


@jax.tree_util.register_dataclass
@dataclass
class DataloadersConfig(ProcessConfig, BaseConfig):
    batch_size: int | None = field(None, help="Size of each batch")
    raise_dataloader_errors: bool = field(False, help="If set, raise dataloader errors inside the worker processes")
    train_workers: int = field(II("xax.num_workers:-1"), help="Number of workers for loading training samples")
    valid_workers: int = field(1, help="Number of workers for loading validation samples")
    debug_dataloader: bool = field(False, help="Debug dataloaders")


Config = TypeVar("Config", bound=DataloadersConfig)


class DataloadersMixin(ProcessMixin[Config], BaseTask[Config], Generic[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get_batch_size(self) -> int:
        raise NotImplementedError(
            "When `batch_size` is not specified in your training config, you should override the `get_batch_size` "
            "method to return the desired training batch size."
        )

    @property
    def batch_size(self) -> int:
        if self.config.batch_size is not None:
            return self.config.batch_size
        return self.get_batch_size()

    def get_num_workers(self, phase: Phase) -> int:
        match phase:
            case "train":
                return self.config.train_workers
            case "valid":
                return self.config.valid_workers
            case _:
                raise KeyError(f"Unknown phase: {phase}")

    @abstractmethod
    def get_dataset(self, phase: Phase) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        """Returns the dataset for the given phase.

        Args:
            phase: The phase for the dataset to return.

        Returns:
            The dataset for the given phase.
        """

    def get_sharding(self) -> NamedSharding:
        ndevices = jax.local_device_count()
        mesh = jax.make_mesh((ndevices,), axis_names=("batch",))
        return jax.sharding.NamedSharding(mesh, P("batch"))

    def _tf_to_jax(self, arr: Any, sharding: NamedSharding) -> Array:  # noqa: ANN401
        if isinstance(arr, tf.Tensor):
            return jax.device_put(arr.numpy(), sharding)
        return arr

    def get_data_iterator(self, phase: Phase) -> Iterator[Batch]:
        ds = self.get_dataset(phase)
        sharding = self.get_sharding()

        def _tf_to_jax(arr: Any) -> Any:  # noqa: ANN401
            return jax.tree.map(functools.partial(self._tf_to_jax, sharding=sharding), arr)

        if isinstance(ds, DatasetDict):
            raise NotImplementedError("DatasetDict is not supported yet")

        elif isinstance(ds, Dataset):
            tfds = ds.to_tf_dataset(
                batch_size=self.batch_size,
                shuffle=phase == "train",
                collate_fn=functools.partial(self.collate_fn, config=self.config),
                num_workers=self.get_num_workers(phase),
                drop_remainder=True,
                prefetch=True,
                num_test_batches=3,
            )
            while True:
                yield from map(_tf_to_jax, tfds)

        elif isinstance(ds, IterableDataset):
            raise NotImplementedError("IterableDataset is not supported yet")

        elif isinstance(ds, IterableDatasetDict):
            raise NotImplementedError("IterableDatasetDict is not supported yet")

        else:
            raise NotImplementedError(f"Unsupported dataset type: {type(ds)}")

    @classmethod
    def collate_fn(cls, items: list[Any], config: Config) -> Batch:
        return minimal_tf_collate_fn(items)
