"""Defines a mixin for instantiating dataloaders."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar

import jax
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from dpshdl.collate import collate
from dpshdl.dataset import Dataset as DatasetWrapper
from jax.sharding import NamedSharding
from omegaconf import II

from xax.core.conf import field
from xax.core.state import Batch
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
    dataset_workers: int = field(II("xax.num_workers:-1"), help="Number of workers for loading training samples")
    shuffle_buffer_size: int = field(100, help="Size of the shuffle buffer")
    shuffle_seed: int = field(1337, help="Seed for the shuffle")
    debug_dataloader: bool = field(False, help="Debug dataloaders")


Config = TypeVar("Config", bound=DataloadersConfig)


def iter_samples(ds: IterableDataset, sharding: NamedSharding) -> Iterator[Batch]:
    def _tf_to_jax_tree(arr: Any) -> Any:  # noqa: ANN401
        return jax.device_put(arr, sharding)

    sample: Batch | None = None
    for next_sample in ds:
        if sample is None:
            sample = _tf_to_jax_tree(next_sample)
            continue
        next_sample = _tf_to_jax_tree(next_sample)
        yield sample
        sample = next_sample


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

    @abstractmethod
    def get_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        """Returns the dataset for the given phase.

        Args:
            phase: The phase for the dataset to return.

        Returns:
            The dataset for the given phase.
        """

    def get_data_iterator(self) -> IterableDataset:
        ds = self.get_dataset()

        if isinstance(ds, Dataset):
            ds = ds.to_iterable_dataset(num_shards=self.config.dataset_workers)

        if isinstance(ds, IterableDataset):
            ds = ds.batch(self.batch_size, drop_last_batch=True)
            ds = ds.repeat(None)
            ds = ds.shuffle(seed=self.config.shuffle_seed, buffer_size=self.config.shuffle_buffer_size)
            return ds

        raise NotImplementedError(f"Unsupported dataset type: {type(ds)}")

    @classmethod
    def collate_fn(cls, items: list[Any], config: Config) -> Batch:
        return collate(items)
