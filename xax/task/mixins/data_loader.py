"""Defines a mixin for instantiating dataloaders."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar

import jax
import tensorflow as tf
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
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


def iter_samples(ds: tf.data.Dataset, sharding: NamedSharding) -> Iterator[Batch]:
    next_sample = None
    for sample in ds:
        sample = jax.tree.map(lambda x: x.numpy() if isinstance(x, tf.Tensor) else x, sample)
        if next_sample is None:
            next_sample = jax.device_put(sample, sharding)
            continue
        sample = jax.device_put(sample, sharding)
        yield next_sample
        next_sample = sample


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
    def get_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset | tf.data.Dataset:
        """Returns the dataset.

        Returns:
            The dataset to train on.
        """

    def get_tf_dataset(self) -> tf.data.Dataset:
        ds = self.get_dataset()

        if isinstance(ds, tf.data.Dataset):
            ds = ds.batch(self.batch_size, drop_remainder=True)
            ds = ds.repeat(None)
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            return ds

        if isinstance(ds, Dataset):
            tfds = ds.to_tf_dataset(
                batch_size=self.batch_size,
                shuffle=True,
                drop_remainder=True,
                prefetch=True,
            )
            tfds = tfds.repeat(None)
            return tfds

        if isinstance(ds, IterableDataset):
            raise NotImplementedError("IterableDataset is not implemented yet")

        raise NotImplementedError(f"Unsupported dataset type: {type(ds)}")
