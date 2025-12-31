"""Defines a mixin for instantiating dataloaders."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generic, Iterator, TypeVar

import jax
import numpy as np
import tensorflow as tf
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from jax.sharding import NamedSharding, PartitionSpec as P
from omegaconf import II

from xax.core.conf import field
from xax.core.state import Batch
from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins.process import ProcessConfig, ProcessMixin

logger = logging.getLogger(__name__)
PRNGKeyArray = jax.Array

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)


@jax.tree_util.register_dataclass
@dataclass
class DataloadersConfig(ProcessConfig, BaseConfig):
    batch_size: int | None = field(None, help="Size of each batch")
    prefetch_buffer_size: int = field(4, help="Number of batches to prefetch in background thread")
    load_in_memory: bool = field(False, help="Load entire dataset into device memory for maximum throughput")
    raise_dataloader_errors: bool = field(False, help="If set, raise dataloader errors inside the worker processes")
    dataset_workers: int = field(II("xax.num_workers:-1"), help="Number of workers for loading training samples")
    shuffle_buffer_size: int = field(100, help="Size of the shuffle buffer")
    shuffle_seed: int = field(1337, help="Seed for the shuffle")
    debug_dataloader: bool = field(False, help="Debug dataloaders")


Config = TypeVar("Config", bound=DataloadersConfig)


class PrefetchIterator(Iterator[Batch]):
    """Prefetches batches in a background thread to overlap data loading with GPU compute.

    This iterator runs a background thread that continuously loads batches from the
    TensorFlow dataset, converts them to numpy, and transfers them to JAX devices.
    The prefetch buffer allows the GPU to keep computing while the next batches
    are being prepared.

    Args:
        ds: TensorFlow dataset to iterate over.
        sharding: JAX sharding specification for device placement.
        prefetch_size: Number of batches to buffer ahead. Larger values provide
            more overlap between data loading and GPU compute, at the cost of
            memory. A value of 4-8 is typically sufficient.
    """

    def __init__(self, ds: tf.data.Dataset, sharding: NamedSharding, prefetch_size: int = 4) -> None:
        self._ds_iter = iter(ds)
        self._sharding = sharding
        self._prefetch_size = prefetch_size
        self._queue: Queue[Batch | BaseException | None] = Queue(maxsize=prefetch_size)
        self._stop_event = Event()
        self._error: BaseException | None = None
        self._thread = Thread(target=self._worker, daemon=True, name="xax-prefetch")
        self._thread.start()

    def _worker(self) -> None:
        """Background worker that loads and prepares batches."""
        try:
            for sample in self._ds_iter:
                if self._stop_event.is_set():
                    break
                # Convert TF tensors to numpy arrays
                sample = jax.tree.map(lambda x: x.numpy() if isinstance(x, tf.Tensor) else x, sample)
                # Transfer to device(s) with specified sharding
                sample = jax.device_put(sample, self._sharding)
                self._queue.put(sample)
        except BaseException as e:
            # Store error to be raised in main thread
            self._queue.put(e)
        finally:
            # Signal end of iteration
            self._queue.put(None)

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> Batch:
        if self._error is not None:
            raise self._error

        try:
            item = self._queue.get(timeout=60.0)  # Timeout to prevent infinite hangs
        except Empty:
            raise RuntimeError("Prefetch queue timed out waiting for data") from None

        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            self._error = item
            raise item

        return item

    def close(self) -> None:
        """Stop the background thread and clean up resources."""
        self._stop_event.set()
        # Drain the queue to unblock the worker if it's waiting on put()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # Suppress errors during interpreter shutdown


def iter_samples(ds: tf.data.Dataset, sharding: NamedSharding, prefetch_size: int = 1) -> Iterator[Batch]:
    """Iterate over a TensorFlow dataset, yielding batches on JAX devices.

    Args:
        ds: TensorFlow dataset to iterate over.
        sharding: JAX sharding specification for device placement.
        prefetch_size: Number of batches to prefetch in a background thread.
            If 1, uses simple single-threaded iteration. If > 1, uses a
            background thread to overlap data loading with GPU compute.

    Yields:
        Batches transferred to the specified JAX devices.
    """
    if prefetch_size > 1:
        yield from PrefetchIterator(ds, sharding, prefetch_size)
    else:
        # Simple single-threaded implementation with 1-sample lookahead
        next_sample = None
        for sample in ds:
            sample = jax.tree.map(lambda x: x.numpy() if isinstance(x, tf.Tensor) else x, sample)
            if next_sample is None:
                next_sample = jax.device_put(sample, sharding)
                continue
            sample = jax.device_put(sample, sharding)
            yield next_sample
            next_sample = sample


class InMemoryBatchIterator(Iterator[Batch]):
    """Iterator that yields batches from pre-batched JAX arrays on device.

    This iterator stores the dataset pre-organized into batches and pre-sharded
    across devices. During training, it simply returns slices of the pre-sharded
    arrays, completely eliminating data loading and sharding overhead.

    Args:
        batched_data: PyTree of JAX arrays with shape (num_batches, batch_size, ...).
        key: JAX random key for shuffling batch order.
    """

    def __init__(
        self,
        batched_data: Batch,
        key: PRNGKeyArray,
    ) -> None:
        self._data = batched_data
        self._key = key

        # Get number of batches from first leaf
        first_leaf = jax.tree.leaves(batched_data)[0]
        self._num_batches = first_leaf.shape[0]

        # Initialize shuffled batch order
        self._key, subkey = jax.random.split(self._key)
        self._batch_order = jax.random.permutation(subkey, self._num_batches)
        self._current_idx = 0

    def __iter__(self) -> "InMemoryBatchIterator":
        return self

    def __next__(self) -> Batch:
        if self._current_idx >= self._num_batches:
            # Reshuffle batch order for next epoch
            self._key, subkey = jax.random.split(self._key)
            self._batch_order = jax.random.permutation(subkey, self._num_batches)
            self._current_idx = 0

        # Use Python int for indexing to avoid traced array overhead
        batch_idx = int(self._batch_order[self._current_idx])
        # Simple slice - no device_put needed, data is already sharded
        batch = jax.tree.map(lambda x: x[batch_idx], self._data)
        self._current_idx += 1

        return batch


def load_dataset_in_memory(ds: Dataset, batch_size: int, sharding: NamedSharding) -> Batch:
    """Load a HuggingFace Dataset entirely into JAX arrays, pre-batched and pre-sharded.

    The data is organized into batches and sharded across devices at load time.
    This eliminates all per-batch data loading and sharding overhead during training.

    Args:
        ds: HuggingFace Dataset to load.
        batch_size: Size of each batch.
        sharding: JAX sharding specification for device placement.

    Returns:
        PyTree of JAX arrays with shape (num_batches, batch_size, ...).
    """
    # Convert to numpy arrays
    ds.set_format("numpy")

    # Build a dict of arrays from the dataset
    data = {}
    for key in ds.column_names:
        col = ds[key]
        if isinstance(col, np.ndarray):
            data[key] = col
        else:
            # Handle nested structures or non-array columns
            data[key] = np.array(col)

    # Compute number of complete batches
    num_samples = len(ds)
    num_batches = num_samples // batch_size

    # Reshape into batches: (num_samples, ...) -> (num_batches, batch_size, ...)
    def reshape_to_batches(arr: np.ndarray) -> np.ndarray:
        # Truncate to complete batches and reshape
        truncated = arr[: num_batches * batch_size]
        new_shape = (num_batches, batch_size) + truncated.shape[1:]
        return truncated.reshape(new_shape)

    batched_data = {key: reshape_to_batches(arr) for key, arr in data.items()}

    batched_sharding = NamedSharding(sharding.mesh, P(None, "batch"))
    batched_data = jax.device_put(batched_data, batched_sharding)

    logger.info(
        "Loaded dataset into memory: %d samples -> %d batches of %d",
        num_samples,
        num_batches,
        batch_size,
    )
    return batched_data


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

    def get_in_memory_iterator(self, sharding: NamedSharding, key: PRNGKeyArray) -> InMemoryBatchIterator:
        """Load dataset into device memory and return an iterator.

        This method loads the entire dataset into JAX arrays on device,
        eliminating all data loading overhead during training. Only use
        this for datasets that fit comfortably in GPU memory.

        Args:
            sharding: JAX sharding specification for device placement.
            key: JAX random key for shuffling.

        Returns:
            Iterator that yields batches from the in-memory dataset.
        """
        ds = self.get_dataset()

        if isinstance(ds, (IterableDataset, IterableDatasetDict)):
            raise ValueError(
                "Cannot load streaming/iterable datasets into memory. Set load_in_memory=False for streaming datasets."
            )

        if isinstance(ds, tf.data.Dataset):
            raise ValueError(
                "Cannot load tf.data.Dataset into memory directly. "
                "Return a HuggingFace Dataset from get_dataset() instead."
            )

        if not isinstance(ds, Dataset):
            raise NotImplementedError(f"Unsupported dataset type for in-memory loading: {type(ds)}")

        # Load dataset into device memory, pre-batched and pre-sharded
        batched_data = load_dataset_in_memory(ds, self.batch_size, sharding)

        return InMemoryBatchIterator(batched_data=batched_data, key=key)
