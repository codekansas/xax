"""Defines a mixin for instantiating dataloaders."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generic, Iterator, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from jax.sharding import NamedSharding
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
    prefetch_buffer_size: int = field(16, help="Number of batches to prefetch in background thread")
    load_in_memory: bool = field(False, help="Load entire dataset into device memory for maximum throughput")
    raise_dataloader_errors: bool = field(False, help="If set, raise dataloader errors inside the worker processes")
    dataset_workers: int = field(II("xax.num_workers:-1"), help="Number of workers for loading training samples")
    shuffle_seed: int = field(1337, help="Seed for the shuffle")
    debug_dataloader: bool = field(False, help="Debug dataloaders")


Config = TypeVar("Config", bound=DataloadersConfig)


class StreamingBatchIterator(Iterator[Batch]):
    """Iterator that streams batches from a HuggingFace Dataset with prefetching.

    This iterator provides efficient data loading by:
    - Using HuggingFace's optimized batch iteration (ds.iter)
    - Shuffling dataset each epoch for randomization
    - Prefetching in a background thread to overlap with GPU compute
    - Optional dtype casting for reduced memory bandwidth

    Args:
        ds: HuggingFace Dataset to iterate over.
        batch_size: Number of samples per batch.
        sharding: JAX sharding specification for device placement.
        prefetch_size: Number of batches to buffer ahead.
        seed: Random seed for shuffling.
        data_dtype: Optional JAX dtype to cast floating point data to (e.g., jnp.float16
            or jnp.float8_e4m3fn for reduced H2D bandwidth).
    """

    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        sharding: NamedSharding,
        prefetch_size: int = 16,
        seed: int = 1337,
        data_dtype: jnp.dtype | None = None,
    ) -> None:
        self._ds = ds
        self._batch_size = batch_size
        self._sharding = sharding
        self._prefetch_size = prefetch_size
        self._seed = seed
        self._data_dtype = data_dtype

        self._queue: Queue[Batch | BaseException | None] = Queue(maxsize=prefetch_size)
        self._stop_event = Event()
        self._error: BaseException | None = None
        self._thread = Thread(target=self._worker, daemon=True, name="xax-streaming")
        self._thread.start()

    def _cast_batch(self, batch: dict) -> dict:
        """Cast floating point arrays to the configured data dtype."""
        if self._data_dtype is None:
            return batch

        result = {}
        for col, arr in batch.items():
            if np.issubdtype(arr.dtype, np.floating):
                # Cast to target dtype
                result[col] = arr.astype(self._data_dtype)
            else:
                result[col] = arr
        return result

    def _worker(self) -> None:
        """Background worker that loads and prepares batches."""
        try:
            epoch = 0

            while True:  # Infinite repeat
                # Shuffle dataset with different seed each epoch
                shuffled_ds = self._ds.shuffle(seed=self._seed + epoch)
                shuffled_ds.set_format("numpy")

                # Use HuggingFace's optimized batch iteration
                for batch in shuffled_ds.iter(batch_size=self._batch_size, drop_last_batch=True):
                    if self._stop_event.is_set():
                        return

                    # Convert to numpy arrays (may already be numpy from set_format)
                    batch = {col: np.asarray(batch[col]) for col in batch}

                    # Cast to target dtype if specified
                    batch = self._cast_batch(batch)

                    # Transfer to device
                    batch = jax.device_put(batch, self._sharding)
                    self._queue.put(batch)

                epoch += 1

        except BaseException as e:
            self._queue.put(e)
        finally:
            self._queue.put(None)

    def __iter__(self) -> "StreamingBatchIterator":
        return self

    def __next__(self) -> Batch:
        if self._error is not None:
            raise self._error

        try:
            item = self._queue.get(timeout=60.0)
        except Empty:
            raise RuntimeError("Streaming iterator timed out waiting for data") from None

        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            self._error = item
            raise item

        return item

    def close(self) -> None:
        """Stop the background thread and clean up resources."""
        self._stop_event.set()
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
            pass


class InMemoryBatchIterator(Iterator[Batch]):
    """Iterator that yields batches from pre-batched JAX arrays on device.

    This iterator stores the dataset pre-organized into batches on device.
    During training, it simply returns slices of the pre-batched arrays,
    completely eliminating data loading overhead.

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
        # Simple slice - data is already on device
        batch = jax.tree.map(lambda x: x[batch_idx], self._data)
        self._current_idx += 1

        return batch

    def get_stacked_batches(self, n: int) -> Batch:
        """Get n batches pre-stacked into a single array.

        This is more efficient than calling __next__ n times and stacking,
        as it uses a single gather operation.

        Args:
            n: Number of batches to get.

        Returns:
            PyTree with shape (n, batch_size, ...).
        """
        if self._current_idx + n > self._num_batches:
            # Reshuffle for next epoch
            self._key, subkey = jax.random.split(self._key)
            self._batch_order = jax.random.permutation(subkey, self._num_batches)
            self._current_idx = 0

        # Get indices for n batches
        indices = self._batch_order[self._current_idx : self._current_idx + n]
        self._current_idx += n

        # Single gather operation for all batches
        stacked = jax.tree.map(lambda x: x[indices], self._data)
        return stacked


def load_dataset_in_memory(
    ds: Dataset,
    batch_size: int,
    sharding: NamedSharding,
    data_dtype: jnp.dtype | None = None,
) -> Batch:
    """Load a HuggingFace Dataset entirely into JAX arrays, pre-batched.

    The data is organized into batches at load time, eliminating all
    per-batch data loading overhead during training.

    Args:
        ds: HuggingFace Dataset to load.
        batch_size: Size of each batch.
        sharding: JAX sharding specification for device placement.
        data_dtype: Optional dtype to cast floating point data to.

    Returns:
        PyTree with shape (num_batches, batch_size, ...).
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

    # Reshape to (num_batches, batch_size, ...)
    def reshape_to_batches(arr: np.ndarray) -> np.ndarray:
        truncated = arr[: num_batches * batch_size]
        new_shape = (num_batches, batch_size) + truncated.shape[1:]
        return truncated.reshape(new_shape)

    batched_data = {key: reshape_to_batches(arr) for key, arr in data.items()}

    # Cast floating point data to target dtype if specified
    if data_dtype is not None:
        batched_data = {
            key: arr.astype(data_dtype) if np.issubdtype(arr.dtype, np.floating) else arr
            for key, arr in batched_data.items()
        }

    batched_data = jax.device_put(batched_data, sharding)

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
    def get_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        """Returns the dataset.

        Returns:
            The dataset to train on.
        """

    def get_streaming_iterator(
        self,
        sharding: NamedSharding,
        data_dtype: jnp.dtype | None = None,
    ) -> StreamingBatchIterator:
        """Get an iterator that streams batches from the dataset.

        This method creates a streaming iterator that loads batches in a
        background thread with shuffling and prefetching. Use this for
        datasets that are too large to fit in memory.

        Args:
            sharding: JAX sharding specification for device placement.
            data_dtype: Optional dtype to cast floating point data to.

        Returns:
            Iterator that yields batches from the dataset.
        """
        ds = self.get_dataset()

        if isinstance(ds, (IterableDataset, IterableDatasetDict)):
            raise NotImplementedError(
                "IterableDataset streaming is not yet implemented. Use a regular Dataset or set load_in_memory=True."
            )

        if isinstance(ds, DatasetDict):
            # Use the train split if available, otherwise first split
            if "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]

        if not isinstance(ds, Dataset):
            raise NotImplementedError(f"Unsupported dataset type: {type(ds)}")

        return StreamingBatchIterator(
            ds=ds,
            batch_size=self.batch_size,
            sharding=sharding,
            prefetch_size=self.config.prefetch_buffer_size,
            seed=self.config.shuffle_seed,
            data_dtype=data_dtype,
        )

    def get_in_memory_iterator(
        self,
        sharding: NamedSharding,
        key: PRNGKeyArray,
        data_dtype: jnp.dtype | None = None,
    ) -> InMemoryBatchIterator:
        """Load dataset into device memory and return an iterator.

        This method loads the entire dataset into JAX arrays on device,
        eliminating all data loading overhead during training. Only use
        this for datasets that fit comfortably in GPU memory.

        Args:
            sharding: JAX sharding specification for device placement.
            key: JAX random key for shuffling.
            data_dtype: Optional dtype to cast floating point data to.

        Returns:
            Iterator that yields batches from the in-memory dataset.
        """
        ds = self.get_dataset()

        if isinstance(ds, (IterableDataset, IterableDatasetDict)):
            raise ValueError(
                "Cannot load streaming/iterable datasets into memory. Set load_in_memory=False for streaming datasets."
            )

        if isinstance(ds, DatasetDict):
            # Use the train split if available, otherwise first split
            if "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]

        if not isinstance(ds, Dataset):
            raise NotImplementedError(f"Unsupported dataset type for in-memory loading: {type(ds)}")

        # Load dataset into device memory, pre-batched
        batched_data = load_dataset_in_memory(ds, self.batch_size, sharding, data_dtype)

        return InMemoryBatchIterator(batched_data=batched_data, key=key)
