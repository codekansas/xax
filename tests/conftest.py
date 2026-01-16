"""Defines PyTest configuration for the project."""

import functools
import os
import random
import shutil

# Force JAX to use CPU by default for tests.
# This must happen BEFORE importing JAX.
# Tests marked with @pytest.mark.gpu will be skipped if no GPU is available.
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from _pytest.python import Function  # noqa: E402


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    random.seed(1337)
    np.random.seed(1337)


@functools.lru_cache()
def has_gpu() -> bool:
    """Check if a GPU is available.

    This checks by looking for CUDA devices. Since we set JAX_PLATFORMS=cpu
    by default, this function checks the environment to see if GPU was
    explicitly requested, or checks for CUDA availability via nvidia-smi.
    """
    # If user explicitly set JAX_PLATFORMS to include cuda/gpu, trust that
    if "JAX_PLATFORMS" in os.environ:
        platforms = os.environ["JAX_PLATFORMS"].lower()
        if "cuda" in platforms or "gpu" in platforms:
            try:
                return len(jax.local_devices(backend="gpu")) > 0
            except RuntimeError:
                return False

    # Otherwise, check if CUDA is available by looking for nvidia-smi
    if shutil.which("nvidia-smi") is not None:
        # nvidia-smi exists, likely has GPU
        return True

    return False


@functools.lru_cache()
def has_mps() -> bool:
    """Check if an MPS device is available."""
    try:
        jax.local_devices(backend="xla_mps")
        return True
    except RuntimeError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU (skipped if no GPU available)")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "has_gpu: mark test as requiring GPU (legacy, use 'gpu' instead)")
    config.addinivalue_line("markers", "has_mps: mark test as requiring MPS device")


def pytest_runtest_setup(item: Function) -> None:
    """Handle test setup, including GPU test skipping."""
    for mark in item.iter_markers():
        if mark.name in ("gpu", "has_gpu") and not has_gpu():
            pytest.skip("Skipping because this test requires a GPU and none is available")
        if mark.name == "has_mps" and not has_mps():
            pytest.skip("Skipping because this test requires an MPS device and none is available")


def pytest_collection_modifyitems(items: list[Function]) -> None:
    """Sort tests so slow tests run last."""
    items.sort(key=lambda x: x.get_closest_marker("slow") is not None)
