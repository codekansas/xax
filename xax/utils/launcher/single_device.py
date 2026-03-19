"""Single-device launcher helpers."""

import logging
import os
import shutil
import subprocess
from importlib import import_module

from xax.utils.devices import get_num_gpus
from xax.utils.logging import configure_logging


def _visible_gpu_indices_from_env() -> list[int] | None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    raw_value = os.environ["CUDA_VISIBLE_DEVICES"].strip()
    if raw_value == "":
        return []
    gpu_indices: list[int] = []
    for token in raw_value.split(","):
        token_stripped = token.strip()
        if not token_stripped.isdigit():
            return None
        gpu_indices.append(int(token_stripped))
    if len(set(gpu_indices)) != len(gpu_indices):
        return None
    return gpu_indices


def get_gpu_memory_info(visible_gpu_indices: set[int] | None = None) -> dict[int, tuple[float, float]]:
    """Get memory information for visible GPUs.

    Returns:
        Dictionary mapping GPU index to ``(total_memory_mb, used_memory_mb)``.
    """
    command = "nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader"
    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            gpu_info: dict[int, tuple[float, float]] = {}
            for line in stdout:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(", ")
                if len(parts) < 3:
                    continue
                gpu_id = int(parts[0])
                if visible_gpu_indices is not None and gpu_id not in visible_gpu_indices:
                    continue
                total_mem = float(parts[1].replace(" MiB", ""))
                used_mem = float(parts[2].replace(" MiB", ""))
                gpu_info[gpu_id] = (total_mem, used_mem)
            return gpu_info
    except Exception as e:
        logger = configure_logging()
        logger.warning("Failed to get GPU memory info: %s", e)
        return {}


def select_best_gpu() -> int | None:
    """Select the visible GPU with the most available memory."""
    visible_gpu_indices = _visible_gpu_indices_from_env()
    visible_gpu_index_set = None if visible_gpu_indices is None else set(visible_gpu_indices)
    gpu_info = get_gpu_memory_info(visible_gpu_index_set)
    if not gpu_info:
        return None

    best_gpu = None
    max_available_mem = -1.0
    for gpu_id, (total_mem, used_mem) in gpu_info.items():
        available_mem = total_mem - used_mem
        if available_mem > max_available_mem:
            max_available_mem = available_mem
            best_gpu = gpu_id
    return best_gpu


def configure_gpu_devices(logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = configure_logging()

    visible_gpu_indices = _visible_gpu_indices_from_env()
    num_gpus = len(visible_gpu_indices) if visible_gpu_indices is not None else get_num_gpus()

    if num_gpus > 1:
        logger.info("Multiple GPUs detected (%d), selecting GPU with most available memory", num_gpus)
        best_gpu = select_best_gpu()
        if best_gpu is None:
            logger.warning("Could not determine best GPU, using default device selection")
            return

        logger.info("Selected GPU %d for training", best_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        try:
            jax = import_module("jax")
            devices = jax.local_devices(backend="gpu")
            if devices:
                jax.config.update("jax_default_device", devices[0])
                logger.info("Configured JAX to use device: %s", devices[0])
        except Exception as e:
            logger.warning("Failed to configure JAX device: %s", e)
    elif num_gpus == 1:
        logger.info("Single GPU detected, using default device selection")


def configure_single_device(logger: logging.Logger | None = None) -> None:
    """Configure a process for single-GPU execution when GPUs are available."""
    if logger is None:
        logger = configure_logging()
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found, cannot configure single-device launcher")
    configure_gpu_devices(logger)
