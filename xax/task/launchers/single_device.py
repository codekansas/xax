"""Defines a launcher to train a model locally, on a single device."""

import logging
import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import jax

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.devices import get_num_gpus
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


def get_gpu_memory_info() -> dict[int, tuple[float, float]]:
    """Get memory information for all GPUs.

    Returns:
        Dictionary mapping GPU index to (total_memory_mb, used_memory_mb)
    """
    command = "nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader"

    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None

            gpu_info = {}
            for line in stdout:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(", ")
                if len(parts) >= 3:
                    gpu_id = int(parts[0])
                    total_mem = float(parts[1].replace(" MiB", ""))
                    used_mem = float(parts[2].replace(" MiB", ""))
                    gpu_info[gpu_id] = (total_mem, used_mem)

            return gpu_info

    except Exception as e:
        logger = configure_logging()
        logger.warning("Failed to get GPU memory info: %s", e)
        return {}


def select_best_gpu() -> int | None:
    """Select the GPU with the most available memory.

    Returns:
        GPU index with most available memory, or None if no GPUs found
    """
    gpu_info = get_gpu_memory_info()

    if not gpu_info:
        return None

    # Find GPU with most available memory
    best_gpu = None
    max_available: float = -1.0

    for gpu_id, (total_mem, used_mem) in gpu_info.items():
        available_mem = total_mem - used_mem
        if available_mem > max_available:
            max_available = available_mem
            best_gpu = gpu_id

    return best_gpu


def configure_gpu_devices(logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = configure_logging()

    # If there are multiple devices, choose the one with the most
    # available memory (i.e., the one which is likely not being used
    # by other processes) and use only that device.
    num_gpus = get_num_gpus()

    if num_gpus > 1:
        logger.info("Multiple GPUs detected (%d), selecting GPU with most available memory", num_gpus)

        best_gpu = select_best_gpu()
        if best_gpu is not None:
            logger.info("Selected GPU %d for training", best_gpu)

            # Set CUDA_VISIBLE_DEVICES to only show the selected GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

            # Configure JAX to use the selected device
            try:
                devices = jax.local_devices(backend="gpu")
                if devices:
                    jax.config.update("jax_default_device", devices[0])
                    logger.info("Configured JAX to use device: %s", devices[0])
            except Exception as e:
                logger.warning("Failed to configure JAX device: %s", e)
        else:
            logger.warning("Could not determine best GPU, using default device selection")
    elif num_gpus == 1:
        logger.info("Single GPU detected, using default device selection")


def configure_devices(logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = configure_logging()

    if shutil.which("nvidia-smi") is not None:
        configure_gpu_devices(logger)


def run_training(
    task: "type[RunnableMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logger = configure_logging()
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.add_logger_handlers(logger)
    task_obj.run()


class SingleDeviceLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        logger = configure_logging()
        configure_devices(logger)
        run_training(task, *cfgs, use_cli=use_cli, logger=logger)
