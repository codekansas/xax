"""Defines a launcher to train a model on multiple CPU devices for testing model parallelism."""

import logging
import os
from typing import TYPE_CHECKING

import jax

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


def configure_cpu_devices(num_cpus: int | None = None, logger: logging.Logger | None = None) -> None:
    """Configure JAX to use CPU devices only.

    Args:
        num_cpus: Number of CPU devices to use. If None, uses all available CPUs.
        logger: Optional logger instance.
    """
    if logger is None:
        logger = configure_logging()

    # Disable GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Configure the default device.
    devices = jax.devices("cpu")
    jax.config.update("jax_default_device", devices[0])

    # Configure number of CPU devices if specified.
    if num_cpus is not None:
        flag = "--xla_force_host_platform_device_count"
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = " ".join([f for f in xla_flags.split() if not f.startswith(flag)])
        xla_flags = f"{xla_flags} {flag}={num_cpus}".strip()
        os.environ["XLA_FLAGS"] = xla_flags
        logger.info("Configured XLA to use %d CPU devices (via XLA_FLAGS)", num_cpus)
    else:
        logger.info("Using default CPU device configuration")

    jax.config.update("jax_platform_name", "cpu")
    cpu_devices = jax.devices("cpu")
    logger.info("JAX configured to use %d CPU device(s): %s", len(cpu_devices), cpu_devices)


def run_training(
    task: "type[RunnableMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
    logger: logging.Logger | None = None,
    num_cpus: int | None = None,
) -> None:
    """Run training on CPU devices.

    Args:
        task: The task class to train
        cfgs: The raw configuration to use for training
        use_cli: Whether to include CLI arguments in the configuration
        logger: Optional logger instance
        num_cpus: Number of CPU devices to use. If None, uses all available CPUs.
    """
    if logger is None:
        logger = configure_logging()
    configure_cpu_devices(num_cpus=num_cpus, logger=logger)
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.add_logger_handlers(logger)
    task_obj.run()


class MultiCpuLauncher(BaseLauncher):
    """Launcher for multi-CPU training, useful for testing model parallelism without GPUs."""

    def __init__(self, num_cpus: int | None = None) -> None:
        """Initialize the multi-CPU launcher.

        Args:
            num_cpus: Number of CPU devices to use. If None, uses all available CPUs.
        """
        super().__init__()
        self.num_cpus = num_cpus

    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        logger = configure_logging()
        run_training(task, *cfgs, use_cli=use_cli, logger=logger, num_cpus=self.num_cpus)
