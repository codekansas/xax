"""Multi-CPU launcher helpers."""

import logging
import os

import jax

from xax.utils.logging import configure_logging


def configure_cpu_devices(num_cpus: int | None = None, logger: logging.Logger | None = None) -> None:
    """Configure JAX to use CPU devices only.

    Args:
        num_cpus: Number of CPU devices to use. If ``None``, uses all available CPUs.
        logger: Optional logger instance.
    """
    if logger is None:
        logger = configure_logging()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    devices = jax.local_devices(backend="cpu")
    jax.config.update("jax_default_device", devices[0])

    if num_cpus is not None:
        flag = "--xla_force_host_platform_device_count"
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = " ".join(flag_value for flag_value in xla_flags.split() if not flag_value.startswith(flag))
        xla_flags = f"{xla_flags} {flag}={num_cpus}".strip()
        os.environ["XLA_FLAGS"] = xla_flags
        logger.info("Configured XLA to use %d CPU devices (via XLA_FLAGS)", num_cpus)
    else:
        logger.info("Using default CPU device configuration")

    jax.config.update("jax_platform_name", "cpu")
    cpu_devices = jax.local_devices(backend="cpu")
    logger.info("JAX configured to use %d CPU device(s): %s", len(cpu_devices), cpu_devices)
