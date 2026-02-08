"""Helpers for avoiding GPU collisions between queue jobs and local launchers."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from xax.utils.launcher.queue_state import is_observer_active, read_observer_info
from xax.utils.logging import configure_logging

QUEUE_GPUS_ENV_VAR = "XAX_QUEUE_GPUS"
QUEUE_NUM_GPUS_ENV_VAR = "XAX_QUEUE_NUM_GPUS"
QUEUE_JOB_FLAG_ENV_VAR = "XAX_IN_QUEUE_JOB"
QUEUE_JOB_ID_ENV_VAR = "XAX_QUEUE_JOB_ID"


def _parse_gpu_list(raw_value: str) -> list[int]:
    if not raw_value.strip():
        raise ValueError("GPU list cannot be empty")
    gpu_indices: list[int] = []
    for token in raw_value.split(","):
        token_stripped = token.strip()
        if not token_stripped:
            raise ValueError("GPU list contains an empty token")
        if not token_stripped.isdigit():
            raise ValueError(f"GPU list contains non-integer token: {token_stripped!r}")
        gpu_indices.append(int(token_stripped))
    if len(set(gpu_indices)) != len(gpu_indices):
        raise ValueError("GPU list contains duplicate entries")
    return gpu_indices


def _read_process_cmdline(pid: int) -> list[str]:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return []
    payload = cmdline_path.read_bytes().split(b"\x00")
    return [token.decode("utf-8", errors="replace") for token in payload if token]


def _read_process_environ(pid: int) -> dict[str, str]:
    environ_path = Path(f"/proc/{pid}/environ")
    if not environ_path.exists():
        return {}
    payload = environ_path.read_bytes().split(b"\x00")
    env_values: dict[str, str] = {}
    for token in payload:
        if not token:
            continue
        key, sep, value = token.partition(b"=")
        if not sep:
            continue
        env_values[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return env_values


def _extract_queue_args_from_cmdline(cmdline: list[str]) -> tuple[str | None, int | None]:
    queue_gpus: str | None = None
    queue_num_gpus: int | None = None
    idx = 0
    while idx < len(cmdline):
        token = cmdline[idx]
        if token.startswith("--queue-gpus="):
            queue_gpus = token.split("=", maxsplit=1)[1]
        elif token == "--queue-gpus" and idx + 1 < len(cmdline):
            queue_gpus = cmdline[idx + 1]
            idx += 1
        elif token.startswith("--queue-num-gpus="):
            queue_num_gpus = int(token.split("=", maxsplit=1)[1])
        elif token == "--queue-num-gpus" and idx + 1 < len(cmdline):
            queue_num_gpus = int(cmdline[idx + 1])
            idx += 1
        idx += 1
    return queue_gpus, queue_num_gpus


def _discover_gpu_indices() -> list[int]:
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"].strip() == "":
        return []
    if shutil.which("nvidia-smi") is None:
        return []
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    gpu_indices: list[int] = []
    for line in proc.stdout.splitlines():
        token = line.strip().split(",", maxsplit=1)[0].strip()
        if token.isdigit():
            gpu_indices.append(int(token))
    return sorted(set(gpu_indices))


def _queue_reserved_gpu_indices_from_observer() -> list[int] | None:
    if not is_observer_active():
        return None
    observer_info = read_observer_info()
    if observer_info is None:
        return None
    pid = observer_info["pid"]

    queue_gpus_raw: str | None = None
    queue_num_gpus_raw: int | None = None

    try:
        process_env = _read_process_environ(pid)
    except Exception:
        process_env = {}
    if (queue_gpus_env := process_env.get(QUEUE_GPUS_ENV_VAR)) is not None and queue_gpus_env.strip():
        queue_gpus_raw = queue_gpus_env.strip()
    if (queue_num_env := process_env.get(QUEUE_NUM_GPUS_ENV_VAR)) is not None and queue_num_env.strip():
        queue_num_gpus_raw = int(queue_num_env.strip())

    if queue_gpus_raw is None and queue_num_gpus_raw is None:
        try:
            queue_gpus_cmdline, queue_num_gpus_cmdline = _extract_queue_args_from_cmdline(_read_process_cmdline(pid))
        except Exception:
            queue_gpus_cmdline, queue_num_gpus_cmdline = None, None
        if queue_gpus_cmdline is not None:
            queue_gpus_raw = queue_gpus_cmdline
        if queue_num_gpus_cmdline is not None:
            queue_num_gpus_raw = queue_num_gpus_cmdline

    if queue_gpus_raw is not None and queue_num_gpus_raw is not None:
        raise ValueError("Queue observer has conflicting GPU settings")
    if queue_gpus_raw is not None:
        return _parse_gpu_list(queue_gpus_raw)
    if queue_num_gpus_raw is None:
        return []
    if queue_num_gpus_raw <= 0:
        raise ValueError("Queue observer queue-num-gpus must be >= 1")
    all_gpu_indices = _discover_gpu_indices()
    if not all_gpu_indices:
        return []
    if queue_num_gpus_raw > len(all_gpu_indices):
        raise ValueError(
            f"Queue observer requested {queue_num_gpus_raw} GPUs, but only detected {len(all_gpu_indices)} GPUs"
        )
    return all_gpu_indices[:queue_num_gpus_raw]


def _parse_cuda_visible_devices_env() -> list[int] | None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    raw_value = os.environ["CUDA_VISIBLE_DEVICES"].strip()
    if raw_value == "":
        return []
    try:
        return _parse_gpu_list(raw_value)
    except ValueError:
        return None


def _is_inside_queue_job() -> bool:
    if os.environ.get(QUEUE_JOB_FLAG_ENV_VAR, "").strip() == "1":
        return True
    if os.environ.get(QUEUE_JOB_ID_ENV_VAR):
        return True
    return False


def apply_queue_gpu_visibility(logger: logging.Logger | None = None) -> list[int] | None:
    """Masks queue-reserved GPUs for local launchers.

    Returns:
        The GPU indices that remain visible after masking, or None if no
        masking decision was applied.
    """
    if logger is None:
        logger = configure_logging()
    if _is_inside_queue_job():
        return None

    try:
        queue_reserved = _queue_reserved_gpu_indices_from_observer()
    except Exception as error:
        logger.warning("Failed to read queue GPU reservation; skipping GPU masking: %s", error)
        return None
    if queue_reserved is None:
        return None
    if not queue_reserved:
        return None

    currently_visible = _parse_cuda_visible_devices_env()
    if "CUDA_VISIBLE_DEVICES" in os.environ and currently_visible is None:
        logger.warning(
            "CUDA_VISIBLE_DEVICES=%r is not a simple integer list; leaving GPU visibility unchanged",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        return None
    if currently_visible is None:
        visible_pool = _discover_gpu_indices()
    else:
        visible_pool = currently_visible

    if not visible_pool:
        return []

    queue_reserved_set = set(queue_reserved)
    visible_non_queue = [gpu_idx for gpu_idx in visible_pool if gpu_idx not in queue_reserved_set]
    env_value = ",".join(str(gpu_idx) for gpu_idx in visible_non_queue)
    os.environ["CUDA_VISIBLE_DEVICES"] = env_value
    os.environ["NVIDIA_VISIBLE_DEVICES"] = env_value
    logger.info(
        "Queue observer is active. Hiding queue-reserved GPUs %s for local launch (visible=%s)",
        queue_reserved,
        visible_non_queue,
    )
    return visible_non_queue
