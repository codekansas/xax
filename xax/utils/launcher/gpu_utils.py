"""Shared GPU utility helpers for launcher and queue code."""

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

QUEUE_GPUS_ENV_VAR = "XAX_QUEUE_GPUS"
QUEUE_NUM_GPUS_ENV_VAR = "XAX_QUEUE_NUM_GPUS"
QUEUE_JOB_FLAG_ENV_VAR = "XAX_IN_QUEUE_JOB"
QUEUE_JOB_ID_ENV_VAR = "XAX_QUEUE_JOB_ID"


def parse_gpu_indices(raw_value: str, *, option_name: str = "GPU list") -> list[int]:
    if not raw_value.strip():
        raise ValueError(f"{option_name} cannot be empty")
    gpu_indices: list[int] = []
    for token in raw_value.split(","):
        token_stripped = token.strip()
        if not token_stripped:
            raise ValueError(f"{option_name} contains an empty GPU index")
        if not token_stripped.isdigit():
            raise ValueError(f"{option_name} contains a non-integer GPU index: {token_stripped!r}")
        gpu_indices.append(int(token_stripped))
    if len(set(gpu_indices)) != len(gpu_indices):
        raise ValueError(f"{option_name} contains duplicate GPU indices")
    return gpu_indices


def parse_cuda_visible_devices_env(env: Mapping[str, str] | None = None) -> list[int] | None:
    env_values = os.environ if env is None else env
    if "CUDA_VISIBLE_DEVICES" not in env_values:
        return None
    raw_value = env_values["CUDA_VISIBLE_DEVICES"].strip()
    if raw_value == "":
        return []
    try:
        return parse_gpu_indices(raw_value, option_name="CUDA_VISIBLE_DEVICES")
    except ValueError:
        return None


def discover_gpu_indices(
    *,
    treat_empty_cuda_visible_devices_as_no_gpu: bool = False,
    env: Mapping[str, str] | None = None,
) -> list[int]:
    env_values = os.environ if env is None else env
    if treat_empty_cuda_visible_devices_as_no_gpu:
        cuda_visible_devices = env_values.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None and cuda_visible_devices.strip() == "":
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


def read_process_cmdline(pid: int) -> list[str]:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return []
    payload = cmdline_path.read_bytes().split(b"\x00")
    return [token.decode("utf-8", errors="replace") for token in payload if token]


def read_process_environ(pid: int) -> dict[str, str]:
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


def extract_queue_gpu_args_from_cmdline(cmdline: list[str]) -> tuple[str | None, int | None]:
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
            queue_num_gpus_raw = token.split("=", maxsplit=1)[1]
            queue_num_gpus = int(queue_num_gpus_raw) if queue_num_gpus_raw.isdigit() else None
        elif token == "--queue-num-gpus" and idx + 1 < len(cmdline):
            queue_num_gpus_raw = cmdline[idx + 1].strip()
            queue_num_gpus = int(queue_num_gpus_raw) if queue_num_gpus_raw.isdigit() else None
            idx += 1
        idx += 1
    return queue_gpus, queue_num_gpus
