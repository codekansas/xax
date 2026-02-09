"""CLI for managing queued jobs and the systemd-backed queue observer."""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import psutil
from tensorboard.backend.event_processing.event_accumulator import (
    SCALARS,
    TENSORS,
    EventAccumulator,
)
from tensorboard.util.tensor_util import make_ndarray

import xax

DEFAULT_POLL_SECONDS = 2.0
DEFAULT_HEARTBEAT_SECONDS = 3.0
DEFAULT_TENSORBOARD_PORT = 9249
DEFAULT_PROCESS_GRACE_SECONDS = 10.0
DEFAULT_SERVICE_NAME = "xax-queue-observer"

OOM_PATTERNS = (
    re.compile(r"cuda.*out of memory", re.IGNORECASE),
    re.compile(r"resource exhausted", re.IGNORECASE),
    re.compile(r"cudnn_status_alloc_failed", re.IGNORECASE),
    re.compile(r"xla.*out of memory", re.IGNORECASE),
)


@dataclass
class MetricPoint:
    tag: str
    step: int
    wall_time: float
    value: float
    source: str


@dataclass(frozen=True)
class ObserverArgs:
    poll_seconds: float = xax.field(DEFAULT_POLL_SECONDS, help="Polling interval while idle.")
    heartbeat_seconds: float = xax.field(DEFAULT_HEARTBEAT_SECONDS, help="Observer heartbeat interval.")
    once: bool = xax.field(False, help="Run at most one polling/dispatch cycle, then exit.")
    queue_gpus: str | None = xax.field(
        None,
        help=f"Comma-separated GPU indices reserved for queued jobs (default: ${xax.QUEUE_GPUS_ENV_VAR}).",
    )
    queue_num_gpus: int | None = xax.field(
        None,
        help=(
            "Reserve the first N detected GPUs for queued jobs "
            f"(default: ${xax.QUEUE_NUM_GPUS_ENV_VAR}; mutually exclusive with --queue-gpus)."
        ),
    )


@dataclass(frozen=True)
class QueueStatusArgs:
    recent: int = xax.field(10, help="Number of completed/failed/cancelled jobs to show.")
    as_json: bool = xax.field(
        False,
        help="Emit queue state as JSON.",
        metadata={xax.ARGPARSE_DEST_METADATA_KEY: "json"},
    )


@dataclass(frozen=True)
class QueueWaitArgs:
    poll_seconds: float = xax.field(DEFAULT_POLL_SECONDS, help="Polling interval while waiting for completion.")


@dataclass(frozen=True)
class MoveJobArgs:
    job_id: str = xax.field(help="Queued job id to move.", metadata={xax.CLI_POSITIONAL_METADATA_KEY: True})
    position: int = xax.field(help="1-based queue position.", metadata={xax.CLI_POSITIONAL_METADATA_KEY: True})


@dataclass(frozen=True)
class CancelJobArgs:
    job_id: str = xax.field(help="Queued job id to cancel.", metadata={xax.CLI_POSITIONAL_METADATA_KEY: True})


@dataclass(frozen=True)
class KillJobArgs:
    signal_name: str = xax.field(
        "TERM",
        help="Signal name to send (e.g. TERM, INT, KILL).",
        metadata={xax.ARGPARSE_DEST_METADATA_KEY: "signal"},
    )
    grace_seconds: float = xax.field(10.0, help="Seconds to wait before escalating to SIGKILL.")


@dataclass(frozen=True)
class TailJobArgs:
    job_id: str | None = xax.field(
        None,
        help="Job id to tail logs for. Defaults to the running job, otherwise the most recent job.",
        metadata={xax.CLI_POSITIONAL_METADATA_KEY: True},
    )
    kind: Literal["observer", "task"] = xax.field("observer", help="Log stream to read.")
    lines: int = xax.field(100, help="Number of tail lines to print before follow mode.")
    follow: bool = xax.field(False, help="Keep streaming new log lines.")


@dataclass(frozen=True)
class TensorboardArgs:
    job_ids: list[str] = xax.field(
        default_factory=list,
        help="Job ids to include. Defaults to running + queued jobs.",
        metadata={xax.CLI_POSITIONAL_METADATA_KEY: True},
    )
    all_jobs: bool = xax.field(
        False,
        help="Include all jobs that have TensorBoard data.",
        metadata={xax.ARGPARSE_DEST_METADATA_KEY: "all"},
    )
    port: int = xax.field(DEFAULT_TENSORBOARD_PORT, help="TensorBoard port.")
    bind_all: bool = xax.field(False, help="Bind to all interfaces, not just localhost.")


@dataclass(frozen=True)
class MetricsArgs:
    job_id: str | None = xax.field(
        None,
        help="Job id to read metrics from. Defaults to the running job, otherwise the most recent job.",
        metadata={xax.CLI_POSITIONAL_METADATA_KEY: True},
    )
    tag: str | None = xax.field(None, help="Optional metric tag filter.")
    as_json: bool = xax.field(
        False,
        help="Emit metric points as JSON.",
        metadata={xax.ARGPARSE_DEST_METADATA_KEY: "json"},
    )


@dataclass(frozen=True)
class ServiceStartArgs:
    name: str = xax.field(DEFAULT_SERVICE_NAME, help="systemd user service name (without .service).")
    queue_gpus: str | None = xax.field(
        None,
        help=f"Comma-separated GPU indices reserved for queued jobs (sets {xax.QUEUE_GPUS_ENV_VAR} in service).",
    )
    queue_num_gpus: int | None = xax.field(
        None,
        help=(
            "Reserve the first N detected GPUs for queued jobs "
            f"(sets {xax.QUEUE_NUM_GPUS_ENV_VAR} in service; mutually exclusive with --queue-gpus)."
        ),
    )


@dataclass(frozen=True)
class ServiceActionArgs:
    name: str = xax.field(DEFAULT_SERVICE_NAME, help="systemd user service name (without .service).")


@dataclass(frozen=True)
class RunJobArgs:
    job_id: str = xax.field(help="Internal queued job id.", metadata={xax.ARGPARSE_DEST_METADATA_KEY: "job_id"})


def _resolve_runnable_task_type(task_key: str) -> "type[xax.Task]":
    return xax.Task.from_task_key(task_key)


def _get_descendant_pids(root_pid: int) -> set[int]:
    try:
        process = psutil.Process(root_pid)
        children = process.children(recursive=True)
        return {int(child.pid) for child in children}
    except Exception:
        return set()


def _is_pid_live(pid: int) -> bool:
    try:
        return bool(psutil.pid_exists(pid))
    except Exception:
        return xax.is_pid_alive(pid)


def _kill_pid_list(pids: set[int], signum: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, int(signum))
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _kill_process_group(process_group_id: int, signum: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, int(signum))
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _terminate_job_processes(
    job: "xax.QueuedJob",
    extra_child_pids: set[int] | None = None,
    grace_seconds: float = DEFAULT_PROCESS_GRACE_SECONDS,
) -> list[int]:
    pid = job.pid
    process_group_id = job.process_group_id
    tracked_child_pids: set[int] = set(job.child_pids)
    if extra_child_pids is not None:
        tracked_child_pids |= set(extra_child_pids)
    if pid is not None:
        tracked_child_pids |= _get_descendant_pids(pid)

    if process_group_id is not None:
        _kill_process_group(process_group_id, signal.SIGTERM)
    _kill_pid_list(tracked_child_pids, signal.SIGTERM)
    if pid is not None:
        _kill_pid_list({pid}, signal.SIGTERM)

    deadline_s = time.time() + grace_seconds
    while time.time() < deadline_s:
        pid_live = pid is not None and _is_pid_live(pid)
        group_live = process_group_id is not None and xax.is_process_group_alive(process_group_id)
        child_live = any(_is_pid_live(child_pid) for child_pid in tracked_child_pids)
        if not pid_live and not group_live and not child_live:
            break
        time.sleep(0.2)

    if process_group_id is not None:
        _kill_process_group(process_group_id, signal.SIGKILL)
    if pid is not None:
        _kill_pid_list({pid}, signal.SIGKILL)
    _kill_pid_list(tracked_child_pids, signal.SIGKILL)

    leaked_pids = [child_pid for child_pid in sorted(tracked_child_pids) if _is_pid_live(child_pid)]
    if pid is not None and _is_pid_live(pid):
        leaked_pids = [pid, *leaked_pids]
    return leaked_pids


def _detect_oom_in_log(log_path: Path, max_lines: int = 300) -> str | None:
    if not log_path.exists():
        return None
    with open(log_path, encoding="utf-8", errors="replace") as log_file:
        tail_lines = deque(log_file, maxlen=max_lines)
    for line in reversed(tail_lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if any(pattern.search(line_stripped) for pattern in OOM_PATTERNS):
            return line_stripped[:500]
    return None


def _parse_queue_gpus(queue_gpus: str) -> list[int]:
    try:
        return xax.parse_gpu_indices(queue_gpus, option_name="`--queue-gpus`")
    except ValueError as error:
        if "cannot be empty" not in str(error):
            raise
        raise ValueError("`--queue-gpus` must be a comma-separated list of GPU indices (example: 0,1,2)") from error


def _discover_gpu_indices() -> list[int]:
    return xax.discover_gpu_indices(treat_empty_cuda_visible_devices_as_no_gpu=False)


def _resolve_queue_gpu_devices(
    queue_gpus: str | None,
    queue_num_gpus: int | None,
    *,
    use_env_defaults: bool = True,
) -> str | None:
    if queue_gpus is not None and queue_num_gpus is not None:
        raise ValueError("Pass at most one of --queue-gpus and --queue-num-gpus")

    resolved_queue_gpus = queue_gpus
    if resolved_queue_gpus is None and use_env_defaults:
        resolved_queue_gpus = os.environ.get(xax.QUEUE_GPUS_ENV_VAR)

    resolved_queue_num_gpus = queue_num_gpus
    if resolved_queue_num_gpus is None and use_env_defaults:
        queue_num_gpus_env = os.environ.get(xax.QUEUE_NUM_GPUS_ENV_VAR)
        if queue_num_gpus_env is not None and queue_num_gpus_env.strip():
            try:
                resolved_queue_num_gpus = int(queue_num_gpus_env.strip())
            except ValueError as error:
                raise ValueError(f"Invalid {xax.QUEUE_NUM_GPUS_ENV_VAR} value: {queue_num_gpus_env!r}") from error

    if resolved_queue_gpus is not None and resolved_queue_num_gpus is not None:
        raise ValueError(
            f"Pass at most one of --queue-gpus/{xax.QUEUE_GPUS_ENV_VAR} "
            f"and --queue-num-gpus/{xax.QUEUE_NUM_GPUS_ENV_VAR}"
        )

    if resolved_queue_gpus is not None:
        gpu_indices = _parse_queue_gpus(resolved_queue_gpus)
        return ",".join(str(gpu_idx) for gpu_idx in gpu_indices)

    if resolved_queue_num_gpus is None:
        return None
    if resolved_queue_num_gpus <= 0:
        raise ValueError("`--queue-num-gpus` must be >= 1")

    gpu_indices = _discover_gpu_indices()
    if not gpu_indices:
        raise RuntimeError(
            "Could not auto-detect GPUs with nvidia-smi for --queue-num-gpus. "
            "Pass --queue-gpus explicitly instead."
        )
    if resolved_queue_num_gpus > len(gpu_indices):
        raise ValueError(
            f"Requested --queue-num-gpus={resolved_queue_num_gpus}, but only detected {len(gpu_indices)} GPUs."
        )
    return ",".join(str(gpu_idx) for gpu_idx in gpu_indices[:resolved_queue_num_gpus])


def _canonicalize_service_gpu_options(
    queue_gpus: str | None,
    queue_num_gpus: int | None,
) -> tuple[str | None, int | None]:
    if queue_gpus is not None and queue_num_gpus is not None:
        raise ValueError("Pass at most one of --queue-gpus and --queue-num-gpus")
    if queue_gpus is not None:
        gpu_indices = _parse_queue_gpus(queue_gpus)
        return ",".join(str(gpu_idx) for gpu_idx in gpu_indices), None
    if queue_num_gpus is not None and queue_num_gpus <= 0:
        raise ValueError("`--queue-num-gpus` must be >= 1")
    return None, queue_num_gpus


def _read_process_cmdline(pid: int) -> list[str]:
    return xax.read_process_cmdline(pid)


def _read_process_environ(pid: int) -> dict[str, str]:
    return xax.read_process_environ(pid)


def _extract_queue_args_from_cmdline(cmdline: list[str]) -> tuple[str | None, int | None]:
    return xax.extract_queue_gpu_args_from_cmdline(cmdline)


def _observer_queue_gpu_devices(observer_info: "xax.ObserverInfo | None") -> str | None:
    if observer_info is None:
        return None

    pid = observer_info.pid
    queue_gpus_raw: str | None = None
    queue_num_gpus_raw: int | None = None

    try:
        process_env = _read_process_environ(pid)
    except Exception:
        process_env = {}
    if (queue_gpus_env := process_env.get(xax.QUEUE_GPUS_ENV_VAR)) is not None and queue_gpus_env.strip():
        queue_gpus_raw = queue_gpus_env.strip()
    if (queue_num_env := process_env.get(xax.QUEUE_NUM_GPUS_ENV_VAR)) is not None and queue_num_env.strip():
        try:
            queue_num_gpus_raw = int(queue_num_env.strip())
        except ValueError:
            queue_num_gpus_raw = None

    if queue_gpus_raw is None and queue_num_gpus_raw is None:
        try:
            queue_gpus_cmdline, queue_num_gpus_cmdline = _extract_queue_args_from_cmdline(_read_process_cmdline(pid))
        except Exception:
            queue_gpus_cmdline, queue_num_gpus_cmdline = None, None
        if queue_gpus_cmdline is not None and queue_gpus_cmdline.strip():
            queue_gpus_raw = queue_gpus_cmdline.strip()
        if queue_num_gpus_cmdline is not None:
            queue_num_gpus_raw = queue_num_gpus_cmdline

    if queue_gpus_raw is not None:
        try:
            gpu_indices = _parse_queue_gpus(queue_gpus_raw)
        except ValueError:
            return queue_gpus_raw
        return ",".join(str(gpu_idx) for gpu_idx in gpu_indices)

    if queue_num_gpus_raw is None:
        return None
    if queue_num_gpus_raw <= 0:
        return f"invalid ({queue_num_gpus_raw})"

    gpu_indices = _discover_gpu_indices()
    if not gpu_indices:
        return f"first {queue_num_gpus_raw} GPU(s)"

    queue_gpu_indices = gpu_indices[:queue_num_gpus_raw]
    if not queue_gpu_indices:
        return "none"
    return ",".join(str(gpu_idx) for gpu_idx in queue_gpu_indices)


def _format_epoch_seconds(epoch_seconds: float | None) -> str:
    if epoch_seconds is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_seconds))


def _truncate_table_cell(value: str, max_len: int) -> str:
    value = value.replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def _print_titled_table(out: "xax.CliOutput", title: str, headers: list[str], rows: list[list[str]]) -> None:
    out.table(title=title, headers=headers, rows=rows)


def _cleanup_stale_processes(out: "xax.CliOutput") -> None:
    with xax.edit_queue_state() as state:
        running_job_id = state.running_job_id
        for job_id, job in state.jobs.items():
            pid = job.pid
            process_group_id = job.process_group_id
            child_pids = set(job.child_pids)
            has_live_process = (
                (pid is not None and _is_pid_live(pid))
                or (process_group_id is not None and xax.is_process_group_alive(process_group_id))
                or any(_is_pid_live(child_pid) for child_pid in child_pids)
            )
            if not has_live_process:
                job.pid = None
                job.process_group_id = None
                job.child_pids = []
                continue

            leaked_pids = _terminate_job_processes(job, grace_seconds=DEFAULT_PROCESS_GRACE_SECONDS)
            job.pid = None
            job.process_group_id = None
            job.child_pids = []
            out.warning("Cleaned stale process tree for %s", job_id)

            if job_id == running_job_id:
                job.status = "failed"
                job.ended_at = time.time()
                job.return_code = -1
                job.error = "Recovered stale running process tree from previous observer session"
                state.running_job_id = None

            if leaked_pids:
                out.warning("Could not fully clean stale pids for %s: %s", job_id, leaked_pids)


def _run_queued_job(job_id: str) -> int:
    out = xax.get_cli_output(prefix="queue-job")
    job = xax.get_job(job_id)
    if job is None:
        out.error("Unknown queued job id: %s", job_id)
        return 1

    stage_dir = Path(job.stage_dir).expanduser().resolve()
    config_path = Path(job.config_path).expanduser().resolve()
    if not config_path.exists():
        out.error("Config snapshot missing for %s: %s", job_id, config_path)
        return 1
    if not stage_dir.exists():
        out.error("Staged environment missing for %s: %s", job_id, stage_dir)
        return 1

    stage_dir_str = str(stage_dir)
    if stage_dir_str not in sys.path:
        sys.path.insert(0, stage_dir_str)

    try:
        runnable_task_type = _resolve_runnable_task_type(job.task_key)
        launcher = xax.MultiDeviceLauncher()
        launcher.launch(runnable_task_type, config_path, use_cli=False)
    except Exception:
        out.exception("Queued job %s failed while executing", job_id)
        return 1
    return 0


def _spawn_job_process(
    job: "xax.QueuedJob",
    observer_pid: int,
    heartbeat_seconds: float,
    queue_gpu_devices: str | None = None,
) -> tuple[int, str | None, bool]:
    job_id = job.job_id
    run_dir = Path(job.run_dir).expanduser().resolve()
    stage_dir = Path(job.stage_dir).expanduser().resolve()
    observer_log_path = Path(job.observer_log_path).expanduser().resolve()
    observer_log_path.parent.mkdir(parents=True, exist_ok=True)

    submitter_python = job.python_executable
    command = [submitter_python, "-m", "xax.cli.queue", "_run-job", "--job-id", job_id]
    env = os.environ.copy()
    if pythonpath := env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{stage_dir}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(stage_dir)
    if queue_gpu_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = queue_gpu_devices
        env["NVIDIA_VISIBLE_DEVICES"] = queue_gpu_devices
        env[xax.QUEUE_GPUS_ENV_VAR] = queue_gpu_devices
    env[xax.QUEUE_JOB_FLAG_ENV_VAR] = "1"
    env[xax.QUEUE_JOB_ID_ENV_VAR] = job_id

    with open(observer_log_path, "a", encoding="utf-8") as observer_log_file:
        observer_log_file.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting {job_id} "
            f"(python={submitter_python})\n"
        )
        observer_log_file.flush()
        proc = subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
            stdout=observer_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    xax.set_running_job_pid(job_id, proc.pid, process_group_id=proc.pid)

    child_pids_seen: set[int] = set()
    last_heartbeat_s = 0.0
    while True:
        now_s = time.time()
        if now_s - last_heartbeat_s >= heartbeat_seconds:
            xax.touch_observer(observer_pid, status=f"running:{job_id}")
            last_heartbeat_s = now_s
        child_pids_seen |= _get_descendant_pids(proc.pid)
        if child_pids_seen:
            try:
                xax.set_running_job_children(job_id, list(child_pids_seen))
            except ValueError:
                pass
        if (return_code := proc.poll()) is not None:
            break
        time.sleep(0.5)

    child_pids_seen |= _get_descendant_pids(proc.pid)
    if child_pids_seen:
        try:
            xax.set_running_job_children(job_id, list(child_pids_seen))
        except ValueError:
            pass

    current_job = xax.get_job(job_id)
    process_job = job if current_job is None else current_job
    leaked_pids = _terminate_job_processes(
        process_job,
        extra_child_pids=child_pids_seen,
        grace_seconds=DEFAULT_PROCESS_GRACE_SECONDS,
    )
    try:
        xax.clear_job_process_tracking(job_id)
    except ValueError:
        pass

    oom_line = _detect_oom_in_log(observer_log_path)
    oom_detected = oom_line is not None

    error: str | None = None
    if return_code != 0:
        error = f"Exited with return code {return_code}" if return_code > 0 else f"Terminated by signal {-return_code}"
    if oom_line is not None:
        error = f"CUDA OOM detected: {oom_line}"
    if leaked_pids:
        leak_error = f"Leaked subprocesses after job exit: {leaked_pids}"
        error = leak_error if error is None else f"{error}; {leak_error}"

    if return_code == 0 and error is None:
        return 0, None, False
    return return_code, error, oom_detected


def _command_observer(args: ObserverArgs) -> int:
    out = xax.get_cli_output(prefix="queue")
    queue_gpu_devices = _resolve_queue_gpu_devices(args.queue_gpus, args.queue_num_gpus)
    observer_pid = os.getpid()
    xax.register_observer(observer_pid, status="idle")
    _cleanup_stale_processes(out)
    recovered_job = xax.recover_orphaned_running_job(reason="Recovered stale running job while observer restarted")
    if recovered_job is not None:
        out.warning("Recovered stale running job %s", recovered_job.job_id)

    out.status("Queue observer active (pid=%d, state=%s)", observer_pid, xax.get_queue_paths().state_path)
    if queue_gpu_devices is not None:
        out.status("Queue observer GPU allocation: %s", queue_gpu_devices)

    last_heartbeat_s = 0.0
    try:
        while True:
            now_s = time.time()
            if now_s - last_heartbeat_s >= args.heartbeat_seconds:
                xax.touch_observer(observer_pid, status="idle")
                last_heartbeat_s = now_s

            if (next_job := xax.claim_next_job()) is None:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue

            job_id = next_job.job_id
            xax.touch_observer(observer_pid, status=f"running:{job_id}")
            out.status("Starting queued job %s (%s)", job_id, next_job.task_key)

            try:
                return_code, error, oom_detected = _spawn_job_process(
                    next_job,
                    observer_pid=observer_pid,
                    heartbeat_seconds=args.heartbeat_seconds,
                    queue_gpu_devices=queue_gpu_devices,
                )
            except Exception as error_exc:
                out.exception("Observer failed while supervising %s", job_id)
                return_code, error, oom_detected = -1, str(error_exc), False
            finished_job = xax.finish_running_job(
                job_id,
                return_code=return_code,
                error=error,
                oom_detected=oom_detected,
            )

            if finished_job.status == "completed":
                out.status("Completed queued job %s", job_id)
            else:
                out.error("Queued job %s failed: %s", job_id, finished_job.error)

            if args.once:
                return 0
    finally:
        _cleanup_stale_processes(out)
        xax.recover_orphaned_running_job(reason="Recovered orphaned running job while observer stopping")
        xax.clear_observer(observer_pid)


def _serialize_job_for_output(job: "xax.QueuedJob", queue_idx: int) -> "dict[str, xax.JsonValue]":
    return {
        "job_id": job.job_id,
        "task_key": job.task_key,
        "launcher": job.launcher,
        "python_executable": job.python_executable,
        "status": job.status,
        "run_dir": job.run_dir,
        "stage_dir": job.stage_dir,
        "config_path": job.config_path,
        "observer_log_path": job.observer_log_path,
        "enqueued_at": job.enqueued_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "pid": job.pid,
        "process_group_id": job.process_group_id,
        "child_pids": [int(pid) for pid in job.child_pids],
        "oom_detected": job.oom_detected,
        "return_code": job.return_code,
        "error": job.error,
        "queue_index": queue_idx if job.status == "queued" else None,
    }


def _serialize_jobs_for_output(jobs: "list[xax.QueuedJob]") -> "list[dict[str, xax.JsonValue]]":
    jobs_payload: list[dict[str, xax.JsonValue]] = []
    for queue_idx, job in enumerate(jobs):
        jobs_payload.append(_serialize_job_for_output(job, queue_idx))
    return jobs_payload


def _serialize_observer_for_output(
    observer_info: "xax.ObserverInfo | None",
) -> "dict[str, xax.JsonValue] | None":
    if observer_info is None:
        return None
    return {
        "pid": observer_info.pid,
        "hostname": observer_info.hostname,
        "started_at": observer_info.started_at,
        "updated_at": observer_info.updated_at,
        "status": observer_info.status,
    }


def _command_status(args: QueueStatusArgs) -> int:
    out = xax.get_cli_output()
    state = xax.read_queue_state()
    observer_info = xax.read_observer_info()
    jobs = xax.list_jobs()

    if args.as_json:
        queue_paths = xax.get_queue_paths()
        payload = {
            "observer_active": xax.is_observer_active(ttl_seconds=xax.OBSERVER_HEARTBEAT_TTL_SECONDS),
            "observer": _serialize_observer_for_output(observer_info),
            "running_job_id": state.running_job_id,
            "queue": [job_id for job_id in state.queue],
            "jobs": _serialize_jobs_for_output(jobs),
            "paths": {key: str(value) for key, value in asdict(queue_paths).items()},
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    observer_rows: list[list[str]]
    if observer_info is None:
        observer_rows = [["inactive", "-", "-", "-", "-"]]
    else:
        age_s = time.time() - observer_info.updated_at
        observer_queue_gpus = _observer_queue_gpu_devices(observer_info)
        observer_rows = [
            [
                "active" if xax.is_observer_active() else "stale",
                str(observer_info.pid),
                f"{age_s:.1f}",
                observer_info.status,
                observer_queue_gpus if observer_queue_gpus is not None else "all/default",
            ]
        ]
    _print_titled_table(
        out,
        "Observer",
        ["state", "pid", "age_s", "status", "queue_gpus"],
        observer_rows,
    )

    running_job_id = state.running_job_id
    running_rows: list[list[str]]
    if running_job_id is None:
        running_rows = [["-", "-", "-", "none"]]
    else:
        running_job = state.jobs.get(running_job_id)
        if running_job is None:
            running_rows = [[running_job_id, "(missing metadata)", "-", "unknown"]]
        else:
            task_name = running_job.task_key.rsplit(".", maxsplit=1)[-1]
            running_rows = [[running_job_id, task_name, str(running_job.pid), running_job.status]]
    _print_titled_table(
        out,
        "Running Job",
        ["job_id", "task", "pid", "status"],
        running_rows,
    )

    queued_rows: list[list[str]] = []
    if not state.queue:
        queued_rows = [["-", "-", "-", "empty"]]
    else:
        for queue_idx, queued_job_id in enumerate(state.queue, start=1):
            queued_job = state.jobs.get(queued_job_id)
            if queued_job is None:
                queued_rows.append([str(queue_idx), queued_job_id, "(missing metadata)", "unknown"])
            else:
                task_name = queued_job.task_key.rsplit(".", maxsplit=1)[-1]
                queued_rows.append(
                    [
                        str(queue_idx),
                        queued_job_id,
                        _truncate_table_cell(task_name, 24),
                        queued_job.status,
                    ]
                )
    _print_titled_table(
        out,
        "Queued Jobs",
        ["idx", "job_id", "task", "status"],
        queued_rows,
    )

    recent_jobs = [job for job in jobs if job.status in ("completed", "failed", "cancelled")]
    table_rows: list[list[str]] = []
    if recent_jobs:
        recent_jobs.sort(key=lambda job: job.ended_at or 0.0, reverse=True)
        for job in recent_jobs[: args.recent]:
            task_name = job.task_key.rsplit(".", maxsplit=1)[-1]
            table_rows.append(
                [
                    job.job_id,
                    job.status,
                    _truncate_table_cell(task_name, 24),
                    "-" if job.return_code is None else str(job.return_code),
                    "yes" if job.oom_detected else "no",
                    _format_epoch_seconds(job.ended_at),
                    _truncate_table_cell(job.error or "", 48),
                ]
            )
    else:
        table_rows.append(["-", "none", "-", "-", "-", "-", "-"])
    _print_titled_table(
        out,
        "Recent Jobs",
        ["job_id", "status", "task", "return", "oom", "ended_at", "error"],
        table_rows,
    )
    return 0


def _command_wait(args: QueueWaitArgs) -> int:
    if args.poll_seconds <= 0:
        raise ValueError("poll_seconds must be > 0")

    out = xax.get_cli_output(prefix="queue")
    running_job = xax.get_running_job()
    if running_job is None:
        return 0

    waiting_job_id = running_job.job_id
    out.status("Waiting for running job %s to finish", waiting_job_id)

    while True:
        state = xax.read_queue_state()
        if state.running_job_id != waiting_job_id:
            break
        time.sleep(args.poll_seconds)

    finished_job = xax.get_job(waiting_job_id)
    if finished_job is None:
        out.status("Job %s finished", waiting_job_id)
        return 0

    out.status(
        "Job %s finished: status=%s return=%s",
        waiting_job_id,
        finished_job.status,
        "-" if finished_job.return_code is None else str(finished_job.return_code),
    )
    return 0


def _command_move(args: MoveJobArgs) -> int:
    position_idx = args.position - 1
    if position_idx < 0:
        raise ValueError("position must be >= 1")
    xax.move_queued_job(args.job_id, position_idx=position_idx)
    out = xax.get_cli_output(prefix="queue")
    out.status("Moved %s to queue position %d", args.job_id, args.position)
    return 0


def _command_cancel(args: CancelJobArgs) -> int:
    cancelled_job = xax.cancel_queued_job(args.job_id)
    out = xax.get_cli_output(prefix="queue")
    out.status("Cancelled queued job %s [%s]", cancelled_job.job_id, cancelled_job.task_key)
    return 0


def _signal_from_name(name: str) -> signal.Signals:
    normalized_name = name.upper()
    if not normalized_name.startswith("SIG"):
        normalized_name = f"SIG{normalized_name}"
    signum_raw = getattr(signal, normalized_name, None)
    if signum_raw is None or not isinstance(signum_raw, signal.Signals):
        raise ValueError(f"Unknown signal: {name}")
    return signum_raw


def _command_kill(args: KillJobArgs) -> int:
    out = xax.get_cli_output(prefix="queue")
    running_job = xax.get_running_job()
    if running_job is None:
        out.warning("No running job to kill")
        return 0
    pid = running_job.pid
    process_group_id = running_job.process_group_id
    if pid is None and process_group_id is None:
        out.warning("Running job %s has no process metadata", running_job.job_id)
        return 1

    signum = _signal_from_name(args.signal_name)
    if process_group_id is not None:
        _kill_process_group(process_group_id, signum)
    elif pid is not None:
        _kill_pid_list({pid}, signum)
    out.warning(
        "Sent %s to job %s (pid=%s, pgid=%s)",
        signum.name,
        running_job.job_id,
        pid,
        process_group_id,
    )

    if args.signal_name.upper() != "KILL" and args.grace_seconds > 0:
        deadline_s = time.time() + args.grace_seconds
        while time.time() < deadline_s:
            pid_live = pid is not None and _is_pid_live(pid)
            group_live = process_group_id is not None and xax.is_process_group_alive(process_group_id)
            if not pid_live and not group_live:
                return 0
            time.sleep(0.2)
        if process_group_id is not None:
            _kill_process_group(process_group_id, signal.SIGKILL)
        elif pid is not None:
            _kill_pid_list({pid}, signal.SIGKILL)
        out.warning("Escalated to SIGKILL for job %s", running_job.job_id)
    return 0


def _resolve_log_file(job: "xax.QueuedJob", kind: str) -> Path:
    run_dir = _resolve_existing_run_dir(job.run_dir)
    if kind == "observer":
        return Path(job.observer_log_path).expanduser().resolve()
    return run_dir / "logs.txt"


def _resolve_existing_run_dir(run_dir_raw: str) -> Path:
    run_dir = Path(run_dir_raw).expanduser()
    if run_dir.exists():
        return run_dir.resolve()

    run_roots: list[Path] = []
    if (runs_dir := xax.get_runs_dir()) is not None:
        run_roots.append(runs_dir.expanduser().resolve())
    # Compatibility fallback for environments that still expose RUN_DIR.
    if (run_dir_env := os.environ.get("RUN_DIR")) is not None and run_dir_env.strip():
        run_roots.append(Path(run_dir_env).expanduser().resolve())

    run_dir_resolved = run_dir.resolve(strict=False)
    for run_root in run_roots:
        run_root_parent = run_root.parent
        try:
            run_dir_relative = run_dir_resolved.relative_to(run_root_parent)
        except ValueError:
            continue
        fallback_run_dir = (run_root / run_dir_relative).expanduser().resolve()
        if fallback_run_dir.exists():
            return fallback_run_dir

    return run_dir.resolve(strict=False)


def _tail_file(path: Path, lines: int, follow: bool) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Log file does not exist: {path}")

    with open(path, encoding="utf-8", errors="replace") as log_file:
        recent_lines = deque(log_file, maxlen=lines)
    for line in recent_lines:
        sys.stdout.write(line)
    sys.stdout.flush()

    if not follow:
        return 0

    with open(path, encoding="utf-8", errors="replace") as log_file:
        log_file.seek(0, os.SEEK_END)
        while True:
            line = log_file.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            else:
                time.sleep(0.5)


def _running_or_latest_job() -> "xax.QueuedJob | None":
    if (running_job := xax.get_running_job()) is not None:
        return running_job
    jobs = xax.list_jobs()
    if not jobs:
        return None
    return max(
        jobs,
        key=lambda job: max(
            job.enqueued_at,
            job.started_at or job.enqueued_at,
            job.ended_at or job.enqueued_at,
        ),
    )


def _command_tail(args: TailJobArgs) -> int:
    if args.job_id is None:
        job = _running_or_latest_job()
        if job is None:
            raise ValueError("No queued jobs to tail. Pass a job id explicitly.")
    else:
        job = xax.get_job(args.job_id)
        if job is None:
            raise ValueError(f"Unknown job id: {args.job_id}")
    return _tail_file(_resolve_log_file(job, kind=args.kind), lines=args.lines, follow=args.follow)


def _resolve_jobs_for_tensorboard(job_ids: list[str], include_all: bool) -> "list[xax.QueuedJob]":
    if include_all:
        jobs = xax.list_jobs()
        return [job for job in jobs if (_resolve_existing_run_dir(job.run_dir) / "tensorboard").exists()]

    if job_ids:
        selected_jobs = [xax.get_job(job_id) for job_id in job_ids]
        jobs = [job for job in selected_jobs if job is not None]
        if len(jobs) != len(job_ids):
            missing_job_ids = [job_id for job_id, job in zip(job_ids, selected_jobs, strict=True) if job is None]
            raise ValueError(f"Unknown job id(s): {', '.join(missing_job_ids)}")
        return jobs

    state = xax.read_queue_state()
    selected_job_ids = [job_id for job_id in [state.running_job_id, *state.queue] if job_id is not None]
    jobs = [xax.get_job(job_id) for job_id in selected_job_ids]
    return [job for job in jobs if job is not None]


def _command_tensorboard(args: TensorboardArgs) -> int:
    jobs = _resolve_jobs_for_tensorboard(args.job_ids, include_all=args.all_jobs)
    logdir_specs: list[str] = []
    for job in jobs:
        tensorboard_dir = (_resolve_existing_run_dir(job.run_dir) / "tensorboard").resolve()
        if tensorboard_dir.exists():
            logdir_specs.append(f"{job.job_id}:{tensorboard_dir}")

    if not logdir_specs:
        raise RuntimeError("No TensorBoard directories found for selected jobs")

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "serve",
        "--port",
        str(args.port),
        "--reload_interval",
        "15",
    ]
    if args.bind_all:
        command.append("--bind_all")
    if len(logdir_specs) == 1:
        command += ["--logdir", logdir_specs[0].split(":", 1)[1]]
    else:
        command += ["--logdir_spec", ",".join(logdir_specs)]

    subprocess.run(command, check=False)
    return 0


def _event_subdirs(tensorboard_root: Path) -> list[tuple[str, Path]]:
    light_dir = tensorboard_root / "light"
    heavy_dir = tensorboard_root / "heavy"
    if light_dir.exists() or heavy_dir.exists():
        subdirs: list[tuple[str, Path]] = []
        if light_dir.exists():
            subdirs.append(("light", light_dir))
        if heavy_dir.exists():
            subdirs.append(("heavy", heavy_dir))
        return subdirs
    if tensorboard_root.exists():
        return [("root", tensorboard_root)]
    return []


def _collect_metric_points(tensorboard_root: Path) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for source_name, source_dir in _event_subdirs(tensorboard_root):
        accumulator = EventAccumulator(
            str(source_dir),
            size_guidance={
                SCALARS: 0,
                TENSORS: 0,
            },
        )
        accumulator.Reload()
        tags = accumulator.Tags()

        for tag in tags.get("scalars", []):
            for scalar_event in accumulator.Scalars(tag):
                points.append(
                    MetricPoint(
                        tag=tag,
                        step=int(scalar_event.step),
                        wall_time=float(scalar_event.wall_time),
                        value=float(scalar_event.value),
                        source=source_name,
                    )
                )

        for tag in tags.get("tensors", []):
            for tensor_event in accumulator.Tensors(tag):
                array = make_ndarray(tensor_event.tensor_proto)
                if array.size != 1:
                    continue
                scalar = array.reshape(())
                try:
                    value = float(scalar)
                except (TypeError, ValueError):
                    continue
                points.append(
                    MetricPoint(
                        tag=tag,
                        step=int(tensor_event.step),
                        wall_time=float(tensor_event.wall_time),
                        value=value,
                        source=source_name,
                    )
                )
    return points


def _latest_metric_points_by_tag(points: list[MetricPoint]) -> list[MetricPoint]:
    latest_by_tag: dict[str, MetricPoint] = {}
    for point in points:
        previous = latest_by_tag.get(point.tag)
        if previous is None or (point.step, point.wall_time, point.source) > (
            previous.step,
            previous.wall_time,
            previous.source,
        ):
            latest_by_tag[point.tag] = point
    return [latest_by_tag[tag] for tag in sorted(latest_by_tag)]


def _command_metrics(args: MetricsArgs) -> int:
    if args.job_id is None:
        job = _running_or_latest_job()
        if job is None:
            raise ValueError("No queued jobs to read metrics from. Pass a job id explicitly.")
    else:
        job = xax.get_job(args.job_id)
        if job is None:
            raise ValueError(f"Unknown job id: {args.job_id}")
    job_id = job.job_id

    tensorboard_root = _resolve_existing_run_dir(job.run_dir) / "tensorboard"
    if not tensorboard_root.exists():
        raise FileNotFoundError(f"TensorBoard directory does not exist for {job_id}: {tensorboard_root}")

    points = _collect_metric_points(tensorboard_root)
    if args.tag is not None:
        points = [point for point in points if point.tag == args.tag]
    points = _latest_metric_points_by_tag(points)

    if args.as_json:
        payload = [asdict(point) for point in points]
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    out = xax.get_cli_output(prefix="queue")
    if not points:
        out.warning("No scalar metrics found for job %s", job_id)
        return 0
    out.table(
        title=f"Metrics ({job_id})",
        headers=["tag", "step", "value", "source", "wall_time"],
        rows=[
            [
                _truncate_table_cell(point.tag, 40),
                str(point.step),
                f"{point.value:g}",
                point.source,
                f"{point.wall_time:.3f}",
            ]
            for point in points
        ],
    )
    return 0


def _service_unit_text(
    service_name: str,
    *,
    queue_gpus: str | None = None,
    queue_num_gpus: int | None = None,
) -> str:
    if queue_gpus is not None and queue_num_gpus is not None:
        raise ValueError("Pass at most one of `queue_gpus` and `queue_num_gpus`")

    lines = [
        "[Unit]",
        "Description=XAX Local Queue Observer",
        "After=network-online.target",
        "StartLimitIntervalSec=0",
        "",
        "[Service]",
        "Type=simple",
        f"ExecStart={sys.executable} -m xax.cli.main queue _observer",
        "Restart=always",
        "RestartSec=2s",
        "TimeoutStopSec=30s",
        "KillMode=control-group",
        "SendSIGKILL=yes",
        "OOMPolicy=continue",
        "Environment=PYTHONUNBUFFERED=1",
        f"Environment=XAX_HOME={xax.get_user_global_dir()}",
    ]
    if queue_gpus is not None:
        lines.append(f"Environment={xax.QUEUE_GPUS_ENV_VAR}={queue_gpus}")
    if queue_num_gpus is not None:
        lines.append(f"Environment={xax.QUEUE_NUM_GPUS_ENV_VAR}={queue_num_gpus}")

    lines.extend(
        [
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
            f"# Service name: {service_name}",
        ]
    )
    return "\n".join(lines)


def _service_path(service_name: str) -> Path:
    return _service_dir() / f"{service_name}.service"


def _run_systemctl_command(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *argv],
        check=False,
        capture_output=True,
        text=True,
    )


def _service_dir() -> Path:
    return Path("~/.config/systemd/user").expanduser().resolve()


def _resolve_start_service_gpu_options(
    args: ServiceStartArgs,
    service_path: Path,
) -> tuple[str | None, int | None, bool]:
    if args.queue_gpus is not None or args.queue_num_gpus is not None:
        queue_gpus, queue_num_gpus = _canonicalize_service_gpu_options(args.queue_gpus, args.queue_num_gpus)
        return queue_gpus, queue_num_gpus, True

    if service_path.exists():
        return None, None, False

    env_queue_gpus = os.environ.get(xax.QUEUE_GPUS_ENV_VAR)
    queue_gpus = env_queue_gpus.strip() if env_queue_gpus is not None and env_queue_gpus.strip() else None
    queue_num_gpus: int | None = None
    env_queue_num_gpus = os.environ.get(xax.QUEUE_NUM_GPUS_ENV_VAR)
    if env_queue_num_gpus is not None and env_queue_num_gpus.strip():
        try:
            queue_num_gpus = int(env_queue_num_gpus.strip())
        except ValueError as error:
            raise ValueError(f"Invalid {xax.QUEUE_NUM_GPUS_ENV_VAR} value: {env_queue_num_gpus!r}") from error

    queue_gpus, queue_num_gpus = _canonicalize_service_gpu_options(queue_gpus, queue_num_gpus)
    return queue_gpus, queue_num_gpus, True


def _command_start_service(args: ServiceStartArgs) -> int:
    out = xax.get_cli_output(prefix="queue")
    if shutil.which("systemctl") is None:
        raise RuntimeError("systemctl is not available; queue service commands require user systemd")

    service_name = args.name
    service_path = _service_path(service_name)
    queue_gpu_devices, queue_num_gpus, should_write_unit = _resolve_start_service_gpu_options(args, service_path)

    if should_write_unit:
        service_path.parent.mkdir(parents=True, exist_ok=True)
        unit_text = _service_unit_text(
            service_name,
            queue_gpus=queue_gpu_devices,
            queue_num_gpus=queue_num_gpus,
        )
        service_path.write_text(unit_text, encoding="utf-8")
        out.status("Wrote service unit: %s", service_path)

    daemon_reload_proc = _run_systemctl_command("daemon-reload")
    if daemon_reload_proc.returncode != 0:
        error_message = daemon_reload_proc.stderr.strip() or daemon_reload_proc.stdout.strip()
        raise RuntimeError(f"systemctl daemon-reload failed: {error_message}")
    start_proc = _run_systemctl_command("start", f"{service_name}.service")
    if start_proc.returncode != 0:
        error_message = start_proc.stderr.strip() or start_proc.stdout.strip()
        raise RuntimeError(f"systemctl start failed: {error_message}")
    out.status("Started service: %s.service", service_name)
    return 0


def _terminate_observer_process(out: "xax.CliOutput", grace_seconds: float = DEFAULT_PROCESS_GRACE_SECONDS) -> None:
    observer_info = xax.read_observer_info()
    if observer_info is None:
        return

    observer_pid = observer_info.pid
    if observer_pid == os.getpid() or not _is_pid_live(observer_pid):
        return

    _kill_pid_list({observer_pid}, signal.SIGTERM)
    deadline_s = time.time() + grace_seconds
    while time.time() < deadline_s:
        if not _is_pid_live(observer_pid):
            break
        time.sleep(0.2)
    if _is_pid_live(observer_pid):
        _kill_pid_list({observer_pid}, signal.SIGKILL)
    if _is_pid_live(observer_pid):
        out.warning("Observer process is still alive after SIGKILL: pid=%d", observer_pid)
    else:
        out.status("Stopped observer process: pid=%d", observer_pid)


def _stop_queue_runtime_state(out: "xax.CliOutput") -> None:
    _terminate_observer_process(out)
    _cleanup_stale_processes(out)
    recovered_job = xax.recover_orphaned_running_job(reason="Queue stopped while job was running")
    if recovered_job is not None:
        out.warning("Marked running job failed during stop: %s", recovered_job.job_id)
    xax.clear_observer(None)


def _command_stop_service(args: ServiceActionArgs) -> int:
    out = xax.get_cli_output(prefix="queue")
    service_name = args.name
    service = f"{service_name}.service"
    service_path = _service_path(service_name)
    systemctl_available = shutil.which("systemctl") is not None

    if systemctl_available:
        stop_proc = _run_systemctl_command("stop", service)
        if stop_proc.returncode != 0:
            stop_error = stop_proc.stderr.strip() or stop_proc.stdout.strip()
            if "not loaded" not in stop_error.lower() and "not found" not in stop_error.lower():
                raise RuntimeError(f"systemctl stop failed: {stop_error}")
        disable_proc = _run_systemctl_command("disable", service)
        if disable_proc.returncode != 0:
            disable_error = disable_proc.stderr.strip() or disable_proc.stdout.strip()
            if "not found" not in disable_error.lower():
                raise RuntimeError(f"systemctl disable failed: {disable_error}")

    _stop_queue_runtime_state(out)

    if service_path.exists():
        service_path.unlink()
        out.status("Removed service unit: %s", service_path)
    else:
        out.warning("Service unit not found: %s", service_path)

    if systemctl_available:
        daemon_reload_proc = _run_systemctl_command("daemon-reload")
        if daemon_reload_proc.returncode != 0:
            reload_error = daemon_reload_proc.stderr.strip() or daemon_reload_proc.stdout.strip()
            raise RuntimeError(f"systemctl daemon-reload failed: {reload_error}")
        _run_systemctl_command("reset-failed", service)

    out.status("Stopped and uninstalled service: %s", service)
    return 0


def _command_restart_service(args: ServiceStartArgs) -> int:
    _command_stop_service(ServiceActionArgs(name=args.name))
    return _command_start_service(args)


def _is_systemctl_available() -> bool:
    return shutil.which("systemctl") is not None


def _systemctl_error(action: str) -> RuntimeError:
    return RuntimeError(f"systemctl is not available; cannot run `xax queue {action}`")


def _command_service_lifecycle(command: Literal["start", "stop", "restart"], sub_argv: list[str]) -> int:
    if command == "start":
        if not _is_systemctl_available():
            raise _systemctl_error("start")
        return _command_start_service(xax.parse_args_as(ServiceStartArgs, sub_argv))
    if command == "stop":
        return _command_stop_service(xax.parse_args_as(ServiceActionArgs, sub_argv))
    if command == "restart":
        if not _is_systemctl_available():
            raise _systemctl_error("restart")
        return _command_restart_service(xax.parse_args_as(ServiceStartArgs, sub_argv))
    raise ValueError(f"Unsupported service lifecycle command: {command}")


@dataclass(frozen=True)
class _CommandSpec:
    description: str
    args_type: type | None = None
    visible: bool = True


COMMAND_SPECS: dict[str, _CommandSpec] = {
    "start": _CommandSpec(
        description="Install (if needed) and start the queue observer user service.",
        args_type=ServiceStartArgs,
    ),
    "stop": _CommandSpec(
        description="Stop queue observer, kill running queued jobs, and uninstall service unit.",
        args_type=ServiceActionArgs,
    ),
    "restart": _CommandSpec(
        description="Stop+uninstall then reinstall+start the queue observer service.",
        args_type=ServiceStartArgs,
    ),
    "status": _CommandSpec(
        description="Show observer status, running job, queued jobs, and recent completions.",
        args_type=QueueStatusArgs,
    ),
    "wait": _CommandSpec(
        description="Wait for the currently running queued job to finish.",
        args_type=QueueWaitArgs,
    ),
    "move": _CommandSpec(
        description="Move a queued job to a new position.",
        args_type=MoveJobArgs,
    ),
    "cancel": _CommandSpec(
        description="Cancel a queued job before it starts.",
        args_type=CancelJobArgs,
    ),
    "kill": _CommandSpec(
        description="Send a signal to the currently running queued job.",
        args_type=KillJobArgs,
    ),
    "tail": _CommandSpec(
        description="Tail observer or task logs for a job.",
        args_type=TailJobArgs,
    ),
    "tensorboard": _CommandSpec(
        description="Serve TensorBoard for selected job directories.",
        args_type=TensorboardArgs,
    ),
    "metrics": _CommandSpec(
        description="Print scalar metrics from a job's TensorBoard logs.",
        args_type=MetricsArgs,
    ),
    "_observer": _CommandSpec(
        description="Internal observer loop command used by systemd.",
        args_type=ObserverArgs,
        visible=False,
    ),
    "_run-job": _CommandSpec(
        description="Internal command for running a claimed queued job.",
        args_type=RunJobArgs,
        visible=False,
    ),
}


def _command_help_text(command: str) -> str:
    if command not in COMMAND_SPECS:
        raise ValueError(f"Unknown queue command: {command}")
    spec = COMMAND_SPECS[command]
    if spec.args_type is None:
        return "\n".join(
            [
                f"Usage: xax queue {command}",
                "",
                spec.description,
            ]
        )
    return xax.render_help_text(
        spec.args_type,
        prog=f"xax queue {command}",
        description=spec.description,
    )


def _show_queue_help(out: "xax.CliOutput") -> None:
    out.plain("Usage: xax queue <command> [args]")
    out.plain("")
    out.table(
        title="Queue Commands",
        headers=["command", "description"],
        rows=[
            [command_name, spec.description]
            for command_name, spec in COMMAND_SPECS.items()
            if spec.visible
        ],
    )
    out.plain("Run `xax queue <command> --help` for command usage.")


def main(argv: list[str] | None = None) -> None:
    out = xax.get_cli_output()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] in ("-h", "--help"):
        _show_queue_help(out)
        raise SystemExit(0)

    command, sub_argv = argv_list[0], argv_list[1:]
    if command not in COMMAND_SPECS:
        out.error("Unknown queue command: %s", command)
        _show_queue_help(out)
        raise SystemExit(2)

    if any(token in ("-h", "--help") for token in sub_argv):
        out.plain(_command_help_text(command))
        raise SystemExit(0)

    try:
        match command:
            case "_observer":
                return_code = int(_command_observer(xax.parse_args_as(ObserverArgs, sub_argv)))
            case "start":
                return_code = int(_command_service_lifecycle("start", sub_argv))
            case "stop":
                return_code = int(_command_service_lifecycle("stop", sub_argv))
            case "restart":
                return_code = int(_command_service_lifecycle("restart", sub_argv))
            case "status":
                return_code = int(_command_status(xax.parse_args_as(QueueStatusArgs, sub_argv)))
            case "wait":
                return_code = int(_command_wait(xax.parse_args_as(QueueWaitArgs, sub_argv)))
            case "move":
                return_code = int(_command_move(xax.parse_args_as(MoveJobArgs, sub_argv)))
            case "cancel":
                return_code = int(_command_cancel(xax.parse_args_as(CancelJobArgs, sub_argv)))
            case "kill":
                return_code = int(_command_kill(xax.parse_args_as(KillJobArgs, sub_argv)))
            case "tail":
                return_code = int(_command_tail(xax.parse_args_as(TailJobArgs, sub_argv)))
            case "tensorboard":
                return_code = int(_command_tensorboard(xax.parse_args_as(TensorboardArgs, sub_argv)))
            case "metrics":
                return_code = int(_command_metrics(xax.parse_args_as(MetricsArgs, sub_argv)))
            case "_run-job":
                run_args = xax.parse_args_as(RunJobArgs, sub_argv)
                return_code = int(_run_queued_job(run_args.job_id))
            case _:
                out.error("Unsupported command: %s", command)
                return_code = 2
    except KeyboardInterrupt:
        return_code = 130
    except Exception as error:
        out = xax.get_cli_output(prefix="queue")
        out.error("Command failed: %s", error)
        return_code = 1
    raise SystemExit(return_code)


if __name__ == "__main__":
    # python -m xax.cli.queue
    main()
