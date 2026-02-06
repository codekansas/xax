"""CLI for managing the local queued launcher observer and job queue."""

import argparse
import json
import logging
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
from typing import Callable, cast

import psutil
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util

from xax.core.conf import get_user_global_dir
from xax.task.base import BaseTask
from xax.task.launchers.base import BaseLauncher
from xax.task.launchers.dataset import DatasetLauncher
from xax.task.launchers.multi_cpu import MultiCpuLauncher
from xax.task.launchers.multi_device import MultiDeviceLauncher
from xax.task.launchers.queue_state import (
    OBSERVER_HEARTBEAT_TTL_SECONDS,
    QueuedJob,
    cancel_queued_job,
    claim_next_job,
    clear_job_process_tracking,
    clear_observer,
    edit_queue_state,
    finish_running_job,
    get_job,
    get_queue_paths,
    get_running_job,
    is_observer_active,
    is_pid_alive,
    is_process_group_alive,
    list_jobs,
    move_queued_job,
    read_observer_info,
    read_queue_state,
    recover_orphaned_running_job,
    register_observer,
    set_running_job_children,
    set_running_job_pid,
    touch_observer,
)
from xax.task.launchers.single_device import SingleDeviceLauncher
from xax.task.mixins.runnable import RunnableMixin
from xax.utils.logging import LOG_STATUS, configure_logging

DEFAULT_POLL_SECONDS = 2.0
DEFAULT_HEARTBEAT_SECONDS = 3.0
DEFAULT_TENSORBOARD_PORT = 9249
DEFAULT_PROCESS_GRACE_SECONDS = 10.0

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


def _get_launcher(launcher_name: str) -> BaseLauncher:
    match launcher_name:
        case "single":
            return SingleDeviceLauncher()
        case "multi":
            return MultiDeviceLauncher()
        case "multi_cpu":
            return MultiCpuLauncher()
        case "dataset":
            return DatasetLauncher()
        case _:
            raise ValueError(f"Unsupported queued job launcher: {launcher_name}")


def _get_descendant_pids(root_pid: int) -> set[int]:
    try:
        return {child.pid for child in psutil.Process(root_pid).children(recursive=True)}
    except psutil.Error:
        return set()


def _is_pid_live(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return is_pid_alive(pid)


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
    job: QueuedJob,
    extra_child_pids: set[int] | None = None,
    grace_seconds: float = DEFAULT_PROCESS_GRACE_SECONDS,
) -> list[int]:
    pid = job["pid"]
    process_group_id = job["process_group_id"]
    tracked_child_pids: set[int] = set(job["child_pids"])
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
        group_live = process_group_id is not None and is_process_group_alive(process_group_id)
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


def _cleanup_stale_processes(logger: logging.Logger) -> None:
    with edit_queue_state() as state:
        running_job_id = state["running_job_id"]
        for job_id, job in state["jobs"].items():
            pid = job["pid"]
            process_group_id = job["process_group_id"]
            child_pids = set(job["child_pids"])
            has_live_process = (
                (pid is not None and _is_pid_live(pid))
                or (process_group_id is not None and is_process_group_alive(process_group_id))
                or any(_is_pid_live(child_pid) for child_pid in child_pids)
            )
            if not has_live_process:
                job["pid"] = None
                job["process_group_id"] = None
                job["child_pids"] = []
                continue

            leaked_pids = _terminate_job_processes(job, grace_seconds=DEFAULT_PROCESS_GRACE_SECONDS)
            job["pid"] = None
            job["process_group_id"] = None
            job["child_pids"] = []
            logger.warning("Cleaned stale process tree for %s", job_id)

            if job_id == running_job_id:
                job["status"] = "failed"
                job["ended_at"] = time.time()
                job["return_code"] = -1
                job["error"] = "Recovered stale running process tree from previous observer session"
                state["running_job_id"] = None

            if leaked_pids:
                logger.warning("Could not fully clean stale pids for %s: %s", job_id, leaked_pids)


def _run_queued_job(job_id: str) -> int:
    logger = configure_logging(prefix="queue-job")
    job = get_job(job_id)
    if job is None:
        logger.error("Unknown queued job id: %s", job_id)
        return 1

    stage_dir = Path(job["stage_dir"]).expanduser().resolve()
    config_path = Path(job["config_path"]).expanduser().resolve()
    if not config_path.exists():
        logger.error("Config snapshot missing for %s: %s", job_id, config_path)
        return 1
    if not stage_dir.exists():
        logger.error("Staged environment missing for %s: %s", job_id, stage_dir)
        return 1

    stage_dir_str = str(stage_dir)
    if stage_dir_str not in sys.path:
        sys.path.insert(0, stage_dir_str)

    try:
        task_type = BaseTask.from_task_key(job["task_key"])
        if not issubclass(task_type, RunnableMixin):
            raise TypeError(f"Task for {job_id} is not runnable: {job['task_key']}")
        launcher = _get_launcher(job["launcher"])
        launcher.launch(task_type, config_path, use_cli=False)
    except Exception:
        logger.exception("Queued job %s failed while executing", job_id)
        return 1
    return 0


def _spawn_job_process(
    job: QueuedJob,
    observer_pid: int,
    heartbeat_seconds: float,
) -> tuple[int, str | None, bool]:
    job_id = job["job_id"]
    exp_dir = Path(job["exp_dir"]).expanduser().resolve()
    stage_dir = Path(job["stage_dir"]).expanduser().resolve()
    observer_log_path = Path(job["observer_log_path"]).expanduser().resolve()
    observer_log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "xax.cli.queue", "_run-job", "--job-id", job_id]
    env = os.environ.copy()
    if (pythonpath := env.get("PYTHONPATH")):
        env["PYTHONPATH"] = f"{stage_dir}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(stage_dir)

    with open(observer_log_path, "a", encoding="utf-8") as observer_log_file:
        observer_log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting {job_id}\n")
        observer_log_file.flush()
        proc = subprocess.Popen(
            command,
            cwd=exp_dir,
            env=env,
            stdout=observer_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    set_running_job_pid(job_id, proc.pid, process_group_id=proc.pid)

    child_pids_seen: set[int] = set()
    last_heartbeat_s = 0.0
    while True:
        now_s = time.time()
        if now_s - last_heartbeat_s >= heartbeat_seconds:
            touch_observer(observer_pid, status=f"running:{job_id}")
            last_heartbeat_s = now_s
        child_pids_seen |= _get_descendant_pids(proc.pid)
        if child_pids_seen:
            try:
                set_running_job_children(job_id, list(child_pids_seen))
            except ValueError:
                pass
        if (return_code := proc.poll()) is not None:
            break
        time.sleep(0.5)

    child_pids_seen |= _get_descendant_pids(proc.pid)
    if child_pids_seen:
        try:
            set_running_job_children(job_id, list(child_pids_seen))
        except ValueError:
            pass

    current_job = get_job(job_id)
    process_job = job if current_job is None else current_job
    leaked_pids = _terminate_job_processes(
        process_job,
        extra_child_pids=child_pids_seen,
        grace_seconds=DEFAULT_PROCESS_GRACE_SECONDS,
    )
    try:
        clear_job_process_tracking(job_id)
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


def _command_start(args: argparse.Namespace) -> int:
    logger = configure_logging(prefix="queue")
    observer_pid = os.getpid()
    register_observer(observer_pid, status="idle")
    _cleanup_stale_processes(logger)
    recovered_job = recover_orphaned_running_job(reason="Recovered stale running job while observer restarted")
    if recovered_job is not None:
        logger.warning("Recovered stale running job %s", recovered_job["job_id"])

    logger.log(LOG_STATUS, "Queue observer active (pid=%d, state=%s)", observer_pid, get_queue_paths().state_path)

    last_heartbeat_s = 0.0
    try:
        while True:
            now_s = time.time()
            if now_s - last_heartbeat_s >= args.heartbeat_seconds:
                touch_observer(observer_pid, status="idle")
                last_heartbeat_s = now_s

            if (next_job := claim_next_job()) is None:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue

            job_id = next_job["job_id"]
            touch_observer(observer_pid, status=f"running:{job_id}")
            logger.log(LOG_STATUS, "Starting queued job %s (%s)", job_id, next_job["task_key"])

            try:
                return_code, error, oom_detected = _spawn_job_process(
                    next_job,
                    observer_pid=observer_pid,
                    heartbeat_seconds=args.heartbeat_seconds,
                )
            except Exception as error_exc:
                logger.exception("Observer failed while supervising %s", job_id)
                return_code, error, oom_detected = -1, str(error_exc), False
            finished_job = finish_running_job(
                job_id,
                return_code=return_code,
                error=error,
                oom_detected=oom_detected,
            )

            if finished_job["status"] == "completed":
                logger.log(LOG_STATUS, "Completed queued job %s", job_id)
            else:
                logger.error("Queued job %s failed: %s", job_id, finished_job["error"])

            if args.once:
                return 0
    finally:
        clear_observer(observer_pid)


def _serialize_jobs_for_output(jobs: list[QueuedJob]) -> list[dict]:
    jobs_payload: list[dict] = []
    for queue_idx, job in enumerate(jobs):
        payload: dict = dict(job)
        payload["queue_index"] = queue_idx if job["status"] == "queued" else None
        jobs_payload.append(payload)
    return jobs_payload


def _command_status(args: argparse.Namespace) -> int:
    logger = configure_logging(prefix="queue")
    state = read_queue_state()
    observer_info = read_observer_info()
    jobs = list_jobs()

    if args.json:
        queue_paths = get_queue_paths()
        payload = {
            "observer_active": is_observer_active(ttl_seconds=OBSERVER_HEARTBEAT_TTL_SECONDS),
            "observer": observer_info,
            "running_job_id": state["running_job_id"],
            "queue": state["queue"],
            "jobs": _serialize_jobs_for_output(jobs),
            "paths": {key: str(value) for key, value in asdict(queue_paths).items()},
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    if observer_info is None:
        logger.warning("Observer: inactive")
    else:
        age_s = time.time() - observer_info["updated_at"]
        logger.log(
            LOG_STATUS,
            "Observer: %s (pid=%d, age=%.1fs, status=%s)",
            "active" if is_observer_active() else "stale",
            observer_info["pid"],
            age_s,
            observer_info["status"],
        )

    running_job_id = state["running_job_id"]
    if running_job_id is None:
        logger.info("Running: none")
    else:
        running_job = state["jobs"].get(running_job_id)
        if running_job is None:
            logger.warning("Running: %s (missing job metadata)", running_job_id)
        else:
            logger.info("Running: %s [%s] pid=%s", running_job_id, running_job["task_key"], running_job["pid"])

    if not state["queue"]:
        logger.info("Queued: empty")
    else:
        for queue_idx, queued_job_id in enumerate(state["queue"], start=1):
            queued_job = state["jobs"].get(queued_job_id)
            if queued_job is None:
                logger.warning("Queued[%d]: %s (missing metadata)", queue_idx, queued_job_id)
            else:
                logger.info("Queued[%d]: %s [%s]", queue_idx, queued_job_id, queued_job["task_key"])

    recent_jobs = [job for job in jobs if job["status"] in ("completed", "failed", "cancelled")]
    if recent_jobs:
        recent_jobs.sort(key=lambda job: job["ended_at"] or 0.0, reverse=True)
        for job in recent_jobs[: args.recent]:
            logger.info(
                "Recent: %s status=%s return=%s oom=%s error=%s",
                job["job_id"],
                job["status"],
                job["return_code"],
                job["oom_detected"],
                job["error"],
            )
    return 0


def _command_move(args: argparse.Namespace) -> int:
    position_idx = args.position - 1
    if position_idx < 0:
        raise ValueError("--position must be >= 1")
    move_queued_job(args.job_id, position_idx=position_idx)
    logger = configure_logging(prefix="queue")
    logger.log(LOG_STATUS, "Moved %s to queue position %d", args.job_id, args.position)
    return 0


def _command_cancel(args: argparse.Namespace) -> int:
    cancelled_job = cancel_queued_job(args.job_id)
    logger = configure_logging(prefix="queue")
    logger.log(LOG_STATUS, "Cancelled queued job %s [%s]", cancelled_job["job_id"], cancelled_job["task_key"])
    return 0


def _signal_from_name(name: str) -> signal.Signals:
    normalized_name = name.upper()
    if not normalized_name.startswith("SIG"):
        normalized_name = f"SIG{normalized_name}"
    signum_raw = getattr(signal, normalized_name, None)
    if signum_raw is None or not isinstance(signum_raw, signal.Signals):
        raise ValueError(f"Unknown signal: {name}")
    return signum_raw


def _command_kill(args: argparse.Namespace) -> int:
    logger = configure_logging(prefix="queue")
    running_job = get_running_job()
    if running_job is None:
        logger.warning("No running job to kill")
        return 0
    pid = running_job["pid"]
    process_group_id = running_job["process_group_id"]
    if pid is None and process_group_id is None:
        logger.warning("Running job %s has no process metadata", running_job["job_id"])
        return 1

    signum = _signal_from_name(args.signal)
    if process_group_id is not None:
        _kill_process_group(process_group_id, signum)
    elif pid is not None:
        _kill_pid_list({pid}, signum)
    logger.warning(
        "Sent %s to job %s (pid=%s, pgid=%s)",
        signum.name,
        running_job["job_id"],
        pid,
        process_group_id,
    )

    if args.signal.upper() != "KILL" and args.grace_seconds > 0:
        deadline_s = time.time() + args.grace_seconds
        while time.time() < deadline_s:
            pid_live = pid is not None and _is_pid_live(pid)
            group_live = process_group_id is not None and is_process_group_alive(process_group_id)
            if not pid_live and not group_live:
                return 0
            time.sleep(0.2)
        if process_group_id is not None:
            _kill_process_group(process_group_id, signal.SIGKILL)
        elif pid is not None:
            _kill_pid_list({pid}, signal.SIGKILL)
        logger.warning("Escalated to SIGKILL for job %s", running_job["job_id"])
    return 0


def _resolve_log_file(job: QueuedJob, kind: str) -> Path:
    exp_dir = Path(job["exp_dir"]).expanduser().resolve()
    if kind == "observer":
        return Path(job["observer_log_path"]).expanduser().resolve()
    return exp_dir / "logs.txt"


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


def _command_tail(args: argparse.Namespace) -> int:
    job = get_job(args.job_id)
    if job is None:
        raise ValueError(f"Unknown job id: {args.job_id}")
    return _tail_file(_resolve_log_file(job, kind=args.kind), lines=args.lines, follow=args.follow)


def _resolve_jobs_for_tensorboard(job_ids: list[str], include_all: bool) -> list[QueuedJob]:
    if include_all:
        jobs = list_jobs()
        return [job for job in jobs if (Path(job["exp_dir"]) / "tensorboard").exists()]

    if job_ids:
        selected_jobs = [get_job(job_id) for job_id in job_ids]
        jobs = [job for job in selected_jobs if job is not None]
        if len(jobs) != len(job_ids):
            missing_job_ids = [job_id for job_id, job in zip(job_ids, selected_jobs, strict=True) if job is None]
            raise ValueError(f"Unknown job id(s): {', '.join(missing_job_ids)}")
        return jobs

    state = read_queue_state()
    selected_job_ids = [job_id for job_id in [state["running_job_id"], *state["queue"]] if job_id is not None]
    jobs = [get_job(job_id) for job_id in selected_job_ids]
    return [job for job in jobs if job is not None]


def _command_tensorboard(args: argparse.Namespace) -> int:
    jobs = _resolve_jobs_for_tensorboard(args.job_ids, include_all=args.all)
    logdir_specs: list[str] = []
    for job in jobs:
        tensorboard_dir = (Path(job["exp_dir"]).expanduser().resolve() / "tensorboard").resolve()
        if tensorboard_dir.exists():
            logdir_specs.append(f"{job['job_id']}:{tensorboard_dir}")

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
        accumulator = event_accumulator.EventAccumulator(
            str(source_dir),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.TENSORS: 0,
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
                array = tensor_util.make_ndarray(tensor_event.tensor_proto)
                if array.size != 1:
                    continue
                points.append(
                    MetricPoint(
                        tag=tag,
                        step=int(tensor_event.step),
                        wall_time=float(tensor_event.wall_time),
                        value=float(array.reshape(())),
                        source=source_name,
                    )
                )
    return points


def _command_metrics(args: argparse.Namespace) -> int:
    job = get_job(args.job_id)
    if job is None:
        raise ValueError(f"Unknown job id: {args.job_id}")

    tensorboard_root = Path(job["exp_dir"]).expanduser().resolve() / "tensorboard"
    if not tensorboard_root.exists():
        raise FileNotFoundError(f"TensorBoard directory does not exist for {args.job_id}: {tensorboard_root}")

    points = _collect_metric_points(tensorboard_root)
    if args.tag is not None:
        points = [point for point in points if point.tag == args.tag]
    points.sort(key=lambda point: (point.tag, point.step, point.wall_time))

    if args.last is not None:
        grouped_points: dict[str, list[MetricPoint]] = {}
        for point in points:
            grouped_points.setdefault(point.tag, []).append(point)
        points = []
        for grouped_values in grouped_points.values():
            points.extend(grouped_values[-args.last :])
        points.sort(key=lambda point: (point.tag, point.step, point.wall_time))

    if args.json:
        payload = [asdict(point) for point in points]
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    logger = configure_logging(prefix="queue")
    if not points:
        logger.warning("No scalar metrics found for job %s", args.job_id)
        return 0
    for point in points:
        logger.info(
            "metric job=%s tag=%s step=%d value=%g source=%s",
            args.job_id,
            point.tag,
            point.step,
            point.value,
            point.source,
        )
    return 0


def _command_cleanup(_: argparse.Namespace) -> int:
    logger = configure_logging(prefix="queue")
    _cleanup_stale_processes(logger)
    recovered_job = recover_orphaned_running_job(reason="Recovered orphaned running job during manual cleanup")
    if recovered_job is not None:
        logger.warning("Recovered orphaned running job %s", recovered_job["job_id"])
    logger.log(LOG_STATUS, "Queue process cleanup completed")
    return 0


def _service_unit_text(service_name: str) -> str:
    return "\n".join(
        [
            "[Unit]",
            "Description=XAX Local Queue Observer",
            "After=network-online.target",
            "StartLimitIntervalSec=0",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStart={sys.executable} -m xax.cli.main queue start",
            f"ExecStopPost={sys.executable} -m xax.cli.main queue cleanup",
            "Restart=always",
            "RestartSec=2s",
            "TimeoutStopSec=30s",
            "KillMode=control-group",
            "SendSIGKILL=yes",
            "OOMPolicy=continue",
            "Environment=PYTHONUNBUFFERED=1",
            f"Environment=XAX_HOME={get_user_global_dir()}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
            f"# Service name: {service_name}",
        ]
    )


def _run_systemctl_command(*argv: str) -> int:
    if shutil.which("systemctl") is None:
        return 0
    proc = subprocess.run(["systemctl", "--user", *argv], check=False)
    return proc.returncode


def _command_systemd(args: argparse.Namespace) -> int:
    service_name = args.name
    unit_text = _service_unit_text(service_name)
    if not args.install:
        sys.stdout.write(unit_text + "\n")
        return 0

    service_dir = Path("~/.config/systemd/user").expanduser().resolve()
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / f"{service_name}.service"
    if service_path.exists() and not args.force:
        raise FileExistsError(f"Service already exists: {service_path} (pass --force to overwrite)")

    service_path.write_text(unit_text, encoding="utf-8")
    logger = configure_logging(prefix="queue")
    logger.log(LOG_STATUS, "Wrote service unit: %s", service_path)

    if shutil.which("systemctl") is None:
        logger.warning("systemctl not found; install completed but service was not enabled/started")
        return 0

    if _run_systemctl_command("daemon-reload") != 0:
        return 1
    if args.enable and args.start:
        return _run_systemctl_command("enable", "--now", f"{service_name}.service")
    if args.enable:
        return _run_systemctl_command("enable", f"{service_name}.service")
    if args.start:
        return _run_systemctl_command("start", f"{service_name}.service")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xax queue",
        description="Manage the local xax queued launcher observer and queued jobs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the queue observer loop")
    start_parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    start_parser.add_argument("--heartbeat-seconds", type=float, default=DEFAULT_HEARTBEAT_SECONDS)
    start_parser.add_argument("--once", action="store_true", help="Run at most one queued job and then exit")
    start_parser.set_defaults(func=_command_start)

    status_parser = subparsers.add_parser("status", help="Show observer status and queued jobs")
    status_parser.add_argument("--recent", type=int, default=10, help="How many recent finished jobs to show")
    status_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    status_parser.set_defaults(func=_command_status)

    list_parser = subparsers.add_parser("list", help="Alias for `status`")
    list_parser.add_argument("--recent", type=int, default=10, help="How many recent finished jobs to show")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    list_parser.set_defaults(func=_command_status)

    move_parser = subparsers.add_parser("move", help="Move a queued job to a new queue position")
    move_parser.add_argument("job_id", help="Queued job id, e.g. job-0000001")
    move_parser.add_argument("position", type=int, help="1-based queue position")
    move_parser.set_defaults(func=_command_move)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a queued job that has not started")
    cancel_parser.add_argument("job_id")
    cancel_parser.set_defaults(func=_command_cancel)

    kill_parser = subparsers.add_parser("kill", help="Kill the currently running queued job")
    kill_parser.add_argument("--signal", default="TERM", help="Signal name: TERM, INT, KILL, ...")
    kill_parser.add_argument("--grace-seconds", type=float, default=10.0)
    kill_parser.set_defaults(func=_command_kill)

    tail_parser = subparsers.add_parser("tail", help="Tail log files for a queued job")
    tail_parser.add_argument("job_id")
    tail_parser.add_argument("--kind", choices=("observer", "task"), default="observer")
    tail_parser.add_argument("--lines", type=int, default=100)
    tail_parser.add_argument("--follow", action="store_true")
    tail_parser.set_defaults(func=_command_tail)

    tensorboard_parser = subparsers.add_parser("tensorboard", help="Open TensorBoard for one or more queued jobs")
    tensorboard_parser.add_argument("job_ids", nargs="*", help="Job ids. Omit to use running + queued jobs.")
    tensorboard_parser.add_argument("--all", action="store_true", help="Use all jobs with TensorBoard logs")
    tensorboard_parser.add_argument("--port", type=int, default=DEFAULT_TENSORBOARD_PORT)
    tensorboard_parser.add_argument("--bind-all", action="store_true")
    tensorboard_parser.set_defaults(func=_command_tensorboard)

    metrics_parser = subparsers.add_parser("metrics", help="Extract scalar metrics from a queued job")
    metrics_parser.add_argument("job_id")
    metrics_parser.add_argument("--tag", default=None, help="Filter to one metric tag")
    metrics_parser.add_argument("--last", type=int, default=None, help="Keep only the last N points per tag")
    metrics_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    metrics_parser.set_defaults(func=_command_metrics)

    cleanup_parser = subparsers.add_parser("cleanup", help="Kill stale process trees and repair queue state")
    cleanup_parser.set_defaults(func=_command_cleanup)

    systemd_parser = subparsers.add_parser("systemd", help="Print or install a user systemd service unit")
    systemd_parser.add_argument("--name", default="xax-queue-observer")
    systemd_parser.add_argument("--install", action="store_true")
    systemd_parser.add_argument("--force", action="store_true")
    systemd_parser.add_argument("--enable", action="store_true")
    systemd_parser.add_argument("--start", action="store_true")
    systemd_parser.set_defaults(func=_command_systemd)

    run_parser = subparsers.add_parser("_run-job", help=argparse.SUPPRESS)
    run_parser.add_argument("--job-id", required=True)
    run_parser.set_defaults(func=lambda parsed_args: _run_queued_job(parsed_args.job_id))

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command_fn_raw = getattr(args, "func", None)
    if command_fn_raw is None or not callable(command_fn_raw):
        parser.error("No command selected")
    command_fn = cast(Callable[[argparse.Namespace], int], command_fn_raw)

    try:
        return_code = int(command_fn(args))
    except KeyboardInterrupt:
        return_code = 130
    except Exception as error:
        logger = configure_logging(prefix="queue")
        logger.error("Command failed: %s", error)
        return_code = 1
    raise SystemExit(return_code)


if __name__ == "__main__":
    # python -m xax.cli.queue
    main()
