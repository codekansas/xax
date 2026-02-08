"""Defines durable state for the local queued launcher."""

import contextlib
import copy
import fcntl
import json
import os
import signal
import socket
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Iterator, Literal, Mapping, TypeAlias

import cattrs

from xax.core.conf import get_user_global_dir

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

QUEUE_STATE_VERSION = 1
OBSERVER_HEARTBEAT_TTL_SECONDS = 20.0


def _to_json_value(value: object, *, context: str) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_json_value(item, context=f"{context}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, dict):
        payload: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid JSON key type at {context}: {type(key)!r}")
            payload[key] = _to_json_value(item, context=f"{context}.{key}")
        return payload
    raise ValueError(f"Invalid JSON value type at {context}: {type(value)!r}")


def _to_json_mapping(value: object, *, context: str) -> dict[str, JsonValue]:
    json_value = _to_json_value(value, context=context)
    if not isinstance(json_value, dict):
        raise ValueError(f"Expected JSON object at {context}, got {type(json_value)!r}")
    return json_value


@dataclass
class QueuedJob:
    job_id: str
    task_key: str
    launcher: str
    python_executable: str
    status: JobStatus
    run_dir: str
    stage_dir: str
    config_path: str
    observer_log_path: str
    enqueued_at: float
    started_at: float | None = None
    ended_at: float | None = None
    pid: int | None = None
    process_group_id: int | None = None
    child_pids: list[int] = dataclass_field(default_factory=list)
    oom_detected: bool = False
    return_code: int | None = None
    error: str | None = None


@dataclass
class QueueState:
    version: int = QUEUE_STATE_VERSION
    next_job_idx: int = 1
    queue: list[str] = dataclass_field(default_factory=list)
    running_job_id: str | None = None
    jobs: dict[str, QueuedJob] = dataclass_field(default_factory=dict)


@dataclass
class ObserverInfo:
    pid: int
    hostname: str
    started_at: float
    updated_at: float
    status: str = "idle"


@dataclass(frozen=True)
class QueuePaths:
    root_dir: Path
    state_path: Path
    lock_path: Path
    observer_path: Path


def get_queue_paths() -> QueuePaths:
    root_dir = get_user_global_dir().expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    return QueuePaths(
        root_dir=root_dir,
        state_path=root_dir / "queue_state.json",
        lock_path=root_dir / "queue_state.lock",
        observer_path=root_dir / "queue_observer.json",
    )


def _default_queue_state() -> QueueState:
    return QueueStateSerializer.default_state()


class QueueStateSerializer:
    """JSON serialization and coercion helpers for queue state payloads."""

    _converter = cattrs.Converter(forbid_extra_keys=True)

    @classmethod
    def default_state(cls) -> QueueState:
        return QueueState()

    @classmethod
    def parse_job(cls, *, job_id: str, payload: Mapping[str, JsonValue]) -> QueuedJob:
        payload_dict = {str(key): value for key, value in payload.items()}
        payload_dict.setdefault("job_id", job_id)
        try:
            return cls._converter.structure(payload_dict, QueuedJob)
        except Exception as error:
            raise ValueError(f"Invalid queued job payload for {job_id!r}: {error}") from error

    @classmethod
    def parse_state(cls, payload: Mapping[str, JsonValue]) -> QueueState:
        payload_dict = {str(key): value for key, value in payload.items()}
        try:
            return cls._converter.structure(payload_dict, QueueState)
        except Exception as error:
            raise ValueError(f"Invalid queue state payload: {error}") from error

    @classmethod
    def parse_observer_info(cls, payload: Mapping[str, JsonValue]) -> ObserverInfo:
        payload_dict = {str(key): value for key, value in payload.items()}
        try:
            return cls._converter.structure(payload_dict, ObserverInfo)
        except Exception as error:
            raise ValueError(f"Invalid observer payload: {error}") from error

    @classmethod
    def to_json_payload(cls, payload: QueueState | ObserverInfo | JsonValue) -> JsonValue:
        if isinstance(payload, (QueueState, ObserverInfo)):
            return _to_json_value(cls._converter.unstructure(payload), context=payload.__class__.__name__)
        return _to_json_value(payload, context="payload")


def _write_json_atomic(path: Path, payload: QueueState | ObserverInfo | JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload_json = QueueStateSerializer.to_json_payload(payload)
    tmp_path.write_text(json.dumps(payload_json, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_state_unlocked(paths: QueuePaths) -> QueueState:
    if not paths.state_path.exists():
        return _default_queue_state()
    payload_raw = json.loads(paths.state_path.read_text(encoding="utf-8"))
    try:
        payload = _to_json_mapping(payload_raw, context=str(paths.state_path))
        return QueueStateSerializer.parse_state(payload)
    except ValueError as error:
        raise RuntimeError(f"Invalid queue state at {paths.state_path}: {error}") from error


@contextlib.contextmanager
def queue_lock(paths: QueuePaths | None = None) -> Iterator[None]:
    paths = get_queue_paths() if paths is None else paths
    paths.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def edit_queue_state(paths: QueuePaths | None = None) -> Iterator[QueueState]:
    paths = get_queue_paths() if paths is None else paths
    with queue_lock(paths):
        state = _load_state_unlocked(paths)
        yield state
        _write_json_atomic(paths.state_path, state)


def read_queue_state(paths: QueuePaths | None = None) -> QueueState:
    paths = get_queue_paths() if paths is None else paths
    with queue_lock(paths):
        return copy.deepcopy(_load_state_unlocked(paths))


def _canonical_launcher(launcher: str) -> str:
    mapping = {
        "s": "single",
        "single": "single",
        "m": "multi",
        "multi": "multi",
        "mc": "multi_cpu",
        "multi_cpu": "multi_cpu",
        "d": "dataset",
        "dataset": "dataset",
    }
    if launcher not in mapping:
        raise ValueError(f"Unsupported launcher for queued job: {launcher}")
    return mapping[launcher]


def _insert_queue_job(queue: list[str], job_id: str, position_idx: int | None) -> None:
    if position_idx is None:
        queue.append(job_id)
        return
    bounded_position_idx = max(0, min(position_idx, len(queue)))
    queue.insert(bounded_position_idx, job_id)


def enqueue_job(
    *,
    task_key: str,
    launcher: str,
    python_executable: str,
    run_dir: Path,
    stage_dir: Path,
    config_path: Path,
    observer_log_path: Path,
    position_idx: int | None = None,
    paths: QueuePaths | None = None,
) -> str:
    paths = get_queue_paths() if paths is None else paths
    launcher_name = _canonical_launcher(launcher)
    python_executable_str = python_executable.strip()
    if not python_executable_str:
        raise ValueError("python_executable must be a non-empty path")
    now = time.time()
    with edit_queue_state(paths) as state:
        job_id = f"job-{state.next_job_idx:07d}"
        state.next_job_idx += 1
        job = QueuedJob(
            job_id=job_id,
            task_key=task_key,
            launcher=launcher_name,
            python_executable=python_executable_str,
            status="queued",
            run_dir=str(run_dir),
            stage_dir=str(stage_dir),
            config_path=str(config_path),
            observer_log_path=str(observer_log_path),
            enqueued_at=now,
        )
        state.jobs[job_id] = job
        _insert_queue_job(state.queue, job_id, position_idx=position_idx)
    return job_id


def get_job(job_id: str, paths: QueuePaths | None = None) -> QueuedJob | None:
    state = read_queue_state(paths)
    job = state.jobs.get(job_id)
    return None if job is None else copy.deepcopy(job)


def list_jobs(paths: QueuePaths | None = None) -> list[QueuedJob]:
    state = read_queue_state(paths)
    queue_order: list[QueuedJob] = [state.jobs[job_id] for job_id in state.queue if job_id in state.jobs]
    queue_ids = {job.job_id for job in queue_order}
    other_jobs = [job for job_id, job in state.jobs.items() if job_id not in queue_ids]
    other_jobs.sort(key=lambda job: job.enqueued_at, reverse=True)
    return queue_order + other_jobs


def move_queued_job(job_id: str, position_idx: int, paths: QueuePaths | None = None) -> None:
    with edit_queue_state(paths) as state:
        if job_id not in state.queue:
            raise ValueError(f"Job {job_id} is not currently queued")
        state.queue.remove(job_id)
        _insert_queue_job(state.queue, job_id, position_idx=position_idx)


def cancel_queued_job(job_id: str, paths: QueuePaths | None = None) -> QueuedJob:
    with edit_queue_state(paths) as state:
        if (job := state.jobs.get(job_id)) is None:
            raise ValueError(f"Unknown job {job_id}")
        if job.status == "running":
            raise ValueError(f"Job {job_id} is running; use `xax queue kill`")
        if job_id in state.queue:
            state.queue.remove(job_id)
        job.status = "cancelled"
        job.ended_at = time.time()
        job.error = "Cancelled before execution"
        return copy.deepcopy(job)


def claim_next_job(paths: QueuePaths | None = None) -> QueuedJob | None:
    with edit_queue_state(paths) as state:
        if state.running_job_id is not None:
            return None
        while state.queue:
            next_job_id = state.queue.pop(0)
            if (job := state.jobs.get(next_job_id)) is None:
                continue
            if job.status != "queued":
                continue
            job.status = "running"
            job.started_at = time.time()
            job.ended_at = None
            job.return_code = None
            job.pid = None
            job.process_group_id = None
            job.child_pids = []
            job.oom_detected = False
            job.error = None
            state.running_job_id = next_job_id
            return copy.deepcopy(job)
        return None


def set_running_job_pid(
    job_id: str,
    pid: int,
    process_group_id: int | None = None,
    paths: QueuePaths | None = None,
) -> None:
    with edit_queue_state(paths) as state:
        if state.running_job_id != job_id:
            raise ValueError(f"Job {job_id} is not the active running job")
        if (job := state.jobs.get(job_id)) is None:
            raise ValueError(f"Unknown job {job_id}")
        job.pid = pid
        job.process_group_id = process_group_id


def set_running_job_children(job_id: str, child_pids: list[int], paths: QueuePaths | None = None) -> None:
    with edit_queue_state(paths) as state:
        if state.running_job_id != job_id:
            raise ValueError(f"Job {job_id} is not the active running job")
        if (job := state.jobs.get(job_id)) is None:
            raise ValueError(f"Unknown job {job_id}")
        job.child_pids = [int(pid) for pid in sorted(set(child_pids))]


def clear_job_process_tracking(job_id: str, paths: QueuePaths | None = None) -> None:
    with edit_queue_state(paths) as state:
        if (job := state.jobs.get(job_id)) is None:
            raise ValueError(f"Unknown job {job_id}")
        job.pid = None
        job.process_group_id = None
        job.child_pids = []


def finish_running_job(
    job_id: str,
    return_code: int,
    error: str | None = None,
    oom_detected: bool = False,
    paths: QueuePaths | None = None,
) -> QueuedJob:
    with edit_queue_state(paths) as state:
        if (job := state.jobs.get(job_id)) is None:
            raise ValueError(f"Unknown job {job_id}")
        job.return_code = return_code
        job.ended_at = time.time()
        job.pid = None
        job.process_group_id = None
        job.child_pids = []
        job.oom_detected = oom_detected
        if return_code == 0 and error is None:
            job.status = "completed"
            job.error = None
        else:
            job.status = "failed"
            job.error = error if error is not None else f"Process exited with return code {return_code}"
        if state.running_job_id == job_id:
            state.running_job_id = None
        return copy.deepcopy(job)


def get_running_job(paths: QueuePaths | None = None) -> QueuedJob | None:
    state = read_queue_state(paths)
    running_job_id = state.running_job_id
    if running_job_id is None:
        return None
    job = state.jobs.get(running_job_id)
    return None if job is None else copy.deepcopy(job)


def recover_orphaned_running_job(
    reason: str = "Recovered orphaned running job",
    paths: QueuePaths | None = None,
) -> QueuedJob | None:
    with edit_queue_state(paths) as state:
        running_job_id = state.running_job_id
        if running_job_id is None:
            return None
        if (job := state.jobs.get(running_job_id)) is None:
            state.running_job_id = None
            return None
        pid = job.pid
        process_group_id = job.process_group_id
        if pid is not None and is_pid_alive(pid):
            return None
        if process_group_id is not None and is_process_group_alive(process_group_id):
            return None
        job.status = "failed"
        job.error = reason
        job.ended_at = time.time()
        job.pid = None
        job.process_group_id = None
        job.child_pids = []
        if job.return_code is None:
            job.return_code = -1
        state.running_job_id = None
        return copy.deepcopy(job)


def _read_observer_info_unlocked(paths: QueuePaths) -> ObserverInfo | None:
    if not paths.observer_path.exists():
        return None
    payload_raw = json.loads(paths.observer_path.read_text(encoding="utf-8"))
    try:
        payload = _to_json_mapping(payload_raw, context=str(paths.observer_path))
        return QueueStateSerializer.parse_observer_info(payload)
    except ValueError:
        return None


def read_observer_info(paths: QueuePaths | None = None) -> ObserverInfo | None:
    paths = get_queue_paths() if paths is None else paths
    with queue_lock(paths):
        info = _read_observer_info_unlocked(paths)
    return None if info is None else copy.deepcopy(info)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def is_process_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def is_observer_info_active(
    info: ObserverInfo | None,
    ttl_seconds: float = OBSERVER_HEARTBEAT_TTL_SECONDS,
) -> bool:
    if info is None:
        return False
    if time.time() - info.updated_at > ttl_seconds:
        return False
    return is_pid_alive(info.pid)


def is_observer_active(
    paths: QueuePaths | None = None,
    ttl_seconds: float = OBSERVER_HEARTBEAT_TTL_SECONDS,
) -> bool:
    return is_observer_info_active(read_observer_info(paths), ttl_seconds=ttl_seconds)


def register_observer(
    pid: int,
    status: str = "idle",
    paths: QueuePaths | None = None,
    ttl_seconds: float = OBSERVER_HEARTBEAT_TTL_SECONDS,
) -> ObserverInfo:
    paths = get_queue_paths() if paths is None else paths
    with queue_lock(paths):
        prev_info = _read_observer_info_unlocked(paths)
        if (
            is_observer_info_active(prev_info, ttl_seconds=ttl_seconds)
            and prev_info is not None
            and prev_info.pid != pid
        ):
            raise RuntimeError(f"Queue observer is already running (pid={prev_info.pid}, host={prev_info.hostname})")
        now = time.time()
        info = ObserverInfo(
            pid=pid,
            hostname=socket.gethostname(),
            started_at=prev_info.started_at if prev_info is not None and prev_info.pid == pid else now,
            updated_at=now,
            status=status,
        )
        _write_json_atomic(paths.observer_path, info)
        return info


def touch_observer(
    pid: int,
    status: str = "idle",
    paths: QueuePaths | None = None,
) -> ObserverInfo:
    paths = get_queue_paths() if paths is None else paths
    with queue_lock(paths):
        prev_info = _read_observer_info_unlocked(paths)
        if prev_info is not None and prev_info.pid != pid and is_observer_info_active(prev_info):
            raise RuntimeError(
                f"Queue observer heartbeat owned by different process (pid={prev_info.pid}); refusing update"
            )
        now = time.time()
        info = ObserverInfo(
            pid=pid,
            hostname=socket.gethostname(),
            started_at=prev_info.started_at if prev_info is not None and prev_info.pid == pid else now,
            updated_at=now,
            status=status,
        )
        _write_json_atomic(paths.observer_path, info)
        return info


def clear_observer(pid: int | None = None, paths: QueuePaths | None = None) -> None:
    paths = get_queue_paths() if paths is None else paths
    with queue_lock(paths):
        if not paths.observer_path.exists():
            return
        if pid is None:
            paths.observer_path.unlink()
            return
        info = _read_observer_info_unlocked(paths)
        if info is None or info.pid == pid:
            paths.observer_path.unlink()


def kill_running_job_process(
    signum: signal.Signals = signal.SIGTERM,
    paths: QueuePaths | None = None,
) -> QueuedJob | None:
    running_job = get_running_job(paths)
    if running_job is None:
        return None
    if (process_group_id := running_job.process_group_id) is not None:
        os.killpg(process_group_id, int(signum))
    elif running_job.pid is not None:
        os.kill(running_job.pid, int(signum))
    else:
        return None
    return running_job
