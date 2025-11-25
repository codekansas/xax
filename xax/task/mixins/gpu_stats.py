"""A task mixin for logging GPU statistics.

This logs GPU memory and utilization in a background process using
``nvidia-smi``, if a GPU is available in the system.
"""

import logging
import os
import shutil
import subprocess
from ctypes import Structure, c_double, c_uint32
from dataclasses import dataclass
from multiprocessing.context import BaseContext, Process
from multiprocessing.managers import SyncManager, ValueProxy
from multiprocessing.synchronize import Event
from typing import Generic, Iterable, TypeVar

import jax

from xax.core.conf import field
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.process import ProcessConfig, ProcessMixin
from xax.utils.devices import get_num_gpus, parse_number

logger: logging.Logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class GPUStatsOptions:
    ping_interval: int = field(10, help="How often to check stats (in seconds)")
    only_log_once: bool = field(False, help="If set, only log read stats one time")


@jax.tree_util.register_dataclass
@dataclass
class GPUStatsConfig(ProcessConfig, LoggerConfig):
    gpu_stats: GPUStatsOptions = field(GPUStatsOptions(), help="GPU stats configuration")


Config = TypeVar("Config", bound=GPUStatsConfig)


class GPUStats(Structure):
    _fields_ = [
        ("index", c_uint32),
        ("memory_used", c_double),
        ("temperature", c_double),
        ("utilization", c_double),
    ]


@dataclass(frozen=True)
class GPUStatsInfo:
    index: int
    memory_used: float
    temperature: float
    utilization: float

    @classmethod
    def from_stats(cls, stats: GPUStats) -> "GPUStatsInfo":
        return cls(
            index=stats.index,
            memory_used=stats.memory_used,
            temperature=stats.temperature,
            utilization=stats.utilization,
        )


def parse_gpu_stats(row: str) -> GPUStats:
    cols = row.split(",")
    index = int(cols[0].strip())
    memory_total, memory_used, temperature, utilization = (parse_number(col) for col in cols[1:])

    return GPUStats(
        index=index,
        memory_used=100 * memory_used / memory_total,
        temperature=temperature,
        utilization=utilization,
    )


def gen_gpu_stats(loop_secs: int = 5) -> Iterable[GPUStats]:
    fields = ",".join(["index", "memory.total", "memory.used", "temperature.gpu", "utilization.gpu"])
    command = f"nvidia-smi --query-gpu={fields} --format=csv,noheader --loop={loop_secs}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_device_ids = None if visible_devices is None else {int(i.strip()) for i in visible_devices.split(",")}

    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            rows = iter(stdout.readline, "")
            for row in rows:
                try:
                    stats = parse_gpu_stats(row)
                except ValueError:
                    continue
                if visible_device_ids is None or stats.index in visible_device_ids:
                    yield stats

    except BaseException:
        logger.error("Closing GPU stats monitor")


def worker(
    ping_interval: int,
    smems: list[ValueProxy[GPUStats]],
    main_event: Event,
    events: list[Event],
    start_event: Event,
) -> None:
    start_event.set()

    logger.debug("Starting GPU stats monitor with PID %d", os.getpid())

    for gpu_stat in gen_gpu_stats(ping_interval):
        if gpu_stat.index >= len(smems):
            logger.warning("GPU index %d is out of range", gpu_stat.index)
            continue
        smems[gpu_stat.index].set(gpu_stat)
        events[gpu_stat.index].set()
        main_event.set()


class GPUStatsMonitor:
    def __init__(
        self,
        ping_interval: float,
        context: BaseContext,
        manager: SyncManager,
    ) -> None:
        self._ping_interval = ping_interval
        self._context = context
        self._manager = manager

        num_gpus = get_num_gpus()
        self._main_event = manager.Event()
        self._events = [manager.Event() for _ in range(num_gpus)]
        self._start_event = manager.Event()

        self._smems = [
            manager.Value(
                GPUStats,
                GPUStats(
                    index=i,
                    memory_used=0.0,
                    temperature=0.0,
                    utilization=0.0,
                ),
            )
            for i in range(num_gpus)
        ]
        self._gpu_stats: dict[int, GPUStatsInfo] = {}
        self._proc: Process | None = None

    def get_if_set(self) -> dict[int, GPUStatsInfo]:
        gpu_stats: dict[int, GPUStatsInfo] = {}
        if self._main_event.is_set():
            self._main_event.clear()
            for i, event in enumerate(self._events):
                if event.is_set():
                    event.clear()
                    gpu_stats[i] = GPUStatsInfo.from_stats(self._smems[i].get())
        return gpu_stats

    def get(self) -> dict[int, GPUStatsInfo]:
        self._gpu_stats.update(self.get_if_set())
        return self._gpu_stats

    def start(self, wait: bool = False) -> None:
        if self._proc is not None:
            raise RuntimeError("GPUStatsMonitor already started")
        if self._main_event.is_set():
            self._main_event.clear()
        for event in self._events:
            if event.is_set():
                event.clear()
        if self._start_event.is_set():
            self._start_event.clear()
        self._gpu_stats.clear()
        self._proc = self._context.Process(  # type: ignore[attr-defined]
            target=worker,
            args=(self._ping_interval, self._smems, self._main_event, self._events, self._start_event),
            daemon=True,
            name="xax-gpu-stats",
        )
        self._proc.start()
        if wait:
            self._start_event.wait()

    def stop(self) -> None:
        if self._proc is None:
            raise RuntimeError("GPUStatsMonitor not started")
        if self._proc.is_alive():
            self._proc.terminate()
            logger.debug("Terminated GPU stats monitor; joining...")
            self._proc.join()
        self._proc = None


class GPUStatsMixin(ProcessMixin[Config], LoggerMixin[Config], Generic[Config]):
    """Defines a task mixin for getting GPU statistics."""

    _gpu_stats_monitor: GPUStatsMonitor | None

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if (
            shutil.which("nvidia-smi") is not None
            and (ctx := self.multiprocessing_context) is not None
            and (mgr := self.multiprocessing_manager) is not None
        ):
            self._gpu_stats_monitor = GPUStatsMonitor(config.gpu_stats.ping_interval, ctx, mgr)
        else:
            self._gpu_stats_monitor = None

    def on_training_start(self) -> None:
        super().on_training_start()

        if (monitor := self._gpu_stats_monitor) is not None:
            monitor.start()

    def on_training_end(self) -> None:
        super().on_training_end()

        if (monitor := self._gpu_stats_monitor) is not None:
            monitor.stop()

    def on_step_start(self) -> None:
        super().on_step_start()

        if (monitor := self._gpu_stats_monitor) is None:
            return
        stats = monitor.get_if_set() if self.config.gpu_stats.only_log_once else monitor.get()

        for gpu_stat in stats.values():
            if gpu_stat is None:
                continue
            self.logger.log_scalar(f"mem/{gpu_stat.index}", gpu_stat.memory_used, namespace="🔧 gpu", secondary=True)
            self.logger.log_scalar(f"temp/{gpu_stat.index}", gpu_stat.temperature, namespace="🔧 gpu", secondary=True)
            self.logger.log_scalar(f"util/{gpu_stat.index}", gpu_stat.utilization, namespace="🔧 gpu", secondary=True)
