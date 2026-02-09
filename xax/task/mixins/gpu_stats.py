"""A task mixin for logging GPU statistics.

This logs GPU memory and utilization in a background process using
``nvidia-smi``, if a GPU is available in the system. Includes detailed
metrics for memory bandwidth utilization and SM (compute) utilization.
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

from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.process import ProcessConfig, ProcessMixin
from xax.utils.devices import get_num_gpus
from xax.utils.structured_config import field

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
        ("sm_util", c_double),
        ("mem_bw_util", c_double),
        ("enc_util", c_double),
        ("dec_util", c_double),
        ("jpg_util", c_double),
        ("ofa_util", c_double),
    ]


@dataclass(frozen=True)
class GPUStatsInfo:
    index: int
    sm_util: float
    mem_bw_util: float
    enc_util: float
    dec_util: float
    jpg_util: float
    ofa_util: float

    @classmethod
    def from_stats(cls, stats: GPUStats) -> "GPUStatsInfo":
        return cls(
            index=stats.index,
            sm_util=stats.sm_util,
            mem_bw_util=stats.mem_bw_util,
            enc_util=stats.enc_util,
            dec_util=stats.dec_util,
            jpg_util=stats.jpg_util,
            ofa_util=stats.ofa_util,
        )


def parse_dmon_row(row: str, col_indices: dict[str, int]) -> GPUStats | None:
    """Parse a row from nvidia-smi dmon output.

    Args:
        row: A line from dmon output.
        col_indices: Mapping of column names to indices.

    Returns:
        GPUStats or None if parsing failed.
    """
    parts = row.split()
    if not parts or parts[0].startswith("#"):
        return None

    try:
        index = int(parts[col_indices["gpu"]])
        sm_util = float(parts[col_indices["sm"]])
        mem_bw = float(parts[col_indices["mem"]])
        enc_util = float(parts[col_indices["enc"]])
        dec_util = float(parts[col_indices["dec"]])
        jpg_util = float(parts[col_indices["jpg"]])
        ofa_util = float(parts[col_indices["ofa"]])

        return GPUStats(
            index=index,
            sm_util=sm_util,
            mem_bw_util=mem_bw,
            enc_util=enc_util,
            dec_util=dec_util,
            jpg_util=jpg_util,
            ofa_util=ofa_util,
        )
    except (ValueError, IndexError, KeyError):
        return None


def gen_gpu_stats(loop_secs: int = 5) -> Iterable[GPUStats]:
    """Generate GPU stats using nvidia-smi dmon for detailed utilization metrics.

    Uses nvidia-smi dmon to get SM utilization (compute throughput) and memory
    bandwidth utilization, which are better indicators of GPU efficiency than
    the basic utilization metric.

    Args:
        loop_secs: Interval between stat collections in seconds.

    Yields:
        GPUStats objects with utilization metrics.
    """
    # Use dmon with utilization metrics (-s u) for SM and memory bandwidth
    command = f"nvidia-smi dmon -s u -d {loop_secs}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_device_ids = None if visible_devices is None else {int(i.strip()) for i in visible_devices.split(",") if i}

    col_indices: dict[str, int] = {}

    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            rows = iter(stdout.readline, "")

            for row in rows:
                row = row.strip()
                if not row:
                    continue

                # Parse header to get column indices
                # Header format: "# gpu     sm    mem    enc    dec    jpg    ofa"
                if row.startswith("# gpu"):
                    parts = row[1:].split()  # Remove leading '#'
                    col_indices = {name: i for i, name in enumerate(parts)}
                    continue

                # Skip the units row
                if row.startswith("# Idx"):
                    continue

                # Parse data row
                stats = parse_dmon_row(row, col_indices)
                if stats is None:
                    continue

                if visible_device_ids is not None and stats.index not in visible_device_ids:
                    continue

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
                    sm_util=0.0,
                    mem_bw_util=0.0,
                    enc_util=0.0,
                    dec_util=0.0,
                    jpg_util=0.0,
                    ofa_util=0.0,
                ),
            )
            for i in range(num_gpus)
        ]
        self._gpu_stats: dict[int, GPUStatsInfo] = {}
        self._proc: Process | None = None

    def get_if_set(self) -> dict[int, GPUStatsInfo]:
        gpu_stats: dict[int, GPUStatsInfo] = {}
        # The SyncManager backing these proxy objects can exit before the main
        # training loop fully unwinds (e.g. at shutdown / after a queued-job
        # teardown). In that case, accessing the proxies raises BrokenPipeError.
        # GPU stats are best-effort; never fail training due to monitor issues.
        try:
            is_set = self._main_event.is_set()
        except (BrokenPipeError, EOFError, OSError):
            return gpu_stats

        if is_set:
            try:
                self._main_event.clear()
            except (BrokenPipeError, EOFError, OSError):
                return gpu_stats

            for idx, event in enumerate(self._events):
                try:
                    if event.is_set():
                        event.clear()
                        gpu_stats[idx] = GPUStatsInfo.from_stats(self._smems[idx].get())
                except (BrokenPipeError, EOFError, OSError):
                    # Manager died mid-iteration; return what we have.
                    break
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
            self.logger.log_scalar(f"sm/{gpu_stat.index}", gpu_stat.sm_util, namespace="🔧 gpu", secondary=True)
            self.logger.log_scalar(f"bw/{gpu_stat.index}", gpu_stat.mem_bw_util, namespace="🔧 gpu", secondary=True)
            self.logger.log_scalar(f"enc/{gpu_stat.index}", gpu_stat.enc_util, namespace="🔧 gpu", secondary=True)
            # Never used during training in practice.
            # self.logger.log_scalar(f"dec/{gpu_stat.index}", gpu_stat.dec_util, namespace="🔧 gpu", secondary=True)
            self.logger.log_scalar(f"jpg/{gpu_stat.index}", gpu_stat.jpg_util, namespace="🔧 gpu", secondary=True)
            # Never used during training in practice.
            # self.logger.log_scalar(f"ofa/{gpu_stat.index}", gpu_stat.ofa_util, namespace="🔧 gpu", secondary=True)
