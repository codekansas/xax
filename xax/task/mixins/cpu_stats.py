"""A trainer mixin for logging CPU statistics.

This logs memory and CPU utilization in a background process, sending it to
the logging process every now and then. This is useful for detecting memory
leaks in your dataloader, among other issues.
"""

import logging
import os
import time
from ctypes import Structure, c_double, c_uint16, c_uint64
from dataclasses import dataclass
from multiprocessing.context import BaseContext, Process
from multiprocessing.managers import SyncManager, ValueProxy
from multiprocessing.synchronize import Event
from typing import Generic, TypeVar

import jax
import psutil

from xax.task.base import BaseConfig
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.process import ProcessConfig, ProcessMixin
from xax.utils.structured_config import field

logger: logging.Logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class CPUStatsOptions:
    ping_interval: int = field(1, help="How often to check stats (in seconds)")
    only_log_once: bool = field(False, help="If set, only log read stats one time")


@jax.tree_util.register_dataclass
@dataclass
class CPUStatsConfig(ProcessConfig, LoggerConfig, BaseConfig):
    cpu_stats: CPUStatsOptions = field(CPUStatsOptions(), help="CPU stats configuration")


Config = TypeVar("Config", bound=CPUStatsConfig)


class CPUStats(Structure):
    _fields_ = [
        ("cpu_percent", c_double),
        ("mem_percent", c_double),
        ("mem_rss", c_uint64),
        ("mem_vms", c_uint64),
        ("mem_shared", c_uint64),
        ("mem_rss_total", c_uint64),
        ("mem_vms_total", c_uint64),
        ("child_cpu_percent", c_double),
        ("child_mem_percent", c_double),
        ("num_child_procs", c_uint16),
    ]


@dataclass(kw_only=True)
class CPUStatsInfo:
    cpu_percent: float
    mem_percent: float
    mem_rss: int
    mem_vms: int
    mem_shared: int
    mem_rss_total: int
    mem_vms_total: int
    child_cpu_percent: float
    child_mem_percent: float
    num_child_procs: int

    @classmethod
    def from_stats(cls, stats: CPUStats) -> "CPUStatsInfo":
        return cls(
            cpu_percent=stats.cpu_percent,
            mem_percent=stats.mem_percent,
            mem_rss=stats.mem_rss,
            mem_vms=stats.mem_vms,
            mem_shared=stats.mem_shared,
            mem_rss_total=stats.mem_rss_total,
            mem_vms_total=stats.mem_vms_total,
            child_cpu_percent=stats.child_cpu_percent,
            child_mem_percent=stats.child_mem_percent,
            num_child_procs=stats.num_child_procs,
        )


def worker(
    ping_interval: float,
    stats: ValueProxy[CPUStats],
    monitor_event: Event,
    start_event: Event,
    pid: int,
) -> None:
    start_event.set()

    proc, cur_pid = psutil.Process(pid), os.getpid()
    logger.debug("Starting CPU stats monitor for PID %d with PID %d", pid, cur_pid)

    def get_children() -> dict[int, psutil.Process]:
        return {p.pid: p for p in proc.children(recursive=True) if p.pid != cur_pid}

    child_procs = get_children()

    try:
        while True:
            # Updates child processes, preserving the previous child process
            # object. Otherwise the CPU percentage will be zero.
            new_procs = get_children()
            child_procs = {**new_procs, **child_procs}
            child_procs = {pid: child_procs[pid] for pid in new_procs.keys()}

            # Gets process memory info.
            mem_info = proc.memory_info()
            mem_rss_total = sum(p.memory_info().rss for p in child_procs.values()) + mem_info.rss
            mem_vms_total = sum(p.memory_info().vms for p in child_procs.values()) + mem_info.vms

            # Gets child CPU and memory percentages.
            child_cpu_percent_total = sum(p.cpu_percent() for p in child_procs.values()) if child_procs else 0.0
            child_mem_percent_total = sum(p.memory_percent() for p in child_procs.values()) if child_procs else 0.0

            # Sets the CPU stats.
            stats.set(
                CPUStats(
                    cpu_percent=proc.cpu_percent(),
                    mem_percent=proc.memory_percent(),
                    mem_rss=int(mem_info.rss),
                    mem_vms=int(mem_info.vms),
                    mem_shared=int(getattr(mem_info, "shared", 0)),
                    mem_rss_total=int(mem_rss_total),
                    mem_vms_total=int(mem_vms_total),
                    child_cpu_percent=child_cpu_percent_total / len(child_procs),
                    child_mem_percent=child_mem_percent_total / len(child_procs),
                    num_child_procs=len(child_procs),
                ),
            )

            monitor_event.set()
            time.sleep(ping_interval)

    except BaseException:
        logger.error("Closing CPU stats monitor")


class CPUStatsMonitor:
    def __init__(
        self,
        ping_interval: float,
        context: BaseContext,
        manager: SyncManager,
    ) -> None:
        self._ping_interval = ping_interval
        self._manager = manager
        self._context = context

        self._monitor_event = self._manager.Event()
        self._start_event = self._manager.Event()
        self._cpu_stats_smem = self._manager.Value(
            CPUStats,
            CPUStats(
                cpu_percent=0.0,
                mem_percent=0.0,
                mem_rss=0,
                mem_vms=0,
                mem_shared=0,
                mem_rss_total=0,
                mem_vms_total=0,
                child_cpu_percent=0.0,
                child_mem_percent=0.0,
                num_child_procs=0,
            ),
        )
        self._cpu_stats: CPUStatsInfo | None = None
        self._proc: Process | None = None

    def get_if_set(self) -> CPUStatsInfo | None:
        if self._monitor_event.is_set():
            self._monitor_event.clear()
            return CPUStatsInfo.from_stats(self._cpu_stats_smem.get())
        return None

    def get(self) -> CPUStatsInfo | None:
        if (stats := self.get_if_set()) is not None:
            self._cpu_stats = stats
        return self._cpu_stats

    def start(self, wait: bool = False) -> None:
        if self._proc is not None:
            raise RuntimeError("CPU stats monitor already started")
        if self._monitor_event.is_set():
            self._monitor_event.clear()
        if self._start_event.is_set():
            self._start_event.clear()
        self._cpu_stats = None
        self._proc = self._context.Process(  # type: ignore[attr-defined]
            target=worker,
            args=(self._ping_interval, self._cpu_stats_smem, self._monitor_event, self._start_event, os.getpid()),
            daemon=True,
            name="xax-cpu-stats",
        )
        self._proc.start()
        if wait:
            self._start_event.wait()

    def stop(self) -> None:
        if self._proc is None:
            raise RuntimeError("CPU stats monitor not started")
        if self._proc.is_alive():
            self._proc.terminate()
            logger.debug("Terminated CPU stats monitor; joining...")
            self._proc.join()
        self._proc = None
        self._cpu_stats = None


class CPUStatsMixin(ProcessMixin[Config], LoggerMixin[Config], Generic[Config]):
    """Defines a task mixin for getting CPU statistics."""

    _cpu_stats_monitor: CPUStatsMonitor | None

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if (ctx := self.multiprocessing_context) is not None and (mgr := self.multiprocessing_manager) is not None:
            self._cpu_stats_monitor = CPUStatsMonitor(self.config.cpu_stats.ping_interval, ctx, mgr)
        else:
            self._cpu_stats_monitor = None

    def on_training_start(self) -> None:
        super().on_training_start()

        if (monitor := self._cpu_stats_monitor) is not None:
            monitor.start()

    def on_training_end(self) -> None:
        super().on_training_end()

        if (monitor := self._cpu_stats_monitor) is not None:
            monitor.stop()

    def on_step_start(self) -> None:
        super().on_step_start()

        if (monitor := self._cpu_stats_monitor) is None:
            return

        stats = monitor.get_if_set() if self.config.cpu_stats.only_log_once else monitor.get()

        if stats is not None:
            self.logger.log_scalar("child_procs", stats.num_child_procs, namespace="🔧 cpu", secondary=True)
            self.logger.log_scalar("percent", stats.cpu_percent, namespace="🔧 cpu", secondary=True)
            self.logger.log_scalar("child_percent", stats.child_cpu_percent, namespace="🔧 cpu", secondary=True)
            self.logger.log_scalar("percent", stats.mem_percent, namespace="🔧 mem", secondary=True)
            self.logger.log_scalar("shared", stats.mem_shared, namespace="🔧 mem", secondary=True)
            self.logger.log_scalar("child_percent", stats.child_mem_percent, namespace="🔧 mem", secondary=True)
            self.logger.log_scalar("rss/cur", stats.mem_rss, namespace="🔧 mem", secondary=True)
            self.logger.log_scalar("rss/total", stats.mem_rss_total, namespace="🔧 mem", secondary=True)
            self.logger.log_scalar("vms/cur", stats.mem_vms, namespace="🔧 mem", secondary=True)
            self.logger.log_scalar("vms/total", stats.mem_vms_total, namespace="🔧 mem", secondary=True)
