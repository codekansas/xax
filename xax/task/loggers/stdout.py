"""Defines a logger that logs to stdout."""

import datetime
import logging
import sys
from collections import deque
from typing import Any, Deque, Mapping, TextIO

from jaxtyping import Array

from xax.task.logger import (
    LogError,
    LogErrorSummary,
    LoggerImpl,
    LogLine,
    LogPing,
    LogScalar,
    LogStatus,
    LogString,
)
from xax.utils.text import Color, colored, format_timedelta


def format_number(value: int | float, precision: int) -> str:
    if isinstance(value, int):
        return f"{value:,}"  # Add commas to the number
    return f"{value:.{precision}g}"


def as_str(value: Any, precision: int) -> str:  # noqa: ANN401
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, Array):
        value = value.item()
    if isinstance(value, (int, float)):
        return format_number(value, precision)
    raise TypeError(f"Unexpected log type: {type(value)}")


class StdoutLogger(LoggerImpl):
    def __init__(
        self,
        write_fp: TextIO = sys.stdout,
        precision: int = 4,
        log_timers: bool = True,
        log_perf: bool = False,
        log_optim: bool = False,
        log_fp: bool = False,
        log_interval_seconds: float = 1.0,
        remove_temporary_after: datetime.timedelta = datetime.timedelta(seconds=10),
    ) -> None:
        """Defines a logger which shows a pop-up using Curses.

        Args:
            write_fp: The file to write logs to.
            precision: The integer precision to use when logging scalars.
            log_timers: Whether to log timers.
            log_perf: Whether to log performance metrics.
            log_optim: Whether to log optimizer parameters.
            log_fp: Whether to log floating point parameters.
            log_interval_seconds: The interval between successive log lines.
            remove_temporary_after: The time after which temporary toasts
                are removed.
        """
        super().__init__(log_interval_seconds)

        self.write_fp = write_fp
        self.log_timers = log_timers
        self.log_perf = log_perf
        self.log_fp = log_fp
        self.log_optim = log_optim
        self.precision = precision
        self.remove_temporary_after = remove_temporary_after
        self.logger = logging.getLogger("stdout")

        self.statuses: Deque[tuple[str, datetime.datetime]] = deque()
        self.pings: Deque[tuple[str, datetime.datetime]] = deque()
        self.errors: Deque[tuple[str, datetime.datetime]] = deque()
        self.error_summary: tuple[str, datetime.datetime] | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.write_queues()

    def log_file(self, name: str, contents: str) -> None:
        pass

    def write_separator(self) -> None:
        self.write_fp.write("\033[2J\033[H")

    def write_state_window(self, line: LogLine) -> None:
        state_info: dict[str, str] = {
            "Steps": format_number(int(line.state.num_steps.item()), 0),
            "Samples": format_number(int(line.state.num_samples.item()), 0),
            "Elapsed Time": format_timedelta(datetime.timedelta(seconds=line.state.elapsed_time_s.item()), short=True),
        }

        for k, v in state_info.items():
            self.write_fp.write(f" ↪ {k}: {colored(v, 'cyan')}\n")

    def write_log_window(self, line: LogLine) -> None:
        namespace_to_lines: dict[str, dict[str, str]] = {}

        def add_logs(
            log: Mapping[str, Mapping[str, LogScalar | LogString]],
            namespace_to_lines: dict[str, dict[str, str]],
        ) -> None:
            for namespace, values in log.items():
                for k, v in values.items():
                    if v.secondary:
                        continue
                    if namespace not in namespace_to_lines:
                        namespace_to_lines[namespace] = {}
                    v_str = as_str(v.value, self.precision)
                    namespace_to_lines[namespace][k] = v_str

        add_logs(line.scalars, namespace_to_lines)
        add_logs(line.strings, namespace_to_lines)
        if not namespace_to_lines:
            return

        for namespace, lines in sorted(namespace_to_lines.items()):
            self.write_fp.write(f"\n{colored(namespace, 'cyan', bold=True)}\n")
            for k, v in lines.items():
                self.write_fp.write(f" ↪ {k}: {v}\n")

    def write_queue(self, title: str, q: Deque[tuple[str, datetime.datetime]], remove: bool, color: Color) -> None:
        if not q:
            return

        self.write_fp.write(f"\n{colored(title, 'grey', bold=True)}\n")
        self.write_fp.write("\n".join(f" ✦ {colored(msg, color)}" for msg, _ in reversed(q)))
        self.write_fp.write("\n")

        if remove:
            now = datetime.datetime.now()
            while q and now - q[0][1] > self.remove_temporary_after:
                q.popleft()

    def write_queues(self) -> None:
        self.write_queue("Status", self.statuses, False, "green")
        self.write_queue("Pings", self.pings, True, "cyan")
        self.write_queue("Errors", self.errors, False, "red")

    def write_error_summary_to_screen(self) -> None:
        if self.error_summary is not None:
            summary, timestamp = self.error_summary
            timestamp_string = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            self.write_fp.write(f"\n{colored('Exception summary', 'grey', bold=True)}")
            self.write_fp.write(f" {colored(timestamp_string, 'grey')}")
            self.write_fp.write(f"\n{summary}")

    def write(self, line: LogLine) -> None:
        self.write_separator()
        self.write_state_window(line)
        self.write_log_window(line)
        self.write_queues()
        self.write_error_summary_to_screen()
        sys.stdout.flush()

    def write_error_summary(self, error_summary: LogErrorSummary) -> None:
        self.error_summary = error_summary.message, datetime.datetime.now()

    def write_error(self, error: LogError) -> None:
        self.errors.append((error.message_with_location, datetime.datetime.now()))

    def write_status(self, status: LogStatus) -> None:
        self.statuses.append((status.message, datetime.datetime.now()))

    def write_ping(self, ping: LogPing) -> None:
        self.pings.append((ping.message, datetime.datetime.now()))
