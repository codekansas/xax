"""Defines a logger which logs JSON lines to a file."""

import json
from pathlib import Path
from typing import Any, Literal, Mapping, TextIO

from jaxtyping import Array

from xax.nn.parallel import get_rank, get_world_size
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


def get_json_value(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, Array):
        value = value.item()
    return value


class JsonLogger(LoggerImpl):
    def __init__(
        self,
        run_directory: str | Path,
        subdirectory: str = "json",
        flush_immediately: bool = False,
        open_mode: Literal["w", "a"] = "w",
        line_sep: str = "\n",
        remove_unicode_from_namespaces: bool = True,
        log_interval_seconds: float = 10.0,
    ) -> None:
        """Defines a simpler logger which logs to stdout.

        Args:
            run_directory: The directory to log to.
            subdirectory: The subdirectory to log to.
            flush_immediately: Whether to flush the file after every write.
            open_mode: The file open mode.
            line_sep: The line separator to use.
            remove_unicode_from_namespaces: Whether to remove unicode from
                namespaces. This is the typical behavior for namespaces that
                use ASCII art for visibility in other logs, but in the JSON
                log file should be ignored.
            log_interval_seconds: The interval between successive log lines.
        """
        super().__init__(log_interval_seconds)

        self.run_directory = (Path(run_directory).expanduser() / subdirectory).resolve()
        self.flush_immediately = flush_immediately
        self.open_mode = open_mode
        self.line_sep = line_sep
        self.remove_unicode_from_namespaces = remove_unicode_from_namespaces

        # Open file pointers to the log and error files.
        rank, world_size = get_rank(), get_world_size()
        num_digits = len(str(world_size))
        rank_str = f"{rank:0{num_digits}d}" if world_size > 1 else ""
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._log_fp = (self.run_directory / f"log{rank_str}.jsonl").open(open_mode, encoding="utf-8")
        self._err_fp = (self.run_directory / f"error{rank_str}.jsonl").open(open_mode, encoding="utf-8")

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def __del__(self) -> None:
        self._log_fp.close()
        self._err_fp.close()

    @property
    def fp(self) -> TextIO:
        return self._log_fp

    @property
    def err_fp(self) -> TextIO:
        return self._err_fp

    def get_json(self, line: LogLine) -> str:
        data: dict = {"state": line.state.to_dict()}

        def add_logs(log: Mapping[str, Mapping[str, LogScalar | LogString]], data: dict) -> None:
            for namespace, values in log.items():
                if self.remove_unicode_from_namespaces:
                    namespace = namespace.encode("ascii", errors="ignore").decode("ascii").strip()
                if namespace not in data:
                    data[namespace] = {}
                for k, v in values.items():
                    data[namespace][k] = get_json_value(v.value)

        add_logs(line.scalars, data)
        add_logs(line.strings, data)
        return json.dumps(data)

    def write(self, line: LogLine) -> None:
        self.fp.write(self.get_json(line))
        self.fp.write(self.line_sep)
        if self.flush_immediately:
            self.fp.flush()

    def write_error_summary(self, error_summary: LogErrorSummary) -> None:
        pass

    def log_file(self, name: str, contents: str) -> None:
        pass

    def write_error(self, error: LogError) -> None:
        self.err_fp.write(error.message)
        if error.location is not None:
            self.err_fp.write(f" ({error.location})")
        self.err_fp.write(self.line_sep)
        if self.flush_immediately:
            self.err_fp.flush()

    def write_ping(self, ping: LogPing) -> None:
        self.fp.write(
            json.dumps(
                {
                    "ping": {
                        "message": ping.message,
                        "filename": ping.filename,
                        "lineno": ping.lineno,
                    },
                },
            ),
        )
        self.fp.write(self.line_sep)
        if self.flush_immediately:
            self.fp.flush()

    def write_status(self, status: LogStatus) -> None:
        self.fp.write(
            json.dumps(
                {
                    "status": {
                        "message": status.message,
                        "filename": status.filename,
                        "lineno": status.lineno,
                    },
                },
            ),
        )
        self.fp.write(self.line_sep)
        if self.flush_immediately:
            self.fp.flush()
