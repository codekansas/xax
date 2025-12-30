"""Defines a Tensorboard logger backend."""

import atexit
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import TypeVar

from xax.nn.parallel import is_master
from xax.task.logger import LogError, LogErrorSummary, LoggerImpl, LogLine, LogPing, LogStatus
from xax.utils.jax import as_float
from xax.utils.logging import LOG_STATUS, port_is_busy
from xax.utils.tensorboard import TensorboardWriter, TensorboardWriters

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_TENSORBOARD_PORT = 9249


class TensorboardLogger(LoggerImpl):
    def __init__(
        self,
        run_directory: str | Path,
        subdirectory: str = "tensorboard",
        flush_seconds: float = 10.0,
        wait_seconds: float = 0.0,
        start_in_subprocess: bool = True,
        use_localhost: bool | None = None,
        log_interval_seconds: float = 10.0,
    ) -> None:
        """Defines a logger which writes to Tensorboard.

        Args:
            run_directory: The root run directory.
            subdirectory: The subdirectory of the run directory to write
                Tensorboard logs to.
            flush_seconds: How often to flush logs.
            wait_seconds: Time to wait before starting Tensorboard process.
            start_in_subprocess: Start TensorBoard subprocess.
            use_localhost: Use localhost for TensorBoard address.
            log_interval_seconds: The interval between successive log lines.
        """
        super().__init__(log_interval_seconds)

        if use_localhost is None:
            use_localhost = os.environ.get("USE_TENSORBOARD_LOCALHOST", "0") == "1"

        self.log_directory = Path(run_directory).expanduser().resolve() / subdirectory
        self.wait_seconds = wait_seconds
        self.start_in_subprocess = start_in_subprocess
        self.use_localhost = use_localhost

        self.proc: subprocess.Popen | None = None

        self.files: dict[str, str] = {}
        self.writers = TensorboardWriters(log_directory=self.log_directory, flush_seconds=flush_seconds)
        self._started = False

    def _start(self) -> None:
        if self._started:
            return

        if is_master():
            threading.Thread(target=self.worker_thread, daemon=True).start()

        self._started = True

    def worker_thread(self) -> None:
        if os.environ.get("DISABLE_TENSORBOARD", "0") == "1":
            return

        time.sleep(self.wait_seconds)

        port = int(os.environ.get("TENSORBOARD_PORT", DEFAULT_TENSORBOARD_PORT))

        while port_is_busy(port):
            logger.warning("Port %s is busy, waiting...", port)
            time.sleep(10)

        def make_localhost(s: str) -> str:
            if self.use_localhost:
                s = re.sub(rf"://(.+?):{port}", f"://localhost:{port}", s)
            return s

        def parse_url(s: str) -> str:
            m = re.search(r" (http\S+?) ", s)
            if m is None:
                return s
            return f"Tensorboard: {m.group(1)}"

        command: list[str] = [
            "python",
            "-m",
            "tensorboard.main",
            "serve",
            "--logdir",
            str(self.log_directory),
            "--bind_all",
            "--port",
            str(port),
            "--reload_interval",
            "15",
        ]

        if not self.start_in_subprocess:
            logger.warning("Tensorboard subprocess disabled because start_in_subprocess=False")

        else:
            self.proc = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Gets the output line that shows the running address.
            assert self.proc is not None and self.proc.stdout is not None
            lines = []
            for line in self.proc.stdout:
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("TensorBoard"):
                    line_str = parse_url(make_localhost(line_str))
                    logging.log(LOG_STATUS, line_str)
                    break
                lines.append(line_str)
            else:
                line_str = "".join(lines)
                raise RuntimeError(f"Tensorboard failed to start:\n{line_str}")

            atexit.register(self.cleanup)

    def cleanup(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

    def __del__(self) -> None:
        self.cleanup()

    def get_writer(self, heavy: bool) -> TensorboardWriter:
        self._start()
        return self.writers.writer(heavy)

    def log_file(self, name: str, contents: str) -> None:
        if not is_master():
            return
        self.files[name] = f"```\n{contents}\n```"

    def write(self, line: LogLine) -> None:
        if not is_master():
            return

        writer = self.get_writer(line.heavy)

        global_step = line.state.num_steps.item()
        walltime = line.state.start_time_s.item() + line.state.elapsed_time_s.item()

        for namespace, scalars in line.scalars.items():
            for scalar_key, scalar_value in scalars.items():
                writer.add_scalar(
                    f"{namespace}/{scalar_key}",
                    as_float(scalar_value.value),
                    global_step=global_step,
                    walltime=walltime,
                )

        for namespace, distributions in line.distributions.items():
            for distribution_key, distribution_value in distributions.items():
                writer.add_gaussian_distribution(
                    f"{namespace}/{distribution_key}",
                    mean=float(distribution_value.mean),
                    std=float(distribution_value.std),
                    global_step=global_step,
                    walltime=walltime,
                )

        for namespace, histograms in line.histograms.items():
            for histogram_key, histogram_value in histograms.items():
                writer.add_histogram_raw(
                    f"{namespace}/{histogram_key}",
                    min=float(histogram_value.min),
                    max=float(histogram_value.max),
                    num=int(histogram_value.num),
                    sum=float(histogram_value.sum),
                    sum_squares=float(histogram_value.sum_squares),
                    bucket_limits=[float(x) for x in histogram_value.bucket_limits],
                    bucket_counts=[int(x) for x in histogram_value.bucket_counts],
                    global_step=global_step,
                    walltime=walltime,
                )

        for namespace, strings in line.strings.items():
            for string_key, string_value in strings.items():
                writer.add_text(
                    f"{namespace}/{string_key}",
                    string_value.value,
                    global_step=global_step,
                    walltime=walltime,
                )

        for namespace, images in line.images.items():
            for image_key, image_value in images.items():
                writer.add_image(
                    f"{namespace}/{image_key}",
                    image_value.image,
                    global_step=global_step,
                    walltime=walltime,
                )

        for namespace, videos in line.videos.items():
            for video_key, video_value in videos.items():
                writer.add_video(
                    f"{namespace}/{video_key}",
                    video_value.frames,
                    fps=video_value.fps,
                    global_step=global_step,
                    walltime=walltime,
                )

        for namespace, meshes in line.meshes.items():
            for mesh_key, mesh_value in meshes.items():
                writer.add_mesh(
                    f"{namespace}/{mesh_key}",
                    vertices=mesh_value.vertices,
                    faces=mesh_value.faces,
                    colors=mesh_value.colors,
                    config_dict=mesh_value.config_dict,
                    global_step=global_step,
                    walltime=walltime,
                )

        for name, contents in self.files.items():
            writer.add_text(name, contents)
        self.files.clear()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def write_error(self, error: LogError) -> None:
        pass

    def write_error_summary(self, error_summary: LogErrorSummary) -> None:
        pass

    def write_ping(self, ping: LogPing) -> None:
        pass

    def write_status(self, status: LogStatus) -> None:
        pass
