"""Defines a Weights & Biases logger backend."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import jax
import numpy as np

from xax.core.conf import field
from xax.nn.parallel import is_master
from xax.task.logger import LogError, LogErrorSummary, LoggerImpl, LogLine, LogPing, LogStatus
from xax.utils.jax import as_float

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def sanitize_metric_name(name: str) -> str:
    """Remove 4-byte unicode characters from metric names.

    W&B has issues with 4-byte unicode characters in metric names,
    so we need to filter them out.

    Args:
        name: The metric name to sanitize.

    Returns:
        The sanitized metric name.
    """
    # Filter out characters that don't fit in UCS-2 (Basic Multilingual Plane)
    # These are characters with code points > 0xFFFF (4-byte UTF-8)
    return "".join(char for char in name if ord(char) <= 0xFFFF)


WandbConfigResumePolicy = Literal["allow", "never", "must", "auto"]
WandbConfigResume = WandbConfigResumePolicy | bool
WandbConfigMode = Literal["online", "offline", "disabled", "shared"] | None
WandbConfigReinit = Literal["return_previous", "finish_previous"]


@jax.tree_util.register_dataclass
@dataclass
class WandbConfig:
    project: str | None = field(None, help="The name of the W&B project to log to.")
    entity: str | None = field(None, help="The W&B entity (team or user) to log to.")
    name: str | None = field(None, help="The name of this run.")
    config: dict[str, Any] | None = field(None, help="Configuration dictionary to log.")
    tags: list[str] | None = field(None, help="List of tags for this run.")
    notes: str | None = field(None, help="Notes about this run.")
    log_interval_seconds: float = field(10.0, help="The interval between successive log lines.")
    reinit: WandbConfigReinit = field(
        "return_previous",
        help="Whether to allow multiple wandb.init() calls in the same process.",
    )
    resume: WandbConfigResume = field(False, help="Whether to resume a previous run. Can be a run ID string.")
    mode: WandbConfigMode = field(None, help="Mode for wandb (online, offline, or disabled).")


class WandbLogger(LoggerImpl):
    def __init__(self, run_directory: str | Path, config: WandbConfig) -> None:
        """Defines a logger which writes to Weights & Biases.

        Args:
            run_directory: The root run directory. If provided, wandb will save
                files to a subdirectory here.
            config: Configuration for the W&B logger.
        """
        try:
            import wandb as _wandb  # noqa: F401,PLC0415
        except ImportError as e:
            raise RuntimeError(
                "WandbLogger requires the 'wandb' package. Install it with: pip install xax[wandb]"
            ) from e

        self._wandb = _wandb

        super().__init__(config.log_interval_seconds)

        self.project = config.project
        self.entity = config.entity
        self.name = config.name
        self.config = config.config
        self.tags = config.tags
        self.notes = config.notes
        self.reinit = config.reinit
        self.resume: WandbConfigResume = config.resume
        self.mode: WandbConfigMode = config.mode

        # Set wandb directory if run_directory is provided
        self.wandb_dir = Path(run_directory).expanduser().resolve() / "wandb"
        self.wandb_dir.mkdir(parents=True, exist_ok=True)

        self._started = False

        # Store pending files to log
        self.files: dict[str, str] = {}

        self.start()

    def start(self) -> None:
        """Initialize the W&B run."""
        if self._started or not is_master():
            return

        # Set wandb environment variables if needed
        if self.wandb_dir is not None:
            os.environ["WANDB_DIR"] = str(self.wandb_dir)

        # Initialize wandb run
        self.run = self._wandb.init(  # pyright
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
            reinit=self.reinit,
            resume=self.resume,
            mode=self.mode,
        )

        self._started = True
        logger.info("W&B run initialized: %s", self.run.url if self.run else "No URL available")

    def stop(self) -> None:
        """Finish the W&B run."""
        if not self._started or not is_master():
            return

        if self.run is not None:
            self.run.finish()
        self._started = False

    def log_file(self, name: str, contents: str) -> None:
        """Store a file to be logged with the next write call.

        Args:
            name: The name of the file.
            contents: The contents of the file.
        """
        if not is_master():
            return
        self.files[name] = contents

    def write(self, line: LogLine) -> None:
        """Writes the current log line to W&B.

        Args:
            line: The line to write.
        """
        if not is_master() or not self._started:
            return

        # Get step information
        global_step = line.state.num_steps.item()

        # Dictionary to collect all metrics for this step
        metrics: dict[str, Any] = {}

        # Log scalars
        for namespace, scalars in line.scalars.items():
            for scalar_key, scalar_value in scalars.items():
                key = sanitize_metric_name(f"{namespace}/{scalar_key}")
                metrics[key] = as_float(scalar_value.value)

        # Log distributions as custom metrics (mean and std)
        for namespace, distributions in line.distributions.items():
            for distribution_key, distribution_value in distributions.items():
                base_key = sanitize_metric_name(f"{namespace}/{distribution_key}")
                metrics[f"{base_key}/mean"] = float(distribution_value.mean)
                metrics[f"{base_key}/std"] = float(distribution_value.std)

        # Log histograms
        for namespace, histograms in line.histograms.items():
            for histogram_key, histogram_value in histograms.items():
                key = sanitize_metric_name(f"{namespace}/{histogram_key}")
                # Create histogram data for wandb
                # W&B expects a list of values or a numpy array
                # We need to reconstruct the data from the histogram bins
                values = []
                for i, count in enumerate(histogram_value.bucket_counts):
                    if count > 0:
                        # Use the midpoint of each bucket
                        if i == 0:
                            val = histogram_value.bucket_limits[0]
                        else:
                            val = (histogram_value.bucket_limits[i - 1] + histogram_value.bucket_limits[i]) / 2
                        values.extend([val] * count)

                if values:
                    # wandb.Histogram accepts lists directly
                    metrics[key] = self._wandb.Histogram(values)

        # Log strings as HTML
        for namespace, strings in line.strings.items():
            for string_key, string_value in strings.items():
                key = sanitize_metric_name(f"{namespace}/{string_key}")
                # For strings, we can log them as HTML or just as text in a table
                metrics[key] = self._wandb.Html(f"<pre>{string_value.value}</pre>")

        # Log images
        for namespace, images in line.images.items():
            for image_key, image_value in images.items():
                key = sanitize_metric_name(f"{namespace}/{image_key}")
                # Convert PIL image to wandb.Image
                metrics[key] = self._wandb.Image(image_value.image)

        # Log videos
        for namespace, videos in line.videos.items():
            for video_key, video_value in videos.items():
                key = sanitize_metric_name(f"{namespace}/{video_key}")
                # wandb.Video expects shape (time, channels, height, width)
                # Our format is (T, H, W, C) so we need to transpose to (T, C, H, W)
                frames = video_value.frames.transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
                metrics[key] = self._wandb.Video(frames, fps=video_value.fps, format="mp4")

        # Log audio
        for namespace, audios in line.audios.items():
            for audio_key, audio_value in audios.items():
                key = sanitize_metric_name(f"{namespace}/{audio_key}")
                # wandb.Audio expects numpy array of shape (samples,) or (samples, channels)
                # Our format is (T,) or (C, T), so transpose multi-channel to (T, C)
                audio_data = audio_value.audio
                if audio_data.ndim == 2:
                    audio_data = audio_data.T  # (C, T) -> (T, C)
                metrics[key] = self._wandb.Audio(audio_data, sample_rate=audio_value.sample_rate)

        # Log meshes (3D objects)
        for namespace, meshes in line.meshes.items():
            for mesh_key, mesh_value in meshes.items():
                key = sanitize_metric_name(f"{namespace}/{mesh_key}")
                # W&B Object3D expects vertices and faces in specific format
                # vertices: (batch_size, num_vertices, 3) or (num_vertices, 3)
                # faces: (batch_size, num_faces, 3) or (num_faces, 3)
                vertices = mesh_value.vertices

                # Handle batch dimension - take first batch if present
                if vertices.ndim == 3:
                    vertices = vertices[0]

                obj3d_data = {
                    "type": "lidar/beta",
                    "vertices": vertices.tolist(),
                }

                if mesh_value.faces is not None:
                    faces = mesh_value.faces
                    if faces.ndim == 3:
                        faces = faces[0]
                    obj3d_data["faces"] = faces.tolist()

                if mesh_value.colors is not None:
                    colors = mesh_value.colors
                    if colors.ndim == 3:
                        colors = colors[0]
                    # Convert colors to 0-1 range if they're in 0-255 range
                    # The colors are already numpy arrays from LogMesh, converted by as_numpy
                    if colors.dtype == np.uint8:
                        colors = colors.astype(np.float32) / 255.0
                    obj3d_data["colors"] = colors.tolist()

                metrics[key] = self._wandb.Object3D(obj3d_data)

        # Log any pending files as artifacts or text
        for name, contents in self.files.items():
            # Log as HTML text
            key = sanitize_metric_name(name)
            key = f"{self.run.name}_{key}"
            is_training_code = "code" in name
            artifact = self._wandb.Artifact(
                name=key if not is_training_code else "training_code",
                type="code" if is_training_code else "unspecified",
            )
            with artifact.new_file(name) as f:
                f.write(contents)
            artifact.save()
        self.files.clear()

        # Log all metrics at once
        if metrics and self.run:
            self.run.log(metrics, step=global_step)

    def write_error_summary(self, error_summary: LogErrorSummary) -> None:
        pass

    def write_error(self, error: LogError) -> None:
        pass

    def write_status(self, status: LogStatus) -> None:
        pass

    def write_ping(self, ping: LogPing) -> None:
        pass
