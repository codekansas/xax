"""Abstract interfaces for tasks launched through the local queue."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self


class QueuedArtifactsConfig(ABC):
    run_dir: str | None
    runs_dir: str | None


class QueuedArtifactsTask(ABC):
    config: QueuedArtifactsConfig
    task_name: str
    task_key: str

    @property
    @abstractmethod
    def run_dir(self) -> Path:
        """Returns the active run directory."""

    @abstractmethod
    def set_run_dir(self, run_dir: str | Path) -> Self:
        """Sets and normalizes the run directory."""

    @abstractmethod
    def get_queue_runs_dir(self) -> Path:
        """Returns the canonical runs directory used for queued jobs."""
