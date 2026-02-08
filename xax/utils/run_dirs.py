"""Shared utilities for run directory naming and allocation."""

from pathlib import Path

RUN_DIR_INDEX_WIDTH = 3


def format_run_dir_name(run_idx: int) -> str:
    if run_idx < 0:
        raise ValueError("run_idx must be >= 0")
    return f"run_{run_idx:0{RUN_DIR_INDEX_WIDTH}d}"


def next_available_run_dir(runs_dir: str | Path) -> Path:
    runs_root = Path(runs_dir).expanduser().resolve()
    run_idx = 0
    while (candidate := runs_root / format_run_dir_name(run_idx)).exists():
        run_idx += 1
    return candidate


def resolve_configured_run_dir(run_dir: str | Path | None, *, field_name: str = "run_dir") -> Path | None:
    if run_dir is None:
        return None
    if isinstance(run_dir, Path):
        return run_dir.expanduser().resolve()

    run_dir_str = run_dir.strip()
    if not run_dir_str:
        raise ValueError(f"{field_name} must be a non-empty path when provided.")
    return Path(run_dir_str).expanduser().resolve()
