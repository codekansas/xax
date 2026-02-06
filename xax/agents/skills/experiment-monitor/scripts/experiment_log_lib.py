"""Shared helpers for experiment monitor CSV management."""

import csv
import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Mapping

try:
    from xax.core.conf import (
        get_experiments_dir as _get_experiments_dir_from_conf,
        get_user_global_dir as _get_user_global_dir_from_conf,
    )
except Exception:  # pragma: no cover
    _get_experiments_dir_from_conf = None
    _get_user_global_dir_from_conf = None

FIELDNAMES: list[str] = [
    "experiment_id",
    "parent_experiment_id",
    "queue_job_id",
    "status",
    "task_key",
    "command",
    "config_path",
    "exp_dir",
    "objective_metric",
    "objective_mode",
    "objective_value",
    "metrics_json",
    "hypothesis",
    "change_summary",
    "result_summary",
    "next_action",
    "started_at",
    "ended_at",
    "created_at",
    "updated_at",
    "owner",
    "notes",
]

STATUS_CHOICES: tuple[str, ...] = (
    "planned",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "superseded",
)


def _script_root() -> Path:
    return Path(__file__).resolve().parent


def templates_dir() -> Path:
    return _script_root().parent / "templates"


def get_experiments_root_dir(experiments_dir: Path | None = None) -> Path:
    if experiments_dir is not None:
        root_dir = experiments_dir.expanduser().resolve()
    elif (experiments_dir_env := os.environ.get("EXPERIMENTS_DIR")) is not None:
        root_dir = Path(experiments_dir_env).expanduser().resolve()
    else:
        root_dir = Path("~/.xax/experiments").expanduser().resolve()
        if _get_experiments_dir_from_conf is not None:
            configured_experiments_dir = _get_experiments_dir_from_conf()
            if configured_experiments_dir is not None:
                root_dir = configured_experiments_dir.expanduser().resolve()
            elif _get_user_global_dir_from_conf is not None:
                root_dir = (_get_user_global_dir_from_conf() / "experiments").expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def resolve_experiment_dir(experiment_name: str, experiments_dir: Path | None = None) -> Path:
    if not experiment_name.strip():
        raise ValueError("Experiment name must be non-empty")
    experiment_dir = get_experiments_root_dir(experiments_dir) / experiment_name.strip()
    return experiment_dir


def resolve_log_path(
    log_path: Path | None = None,
    *,
    experiment_name: str | None = None,
    experiments_dir: Path | None = None,
) -> Path:
    if log_path is not None:
        return log_path.expanduser().resolve()
    experiment_name = (
        experiment_name
        if experiment_name is not None
        else os.environ.get("XAX_EXPERIMENT_NAME", "").strip() or "default"
    )
    experiment_dir = resolve_experiment_dir(experiment_name, experiments_dir=experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir / "experiment_log.csv"


def resolve_report_path(
    output_path: Path | None = None,
    *,
    experiment_name: str | None = None,
    experiments_dir: Path | None = None,
) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    experiment_name = (
        experiment_name
        if experiment_name is not None
        else os.environ.get("XAX_EXPERIMENT_NAME", "").strip() or "default"
    )
    experiment_dir = resolve_experiment_dir(experiment_name, experiments_dir=experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir / "experiment_report.md"


def ensure_experiment_templates(experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    template_root = templates_dir()
    src_log_path = template_root / "experiment_log.csv"
    dst_log_path = experiment_dir / "experiment_log.csv"
    if src_log_path.exists() and not dst_log_path.exists():
        shutil.copy2(src_log_path, dst_log_path)
    src_report_path = template_root / "experiment_report.md"
    dst_report_path = experiment_dir / "experiment_report.md"
    if src_report_path.exists() and not dst_report_path.exists():
        shutil.copy2(src_report_path, dst_report_path)


def utc_now_iso() -> str:
    now = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def epoch_seconds_to_iso(epoch_seconds: float | int | None) -> str:
    if epoch_seconds is None:
        return ""
    timestamp = dt.datetime.fromtimestamp(float(epoch_seconds), tz=dt.timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def normalize_row(row: Mapping[str, str]) -> dict[str, str]:
    normalized = {fieldname: row.get(fieldname, "") for fieldname in FIELDNAMES}
    if not normalized["experiment_id"]:
        raise ValueError("Each row must include a non-empty experiment_id")
    return normalized


def read_rows(log_path: Path) -> list[dict[str, str]]:
    if not log_path.exists():
        return []
    with open(log_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: list[dict[str, str]] = []
        for row in reader:
            if row.get("experiment_id", "").strip():
                rows.append(normalize_row({fieldname: row.get(fieldname, "") for fieldname in FIELDNAMES}))
    return rows


def write_rows(log_path: Path, rows: list[dict[str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row))


def upsert_row(
    rows: list[dict[str, str]],
    *,
    experiment_id: str,
    updates: Mapping[str, str],
) -> bool:
    now_iso = utc_now_iso()
    target_idx: int | None = None
    for row_idx, row in enumerate(rows):
        if row["experiment_id"] == experiment_id:
            target_idx = row_idx
            break

    if target_idx is None:
        new_row = {fieldname: "" for fieldname in FIELDNAMES}
        new_row["experiment_id"] = experiment_id
        new_row["created_at"] = now_iso
        rows.append(new_row)
        target_idx = len(rows) - 1
        created = True
    else:
        created = False

    target_row = rows[target_idx]
    for key, value in updates.items():
        if key in target_row and value != "":
            target_row[key] = value
    target_row["updated_at"] = now_iso
    if not target_row["created_at"]:
        target_row["created_at"] = now_iso
    rows[target_idx] = normalize_row(target_row)
    return created
