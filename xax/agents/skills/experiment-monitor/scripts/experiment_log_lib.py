"""Shared helpers for experiment-monitor skill scripts."""

import csv
import datetime as dt
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal

from xax.core.conf import get_experiments_dir, get_user_global_dir

FIELDNAMES: list[str] = [
    "experiment_id",
    "parent_experiment_id",
    "queue_job_id",
    "status",
    "task_key",
    "command",
    "config_path",
    "run_dir",
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

ExperimentStatus = Literal[
    "planned",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "superseded",
]


def utc_now_iso() -> str:
    now = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def epoch_seconds_to_iso(epoch_seconds: float | int | None) -> str:
    if epoch_seconds is None:
        return ""
    timestamp = dt.datetime.fromtimestamp(float(epoch_seconds), tz=dt.timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def default_experiment_name() -> str:
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"experiment-{timestamp}"


def _skill_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _templates_dir() -> Path:
    return _skill_root() / "templates"


def get_experiments_root_dir(experiments_dir: Path | None = None) -> Path:
    if experiments_dir is not None:
        root_dir = experiments_dir.expanduser().resolve()
    else:
        configured_experiments_dir = get_experiments_dir()
        if configured_experiments_dir is not None:
            root_dir = configured_experiments_dir.expanduser().resolve()
        else:
            root_dir = (get_user_global_dir() / "experiments").expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def resolve_experiment_dir(experiment_name: str, experiments_dir: Path | None = None) -> Path:
    if not experiment_name.strip():
        raise ValueError("Experiment name must be non-empty")
    return get_experiments_root_dir(experiments_dir) / experiment_name.strip()


def resolve_log_path(
    log_path: Path | None = None,
    *,
    experiment_name: str = "default",
    experiments_dir: Path | None = None,
) -> Path:
    if log_path is not None:
        return log_path.expanduser().resolve()
    experiment_dir = resolve_experiment_dir(experiment_name, experiments_dir=experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir / "experiment_log.csv"


def resolve_report_path(
    output_path: Path | None = None,
    *,
    experiment_name: str = "default",
    experiments_dir: Path | None = None,
) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    experiment_dir = resolve_experiment_dir(experiment_name, experiments_dir=experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir / "experiment_report.md"


def ensure_experiment_templates(experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    template_root = _templates_dir()
    src_log_path = template_root / "experiment_log.csv"
    dst_log_path = experiment_dir / "experiment_log.csv"
    if src_log_path.exists() and not dst_log_path.exists():
        shutil.copy2(src_log_path, dst_log_path)
    src_report_path = template_root / "experiment_report.md"
    dst_report_path = experiment_dir / "experiment_report.md"
    if src_report_path.exists() and not dst_report_path.exists():
        shutil.copy2(src_report_path, dst_report_path)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
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


def upsert_row(rows: list[dict[str, str]], *, experiment_id: str, updates: dict[str, str]) -> bool:
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


def parse_metric_pairs(metric_pairs: list[str]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for metric_pair in metric_pairs:
        if "=" not in metric_pair:
            raise ValueError(f"Invalid --metric value: {metric_pair!r}; expected NAME=VALUE")
        metric_name, metric_value = metric_pair.split("=", maxsplit=1)
        metric_name = metric_name.strip()
        if not metric_name:
            raise ValueError(f"Invalid --metric value: {metric_pair!r}; metric name is empty")
        metrics[metric_name] = metric_value.strip()
    return metrics


def load_metrics_json(metrics_json: str | None, metrics_json_file: Path | None) -> dict[str, object]:
    if metrics_json is not None and metrics_json_file is not None:
        raise ValueError("Pass at most one of --metrics-json and --metrics-json-file")
    if metrics_json is None and metrics_json_file is None:
        return {}

    if metrics_json is not None:
        payload_raw = metrics_json
    else:
        if metrics_json_file is None:
            raise ValueError("Metrics file is required when --metrics-json is not set")
        payload_raw = metrics_json_file.read_text(encoding="utf-8")
    payload = json.loads(payload_raw)
    if not isinstance(payload, dict):
        raise ValueError("Metrics JSON must be an object")
    return {str(key): value for key, value in payload.items()}


def as_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def as_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def get_experiment_id(job_payload: dict[str, object]) -> str | None:
    job_id_raw = job_payload.get("job_id")
    if isinstance(job_id_raw, str) and job_id_raw:
        return job_id_raw
    return None


def escape_cell(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ")


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: row.get("updated_at", "") or row.get("created_at", ""), reverse=True)


def get_best_completed_row(rows: list[dict[str, str]], objective_mode: str) -> dict[str, str] | None:
    completed_rows = [row for row in rows if row.get("status") == "completed" and row.get("objective_value", "")]
    scored_rows: list[tuple[float, dict[str, str]]] = []
    for row in completed_rows:
        objective_value_raw = row.get("objective_value", "")
        try:
            objective_value = float(objective_value_raw)
        except ValueError:
            continue
        scored_rows.append((objective_value, row))

    if not scored_rows:
        return None
    if objective_mode == "min":
        return min(scored_rows, key=lambda pair: pair[0])[1]
    return max(scored_rows, key=lambda pair: pair[0])[1]


def render_report_markdown(
    *,
    title: str,
    rows: list[dict[str, str]],
    max_rows: int,
    objective_mode: str,
) -> str:
    status_counts = Counter(row.get("status", "unknown") for row in rows)
    best_row = get_best_completed_row(rows, objective_mode=objective_mode)

    lines: list[str] = [
        f"# {title}",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Summary",
        "",
        f"- Total experiments: {len(rows)}",
        f"- Objective mode: {objective_mode}",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")

    lines += ["", "## Best Result", ""]
    if best_row is None:
        lines.append("- No completed experiments with numeric objective values.")
    else:
        lines.append(
            f"- {best_row.get('experiment_id', '')}: {best_row.get('objective_metric', '')}="
            f"{best_row.get('objective_value', '')}"
        )

    lines += [
        "",
        "## Recent Experiments",
        "",
        "| Experiment | Status | Metric | Value | Updated | Next Action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows[:max_rows]:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_cell(row.get("experiment_id", "")),
                    escape_cell(row.get("status", "")),
                    escape_cell(row.get("objective_metric", "")),
                    escape_cell(row.get("objective_value", "")),
                    escape_cell(row.get("updated_at", "")),
                    escape_cell(row.get("next_action", "")),
                ]
            )
            + " |"
        )

    return "\n".join(lines).strip() + "\n"
