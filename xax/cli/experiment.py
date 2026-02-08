"""CLI for managing experiment-monitor sessions and logs."""

import csv
import datetime as dt
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from xax.core.conf import get_experiments_dir, get_user_global_dir
from xax.utils.cli_args import parse_args_as, render_help_text
from xax.utils.cli_output import get_cli_output

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

STATUS_CHOICES: tuple[str, ...] = (
    "planned",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "superseded",
)


@dataclass(frozen=True)
class SessionArgs:
    name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    new: bool = field(default=False, metadata={"help": "Require creating a new session"})
    resume: bool = field(default=False, metadata={"help": "Require resuming an existing session"})


@dataclass(frozen=True)
class UpsertArgs:
    experiment_id: str = field(metadata={"help": "Stable experiment id"})
    log_path: Path | None = field(default=None, metadata={"help": "Override experiment log CSV path"})
    experiment_name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    parent_experiment_id: str | None = field(default=None, metadata={"help": "Parent experiment id"})
    queue_job_id: str | None = field(default=None, metadata={"help": "Queue job id"})
    status: str | None = field(default=None, metadata={"help": "Experiment status"})
    task_key: str | None = field(default=None, metadata={"help": "Task key"})
    command: str | None = field(default=None, metadata={"help": "Command string"})
    config_path: str | None = field(default=None, metadata={"help": "Config snapshot path"})
    run_dir: str | None = field(default=None, metadata={"help": "Run directory path"})
    objective_metric: str | None = field(default=None, metadata={"help": "Objective metric"})
    objective_mode: str | None = field(default=None, metadata={"help": "Objective mode (max|min)"})
    objective_value: float | None = field(default=None, metadata={"help": "Objective value"})
    hypothesis: str | None = field(default=None, metadata={"help": "Hypothesis"})
    change_summary: str | None = field(default=None, metadata={"help": "What changed"})
    result_summary: str | None = field(default=None, metadata={"help": "Observed result"})
    next_action: str | None = field(default=None, metadata={"help": "Recommended next step"})
    started_at: str | None = field(default=None, metadata={"help": "ISO start timestamp"})
    ended_at: str | None = field(default=None, metadata={"help": "ISO end timestamp"})
    owner: str | None = field(default=None, metadata={"help": "Owner name"})
    notes: str | None = field(default=None, metadata={"help": "Free-form notes"})
    metrics_json: str | None = field(default=None, metadata={"help": "JSON string object of extra metrics"})
    metrics_json_file: Path | None = field(default=None, metadata={"help": "Path to JSON object with extra metrics"})
    metric: list[str] = field(default_factory=list, metadata={"help": "Repeated NAME=VALUE metric entries"})


@dataclass(frozen=True)
class SyncQueueStatusArgs:
    status_json: Path = field(metadata={"help": "Path to queue status JSON payload"})
    log_path: Path | None = field(default=None, metadata={"help": "Override experiment log CSV path"})
    experiment_name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    owner: str | None = field(default=None, metadata={"help": "Optional owner to apply to imported rows"})


@dataclass(frozen=True)
class ReportArgs:
    log_path: Path | None = field(default=None, metadata={"help": "Override experiment log CSV path"})
    output_path: Path | None = field(default=None, metadata={"help": "Override report output path"})
    experiment_name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    title: str = field(default="Experiment Monitor Report", metadata={"help": "Report title"})
    max_rows: int = field(default=25, metadata={"help": "Number of recent rows to include"})
    objective_mode: str = field(default="auto", metadata={"help": "Objective mode: auto|max|min"})


@dataclass(frozen=True)
class _CommandSpec:
    description: str
    args_type: type


COMMAND_SPECS: dict[str, _CommandSpec] = {
    "session": _CommandSpec(
        description="Create or resume an experiment-monitor session directory.",
        args_type=SessionArgs,
    ),
    "upsert": _CommandSpec(
        description="Insert or update one experiment row in experiment_log.csv.",
        args_type=UpsertArgs,
    ),
    "sync-queue-status": _CommandSpec(
        description="Merge queue status JSON jobs into experiment_log.csv.",
        args_type=SyncQueueStatusArgs,
    ),
    "report": _CommandSpec(
        description="Render experiment_report.md from experiment_log.csv.",
        args_type=ReportArgs,
    ),
}


def _skill_root() -> Path:
    return Path(__file__).resolve().parent.parent / "agents" / "skills" / "experiment-monitor"


def _templates_dir() -> Path:
    return _skill_root() / "templates"


def _default_experiment_name() -> str:
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"experiment-{timestamp}"


def utc_now_iso() -> str:
    now = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def epoch_seconds_to_iso(epoch_seconds: float | int | None) -> str:
    if epoch_seconds is None:
        return ""
    timestamp = dt.datetime.fromtimestamp(float(epoch_seconds), tz=dt.timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def get_experiments_root_dir(experiments_dir: Path | None = None) -> Path:
    if experiments_dir is not None:
        root_dir = experiments_dir.expanduser().resolve()
    elif (experiments_dir_env := os.environ.get("EXPERIMENTS_DIR")) is not None:
        root_dir = Path(experiments_dir_env).expanduser().resolve()
    else:
        root_dir = Path("~/.xax/experiments").expanduser().resolve()
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


def _parse_metric_pairs(metric_pairs: list[str]) -> dict[str, str]:
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


def _load_metrics_json(metrics_json: str | None, metrics_json_file: Path | None) -> dict[str, object]:
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


def _get_experiment_id(job_payload: dict[str, object]) -> str | None:
    job_id_raw = job_payload.get("job_id")
    if isinstance(job_id_raw, str) and job_id_raw:
        return job_id_raw
    return None


def _as_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def _as_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _escape_cell(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ")


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: row.get("updated_at", "") or row.get("created_at", ""), reverse=True)


def _get_best_completed_row(rows: list[dict[str, str]], objective_mode: str) -> dict[str, str] | None:
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


def _command_session(args: SessionArgs) -> int:
    if args.new and args.resume:
        raise ValueError("Pass at most one of --new and --resume")
    if args.resume and args.name is None:
        raise ValueError("--resume requires --name")

    experiments_root_dir = get_experiments_root_dir(args.experiments_dir)
    experiment_name = args.name if args.name is not None else _default_experiment_name()
    experiment_dir = resolve_experiment_dir(experiment_name, experiments_dir=experiments_root_dir)
    experiment_exists = experiment_dir.exists()

    if args.new and experiment_exists:
        raise FileExistsError(f"Experiment already exists: {experiment_dir}")
    if args.resume and not experiment_exists:
        raise FileNotFoundError(f"Experiment does not exist: {experiment_dir}")

    mode = "resume" if experiment_exists else "new"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    ensure_experiment_templates(experiment_dir)

    out = get_cli_output(prefix="experiment")
    out.section("Session")
    out.kv("mode", mode)
    out.kv("experiment_name", experiment_name)
    out.kv("experiment_dir", experiment_dir)
    out.kv("log_path", experiment_dir / "experiment_log.csv")
    out.kv("report_path", experiment_dir / "experiment_report.md")
    return 0


def _command_upsert(args: UpsertArgs) -> int:
    if args.status is not None and args.status not in STATUS_CHOICES:
        raise ValueError(f"--status must be one of: {', '.join(STATUS_CHOICES)}")
    if args.objective_mode is not None and args.objective_mode not in ("max", "min"):
        raise ValueError("--objective-mode must be one of: max, min")

    base_metrics = _load_metrics_json(args.metrics_json, args.metrics_json_file)
    metric_pairs = _parse_metric_pairs(list(args.metric))
    merged_metrics = dict(base_metrics)
    merged_metrics.update(metric_pairs)
    log_path = resolve_log_path(
        args.log_path,
        experiment_name=args.experiment_name,
        experiments_dir=args.experiments_dir,
    )

    row_updates_raw: dict[str, str | None] = {
        "parent_experiment_id": args.parent_experiment_id,
        "queue_job_id": args.queue_job_id,
        "status": args.status,
        "task_key": args.task_key,
        "command": args.command,
        "config_path": args.config_path,
        "run_dir": args.run_dir,
        "objective_metric": args.objective_metric,
        "objective_mode": args.objective_mode,
        "objective_value": None if args.objective_value is None else str(args.objective_value),
        "hypothesis": args.hypothesis,
        "change_summary": args.change_summary,
        "result_summary": args.result_summary,
        "next_action": args.next_action,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "owner": args.owner,
        "notes": args.notes,
    }
    if merged_metrics:
        row_updates_raw["metrics_json"] = json.dumps(merged_metrics, ensure_ascii=True, sort_keys=True)
    row_updates = {key: value for key, value in row_updates_raw.items() if value is not None}

    rows = read_rows(log_path)
    was_created = upsert_row(rows, experiment_id=args.experiment_id, updates=row_updates)
    write_rows(log_path, rows)

    out = get_cli_output(prefix="experiment")
    out.status(
        "Updated %s (%s): %s",
        args.experiment_id,
        "created" if was_created else "existing",
        log_path,
    )
    return 0


def _command_sync_queue_status(args: SyncQueueStatusArgs) -> int:
    status_payload_raw = json.loads(args.status_json.read_text(encoding="utf-8"))
    if not isinstance(status_payload_raw, dict):
        raise ValueError("Queue status JSON payload must be an object")
    jobs_raw = status_payload_raw.get("jobs", [])
    if not isinstance(jobs_raw, list):
        raise ValueError("Queue status JSON payload must include `jobs` as a list")

    log_path = resolve_log_path(
        args.log_path,
        experiment_name=args.experiment_name,
        experiments_dir=args.experiments_dir,
    )
    rows = read_rows(log_path)
    synced_count = 0
    for job_payload_raw in jobs_raw:
        if not isinstance(job_payload_raw, dict):
            continue
        job_payload = {str(key): value for key, value in job_payload_raw.items()}
        if (experiment_id := _get_experiment_id(job_payload)) is None:
            continue

        started_at_iso = epoch_seconds_to_iso(_as_float(job_payload, "started_at"))
        ended_at_iso = epoch_seconds_to_iso(_as_float(job_payload, "ended_at"))
        return_code_raw = job_payload.get("return_code")
        return_code_str = "" if return_code_raw is None else str(return_code_raw)
        error_str = _as_string(job_payload, "error") or ""

        result_summary = ""
        if error_str:
            result_summary = error_str
        elif return_code_str:
            result_summary = f"return_code={return_code_str}"

        row_updates_raw: dict[str, str | None] = {
            "queue_job_id": experiment_id,
            "status": _as_string(job_payload, "status"),
            "task_key": _as_string(job_payload, "task_key"),
            "config_path": _as_string(job_payload, "config_path"),
            "run_dir": _as_string(job_payload, "run_dir"),
            "started_at": started_at_iso,
            "ended_at": ended_at_iso,
            "result_summary": result_summary if result_summary else None,
            "notes": _as_string(job_payload, "launcher"),
            "owner": args.owner,
        }
        row_updates = {key: value for key, value in row_updates_raw.items() if value is not None}
        upsert_row(rows, experiment_id=experiment_id, updates=row_updates)
        synced_count += 1

    write_rows(log_path, rows)
    out = get_cli_output(prefix="experiment")
    out.status("Synced %d jobs into %s", synced_count, log_path)
    return 0


def _command_report(args: ReportArgs) -> int:
    if args.objective_mode not in ("auto", "max", "min"):
        raise ValueError("--objective-mode must be one of: auto, max, min")

    log_path = resolve_log_path(
        args.log_path,
        experiment_name=args.experiment_name,
        experiments_dir=args.experiments_dir,
    )
    output_path = resolve_report_path(
        args.output_path,
        experiment_name=args.experiment_name,
        experiments_dir=args.experiments_dir,
    )
    rows = _sort_rows(read_rows(log_path))
    status_counts = Counter(row.get("status", "unknown") for row in rows)

    if args.objective_mode == "auto":
        first_mode = next((row.get("objective_mode", "") for row in rows if row.get("objective_mode", "")), "max")
        objective_mode = first_mode if first_mode in ("max", "min") else "max"
    else:
        objective_mode = args.objective_mode
    best_row = _get_best_completed_row(rows, objective_mode=objective_mode)

    lines: list[str] = [
        f"# {args.title}",
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
    for row in rows[: args.max_rows]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(row.get("experiment_id", "")),
                    _escape_cell(row.get("status", "")),
                    _escape_cell(row.get("objective_metric", "")),
                    _escape_cell(row.get("objective_value", "")),
                    _escape_cell(row.get("updated_at", "")),
                    _escape_cell(row.get("next_action", "")),
                ]
            )
            + " |"
        )

    report_text = "\n".join(lines).strip() + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    out = get_cli_output(prefix="experiment")
    out.status("Wrote report: %s", output_path)
    return 0


def _command_help_text(command: str) -> str:
    if command not in COMMAND_SPECS:
        raise ValueError(f"Unknown experiment command: {command}")
    spec = COMMAND_SPECS[command]
    return render_help_text(
        spec.args_type,
        prog=f"xax experiment {command}",
        description=spec.description,
    )


def _show_experiment_help() -> None:
    out = get_cli_output()
    out.plain("Usage: xax experiment <command> [args]")
    out.plain("")
    out.table(
        title="Experiment Commands",
        headers=["command", "description"],
        rows=[[command_name, spec.description] for command_name, spec in COMMAND_SPECS.items()],
    )
    out.plain("Run `xax experiment <command> --help` for command usage.")


def main(argv: list[str] | None = None) -> None:
    out = get_cli_output()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] in ("-h", "--help"):
        _show_experiment_help()
        raise SystemExit(0)

    command, sub_argv = argv_list[0], argv_list[1:]
    if command not in COMMAND_SPECS:
        out.error("Unknown experiment command: %s", command)
        _show_experiment_help()
        raise SystemExit(2)
    if any(token in ("-h", "--help") for token in sub_argv):
        out.plain(_command_help_text(command))
        raise SystemExit(0)

    try:
        match command:
            case "session":
                return_code = _command_session(parse_args_as(SessionArgs, sub_argv))
            case "upsert":
                return_code = _command_upsert(parse_args_as(UpsertArgs, sub_argv))
            case "sync-queue-status":
                return_code = _command_sync_queue_status(parse_args_as(SyncQueueStatusArgs, sub_argv))
            case "report":
                return_code = _command_report(parse_args_as(ReportArgs, sub_argv))
            case _:
                return_code = 2
    except KeyboardInterrupt:
        return_code = 130
    except Exception as error:
        out = get_cli_output(prefix="experiment")
        out.error("Command failed: %s", error)
        return_code = 1
    raise SystemExit(int(return_code))


if __name__ == "__main__":
    # python -m xax.cli.experiment
    main()
