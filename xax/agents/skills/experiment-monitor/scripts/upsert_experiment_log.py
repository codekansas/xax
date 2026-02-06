"""Upserts experiment rows in a canonical experiment log CSV."""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from experiment_log_lib import STATUS_CHOICES, read_rows, resolve_log_path, upsert_row, write_rows

from xax.utils.cli_args import parse_args_as

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpsertExperimentArgs:
    experiment_id: str
    log_path: Path | None = None
    experiment_name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    parent_experiment_id: str | None = None
    queue_job_id: str | None = None
    status: str | None = None
    task_key: str | None = None
    command: str | None = None
    config_path: str | None = None
    exp_dir: str | None = None
    objective_metric: str | None = None
    objective_mode: str | None = None
    objective_value: float | None = None
    hypothesis: str | None = None
    change_summary: str | None = None
    result_summary: str | None = None
    next_action: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    owner: str | None = None
    notes: str | None = None
    metrics_json: str | None = None
    metrics_json_file: Path | None = None
    metric: list[str] = field(default_factory=list)


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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args_as(UpsertExperimentArgs, list(sys.argv[1:] if argv is None else argv))
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
        "exp_dir": args.exp_dir,
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

    logger.info(
        "Updated %s (%s): %s",
        args.experiment_id,
        "created" if was_created else "existing",
        log_path,
    )


if __name__ == "__main__":
    main()
