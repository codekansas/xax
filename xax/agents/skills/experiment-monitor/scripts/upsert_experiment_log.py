"""Upserts experiment rows in a canonical experiment log CSV."""

import argparse
import json
import logging
from pathlib import Path

from experiment_log_lib import STATUS_CHOICES, read_rows, resolve_log_path, upsert_row, write_rows

logger = logging.getLogger(__name__)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upsert one experiment row in experiment_log.csv",
    )
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None, help="Experiment session name")
    parser.add_argument("--experiments-dir", type=Path, default=None, help="Override experiments root directory")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-experiment-id", default=None)
    parser.add_argument("--queue-job-id", default=None)
    parser.add_argument("--status", choices=STATUS_CHOICES, default=None)
    parser.add_argument("--task-key", default=None)
    parser.add_argument("--command", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--exp-dir", default=None)
    parser.add_argument("--objective-metric", default=None)
    parser.add_argument("--objective-mode", choices=("max", "min"), default=None)
    parser.add_argument("--objective-value", type=float, default=None)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--change-summary", default=None)
    parser.add_argument("--result-summary", default=None)
    parser.add_argument("--next-action", default=None)
    parser.add_argument("--started-at", default=None, help="ISO timestamp")
    parser.add_argument("--ended-at", default=None, help="ISO timestamp")
    parser.add_argument("--owner", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--metrics-json", default=None, help="JSON object literal")
    parser.add_argument("--metrics-json-file", type=Path, default=None, help="Path to JSON object file")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional metric key/value pair; can be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

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
