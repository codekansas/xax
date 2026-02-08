#!/usr/bin/env python
"""Inserts or updates one experiment row in experiment_log.csv."""

import argparse
import json
import sys
from pathlib import Path
from typing import get_args

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_log_lib import (  # noqa: E402
    ExperimentStatus,
    load_metrics_json,
    parse_metric_pairs,
    read_rows,
    resolve_log_path,
    upsert_row,
    write_rows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", type=str, required=True, help="Stable experiment id")
    parser.add_argument("--log-path", type=Path, default=None, help="Override experiment log CSV path")
    parser.add_argument("--experiment-name", type=str, default="default", help="Experiment session name")
    parser.add_argument("--experiments-dir", type=Path, default=None, help="Override experiments root directory")
    parser.add_argument("--parent-experiment-id", type=str, default=None, help="Parent experiment id")
    parser.add_argument("--queue-job-id", type=str, default=None, help="Queue job id")
    parser.add_argument(
        "--status",
        type=str,
        default=None,
        choices=get_args(ExperimentStatus),
        help="Experiment status",
    )
    parser.add_argument("--task-key", type=str, default=None, help="Task key")
    parser.add_argument("--command", type=str, default=None, help="Command string")
    parser.add_argument("--config-path", type=str, default=None, help="Config snapshot path")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory path")
    parser.add_argument("--objective-metric", type=str, default=None, help="Objective metric")
    parser.add_argument("--objective-mode", type=str, default=None, choices=("max", "min"), help="Objective mode")
    parser.add_argument("--objective-value", type=float, default=None, help="Objective value")
    parser.add_argument("--hypothesis", type=str, default=None, help="Hypothesis")
    parser.add_argument("--change-summary", type=str, default=None, help="What changed")
    parser.add_argument("--result-summary", type=str, default=None, help="Observed result")
    parser.add_argument("--next-action", type=str, default=None, help="Recommended next step")
    parser.add_argument("--started-at", type=str, default=None, help="ISO start timestamp")
    parser.add_argument("--ended-at", type=str, default=None, help="ISO end timestamp")
    parser.add_argument("--owner", type=str, default=None, help="Owner name")
    parser.add_argument("--notes", type=str, default=None, help="Free-form notes")
    parser.add_argument("--metrics-json", type=str, default=None, help="JSON string object of extra metrics")
    parser.add_argument("--metrics-json-file", type=Path, default=None, help="Path to JSON object with extra metrics")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeated metric entries (can be passed multiple times)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    base_metrics = load_metrics_json(args.metrics_json, args.metrics_json_file)
    metric_pairs = parse_metric_pairs(list(args.metric))
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
    payload = {
        "experiment_id": args.experiment_id,
        "log_path": str(log_path),
        "mode": "created" if was_created else "updated",
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        sys.stderr.write(f"upsert_experiment_log failed: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
