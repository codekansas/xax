#!/usr/bin/env python
"""Merges queue status JSON jobs into experiment_log.csv."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_log_lib import (  # noqa: E402
    as_float,
    as_string,
    epoch_seconds_to_iso,
    get_experiment_id,
    read_rows,
    resolve_log_path,
    upsert_row,
    write_rows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, required=True, help="Path to queue status JSON payload")
    parser.add_argument("--log-path", type=Path, default=None, help="Override experiment log CSV path")
    parser.add_argument("--experiment-name", type=str, default="default", help="Experiment session name")
    parser.add_argument("--experiments-dir", type=Path, default=None, help="Override experiments root directory")
    parser.add_argument("--owner", type=str, default=None, help="Optional owner to apply to imported rows")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
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
        experiment_id = get_experiment_id(job_payload)
        if experiment_id is None:
            continue

        started_at_iso = epoch_seconds_to_iso(as_float(job_payload, "started_at"))
        ended_at_iso = epoch_seconds_to_iso(as_float(job_payload, "ended_at"))
        return_code_raw = job_payload.get("return_code")
        return_code_str = "" if return_code_raw is None else str(return_code_raw)
        error_str = as_string(job_payload, "error") or ""

        result_summary = ""
        if error_str:
            result_summary = error_str
        elif return_code_str:
            result_summary = f"return_code={return_code_str}"

        row_updates_raw: dict[str, str | None] = {
            "queue_job_id": experiment_id,
            "status": as_string(job_payload, "status"),
            "task_key": as_string(job_payload, "task_key"),
            "config_path": as_string(job_payload, "config_path"),
            "run_dir": as_string(job_payload, "run_dir"),
            "started_at": started_at_iso,
            "ended_at": ended_at_iso,
            "result_summary": result_summary if result_summary else None,
            "notes": as_string(job_payload, "launcher"),
            "owner": args.owner,
        }
        row_updates = {key: value for key, value in row_updates_raw.items() if value is not None}
        upsert_row(rows, experiment_id=experiment_id, updates=row_updates)
        synced_count += 1

    write_rows(log_path, rows)
    payload = {"log_path": str(log_path), "synced_jobs": synced_count}
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        sys.stderr.write(f"sync_queue_status failed: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
