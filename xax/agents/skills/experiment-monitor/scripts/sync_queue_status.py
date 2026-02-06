"""Syncs queue-status JSON payloads into experiment_log.csv."""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from experiment_log_lib import epoch_seconds_to_iso, read_rows, resolve_log_path, upsert_row, write_rows

from xax.utils.cli_args import parse_args_as

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncQueueStatusArgs:
    status_json: Path = field(metadata={"help": "Path to queue status JSON payload"})
    log_path: Path | None = None
    experiment_name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    owner: str | None = field(default=None, metadata={"help": "Optional owner to apply to imported rows"})


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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args_as(SyncQueueStatusArgs, list(sys.argv[1:] if argv is None else argv))

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
            "exp_dir": _as_string(job_payload, "exp_dir"),
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
    logger.info("Synced %d jobs into %s", synced_count, log_path)


if __name__ == "__main__":
    main()
