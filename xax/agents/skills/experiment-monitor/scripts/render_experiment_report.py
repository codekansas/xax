#!/usr/bin/env python
"""Renders experiment_report.md from experiment_log.csv."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_log_lib import (  # noqa: E402
    read_rows,
    render_report_markdown,
    resolve_log_path,
    resolve_report_path,
    sort_rows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=None, help="Override experiment log CSV path")
    parser.add_argument("--output-path", type=Path, default=None, help="Override report output path")
    parser.add_argument("--experiment-name", type=str, default="default", help="Experiment session name")
    parser.add_argument("--experiments-dir", type=Path, default=None, help="Override experiments root directory")
    parser.add_argument("--title", type=str, default="Experiment Monitor Report", help="Report title")
    parser.add_argument("--max-rows", type=int, default=25, help="Number of recent rows to include")
    parser.add_argument(
        "--objective-mode",
        type=str,
        choices=("auto", "max", "min"),
        default="auto",
        help="Objective mode",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
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
    rows = sort_rows(read_rows(log_path))
    if args.objective_mode == "auto":
        first_mode = next((row.get("objective_mode", "") for row in rows if row.get("objective_mode", "")), "max")
        objective_mode = first_mode if first_mode in ("max", "min") else "max"
    else:
        objective_mode = args.objective_mode

    report_text = render_report_markdown(
        title=args.title,
        rows=rows,
        max_rows=args.max_rows,
        objective_mode=objective_mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    payload = {"report_path": str(output_path), "rows": len(rows)}
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        sys.stderr.write(f"render_experiment_report failed: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
