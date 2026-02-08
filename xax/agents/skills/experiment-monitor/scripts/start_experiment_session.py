#!/usr/bin/env python
"""Creates or resumes an experiment-monitor session directory."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_log_lib import (  # noqa: E402
    default_experiment_name,
    ensure_experiment_templates,
    get_experiments_root_dir,
    resolve_experiment_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default=None, help="Experiment session name")
    parser.add_argument("--experiments-dir", type=Path, default=None, help="Override experiments root directory")
    parser.add_argument("--new", action="store_true", help="Require creating a new session")
    parser.add_argument("--resume", action="store_true", help="Require resuming an existing session")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.new and args.resume:
        raise ValueError("Pass at most one of --new and --resume")
    if args.resume and args.name is None:
        raise ValueError("--resume requires --name")

    experiments_root_dir = get_experiments_root_dir(args.experiments_dir)
    experiment_name = args.name if args.name is not None else default_experiment_name()
    experiment_dir = resolve_experiment_dir(experiment_name, experiments_dir=experiments_root_dir)
    experiment_exists = experiment_dir.exists()

    if args.new and experiment_exists:
        raise FileExistsError(f"Experiment already exists: {experiment_dir}")
    if args.resume and not experiment_exists:
        raise FileNotFoundError(f"Experiment does not exist: {experiment_dir}")

    mode = "resume" if experiment_exists else "new"
    ensure_experiment_templates(experiment_dir)
    payload = {
        "mode": mode,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "log_path": str(experiment_dir / "experiment_log.csv"),
        "report_path": str(experiment_dir / "experiment_report.md"),
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        sys.stderr.write(f"start_experiment_session failed: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
