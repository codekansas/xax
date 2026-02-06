"""Starts or resumes an experiment-monitor session under the experiments directory."""

import argparse
import datetime as dt
import logging
from pathlib import Path

from experiment_log_lib import ensure_experiment_templates, get_experiments_root_dir, resolve_experiment_dir

logger = logging.getLogger(__name__)


def _default_experiment_name() -> str:
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"experiment-{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a new experiment session or resume an existing one.",
    )
    parser.add_argument("--name", default=None, help="Experiment session name")
    parser.add_argument("--experiments-dir", type=Path, default=None, help="Override experiments root directory")
    parser.add_argument("--new", action="store_true", help="Require creating a new session")
    parser.add_argument("--resume", action="store_true", help="Require resuming an existing session")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

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
    logger.info("mode=%s", mode)
    logger.info("experiment_name=%s", experiment_name)
    logger.info("experiment_dir=%s", experiment_dir)
    logger.info("log_path=%s", experiment_dir / "experiment_log.csv")
    logger.info("report_path=%s", experiment_dir / "experiment_report.md")


if __name__ == "__main__":
    main()
