"""Starts or resumes an experiment-monitor session under the experiments directory."""

import datetime as dt
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from experiment_log_lib import ensure_experiment_templates, get_experiments_root_dir, resolve_experiment_dir

from xax.utils.cli_args import parse_args_as

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartExperimentArgs:
    name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    new: bool = field(default=False, metadata={"help": "Require creating a new session"})
    resume: bool = field(default=False, metadata={"help": "Require resuming an existing session"})


def _default_experiment_name() -> str:
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"experiment-{timestamp}"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args_as(StartExperimentArgs, list(sys.argv[1:] if argv is None else argv))

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
