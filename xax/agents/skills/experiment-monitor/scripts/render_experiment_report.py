"""Renders a markdown experiment report from experiment_log.csv."""

import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from experiment_log_lib import read_rows, resolve_log_path, resolve_report_path, utc_now_iso

from xax.utils.cli_args import parse_args_as

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderReportArgs:
    log_path: Path | None = None
    output_path: Path | None = None
    experiment_name: str | None = field(default=None, metadata={"help": "Experiment session name"})
    experiments_dir: Path | None = field(default=None, metadata={"help": "Override experiments root directory"})
    title: str = "Experiment Monitor Report"
    max_rows: int = 25
    objective_mode: str = "auto"


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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args_as(RenderReportArgs, list(sys.argv[1:] if argv is None else argv))
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
    logger.info("Wrote report: %s", output_path)


if __name__ == "__main__":
    main()
