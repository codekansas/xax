---
name: experiment-monitor
description: >
  Iteratively launches and monitors experiments, watching experiment results, proposing and implementing new experiments to try, and logging all results to a common table.
---

# Experiment Monitor

Use this skill when the user wants an iterative experimentation loop:
- launch or queue experiments,
- monitor active/running jobs,
- compare outcomes against an objective,
- propose next experiments,
- keep one canonical experiment log.

## Quick Start

1. Start or resume a session under the configured experiments directory (`xax.core.conf.get_experiments_dir()`):
   - New session:
     - `python scripts/start_experiment_session.py --new --name <experiment_name>`
   - Resume session:
     - `python scripts/start_experiment_session.py --resume --name <experiment_name>`
2. If queue JSON is available, ingest it:
   - `python scripts/sync_queue_status.py --experiment-name <experiment_name> --status-json queue_status.json`
3. Add hypothesis/config/result fields:
   - `python scripts/upsert_experiment_log.py --experiment-name <experiment_name> --experiment-id <id> ...`
4. Render a readable report:
   - `python scripts/render_experiment_report.py --experiment-name <experiment_name>`

## Required Operating Rules

- Every experiment must have a stable `experiment_id`.
- Log each step in `experiment_log.csv` before proposing new experiments.
- Keep objective metadata explicit:
  - `objective_metric` (example: `val/accuracy`),
  - `objective_mode` (`max` or `min`),
  - `objective_value` (when complete).
- Do not discard failed/cancelled experiments; keep them for auditability.
- Propose no more than 1-3 new experiments per iteration unless the user asks for larger sweeps.

## Iteration Loop

1. **Baseline**
   - Identify the best completed experiment from the log.
   - If none exist, define a baseline run and objective.
2. **Monitor**
   - Pull queue/running status from CLI JSON.
   - Update status transitions in the log (`queued -> running -> completed/failed`).
3. **Analyze**
   - Compare completed runs against the objective.
   - Summarize what changed and what likely caused the result.
4. **Propose**
   - Suggest the next 1-3 experiments with concrete config/command deltas.
   - Record hypothesis + expected outcome before launch.
5. **Repeat**
   - Continue until stop criteria are reached (budget, metric target, or diminishing returns).

## Queue CLI Integration

- Prefer the project’s queue-management CLI.
- If it can emit a JSON payload with a `jobs` list, use `scripts/sync_queue_status.py`.
- Current `xax` fallback:
  - `xax queue status --json`
  - `xax queue metrics <job_id> --json`
- If the queue CLI schema changes, map it to the log schema in `references/table-schema.md`.

## Scripts

- `scripts/upsert_experiment_log.py`
  - Upsert one experiment row with hypothesis/metrics/outcome fields.
- `scripts/sync_queue_status.py`
  - Ingest queue status JSON and upsert queue/run/result fields.
- `scripts/render_experiment_report.py`
  - Generate a concise markdown report from the CSV log.
- `scripts/start_experiment_session.py`
  - Create or resume `<experiments_dir>/<experiment_name>` and initialize templates.

## References

- `references/workflow.md` for operational cadence and decision policy.
- `references/queue-integration.md` for queue JSON expectations and integration notes.
- `references/table-schema.md` for canonical log columns and semantics.
