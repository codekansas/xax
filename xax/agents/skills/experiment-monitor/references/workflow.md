# Experiment Monitoring Workflow

## Objective

Run experiments as a controlled loop, not as disconnected one-off jobs.

Each monitoring session lives in:

- `<experiments_dir>/<experiment_name>`

where `experiments_dir` comes from `xax.core.conf.get_experiments_dir()` (or its default fallback).

## Loop Cadence

1. Start from the current best completed experiment.
2. Select the smallest meaningful change for the next run.
3. Launch/queue the run and immediately log:
   - hypothesis,
   - config delta,
   - expected impact,
   - objective metric and direction.
   - On multi-GPU hosts, reserve 1-2 GPUs for local development and let queue jobs use the remaining GPUs via queue service GPU allocation flags.
4. Monitor status transitions and failures.
5. When complete, log observed metrics and write a one-line conclusion.
6. Decide next action: iterate, branch, or stop.

## Decision Policy

- Prefer depth before breadth at the beginning:
  - tune one subsystem at a time until sensitivity is understood.
- Escalate to broader sweeps only after local behavior is predictable.
- If two consecutive iterations fail to improve the objective:
  - rollback to best baseline,
  - test orthogonal changes,
  - reduce simultaneous variable changes.

## Stop Conditions

- Objective reached.
- Compute/time budget reached.
- Three consecutive non-improving runs with no strong new hypothesis.
