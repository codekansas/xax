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

1. Ensure the queue observer service is installed and running:
   - `xax queue start`
   - If the machine has multiple GPUs and you want to reserve some for local dev:
     - Explicit devices: `xax queue start --queue-gpus 0,1,2`
     - First N devices: `xax queue start --queue-num-gpus 3`
2. Submit jobs through normal training commands, using queued launch mode when needed:
   - `python examples/ljspeech.py --launcher queued ...`
3. Monitor queue state and job progress directly from the queue CLI:
   - `xax queue status`
   - `xax queue status --json`
   - `xax queue metrics <job_id> --json`
   - `xax queue tail <job_id> --kind observer --follow`
4. Manage experiment-monitor session artifacts via skill scripts:
   - `python xax/agents/skills/experiment-monitor/scripts/start_experiment_session.py --new --name <experiment_name>`
   - `python xax/agents/skills/experiment-monitor/scripts/sync_queue_status.py --status-json queue_status.json --experiment-name <experiment_name>`
   - `python xax/agents/skills/experiment-monitor/scripts/upsert_experiment_log.py --experiment-id <id> --status planned ...`
   - `python xax/agents/skills/experiment-monitor/scripts/render_experiment_report.py --experiment-name <experiment_name>`
5. Iterate on hypotheses based on queue + metrics output and keep a canonical experiment log.

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

- Prefer the project’s queue-management CLI for all queue operations.
- Before running local test/debug jobs on GPU, first check which GPUs are reserved for queued training via the user service env:
  - `systemctl --user show xax-queue-observer.service --property=Environment | rg 'XAX_QUEUE_GPUS|XAX_QUEUE_NUM_GPUS'`
  - Then set `CUDA_VISIBLE_DEVICES` to a non-overlapping GPU set for local commands.
  - Goal: never run local tests on GPUs currently allocated to the queue observer.
- For GPU training jobs, queue with:
  - `--launcher queued`
  Queue execution always uses `MultiDeviceLauncher` for the running job.
- On multi-GPU machines, prefer reserving 1-2 GPUs for active development/debugging and assigning the remaining GPUs to queued jobs via:
  - `xax queue start --queue-gpus ...`
  - or `xax queue start --queue-num-gpus ...`
- Queue commands to prefer:
  - `xax queue start|stop|restart`
  - `xax queue status`
  - `xax queue move|cancel|kill`
  - `xax queue tail|metrics|tensorboard`
- Use `xax queue` for queue operations and the skill-local scripts under `xax/agents/skills/experiment-monitor/scripts/` for experiment-log operations.

## References

- `references/workflow.md` for operational cadence and decision policy.
- `references/queue-integration.md` for queue JSON expectations and integration notes.
- `references/table-schema.md` for canonical log columns and semantics.
