# Queue Integration Notes

## Preferred Interface

Use `xax queue` commands as the primary integration surface:

- `xax queue start`
- `xax queue start --queue-gpus 0,1,2` (pin queued jobs to a GPU subset)
- `xax queue start --queue-num-gpus 3` (use first N GPUs for queue)
- `xax queue status`
- `xax queue status --json`
- `xax queue metrics <job_id> --json`
- `xax queue tail <job_id> --kind observer --follow`

Use skill scripts for experiment-log operations:

- `start_experiment_session.py`
- `sync_queue_status.py`
- `upsert_experiment_log.py`
- `render_experiment_report.py`

## Queue JSON Shape

`xax queue status --json` returns a JSON object that includes:

- `jobs`: list of job objects.

Each job should include, when available:

- `job_id` (string),
- `status` (`queued`, `running`, `completed`, `failed`, or `cancelled`),
- `task_key`,
- `run_dir`,
- `config_path`,
- `launcher`,
- `enqueued_at` (epoch seconds),
- `started_at` (epoch seconds),
- `ended_at` (epoch seconds),
- `return_code`,
- `error`.

## Current `xax` Command

- `xax queue status --json > queue_status.json`
- Ensure the user service is installed/running:
  - `xax queue start`
- Queue GPU jobs with one-at-a-time observer scheduling:
  - `python examples/ljspeech.py --launcher queued ...`
  - Queued jobs run with `MultiDeviceLauncher`.

## Adapting New CLIs

If another queue CLI emits different field names, map into the canonical columns from `references/table-schema.md` before upserting rows.
