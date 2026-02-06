# Queue Integration Notes

## Expected Input

`scripts/sync_queue_status.py` expects a JSON object with:

- `jobs`: list of job objects.

Each job should include, when available:

- `job_id` (string),
- `status` (`queued`, `running`, `completed`, `failed`, or `cancelled`),
- `task_key`,
- `exp_dir`,
- `config_path`,
- `launcher`,
- `enqueued_at` (epoch seconds),
- `started_at` (epoch seconds),
- `ended_at` (epoch seconds),
- `return_code`,
- `error`.

## Current `xax` Command

- `xax queue status --json > queue_status.json`

Then import:

- `python scripts/sync_queue_status.py --experiment-name <experiment_name> --status-json queue_status.json`

## Adapting New CLIs

If another queue CLI emits different field names, map into the canonical columns from `references/table-schema.md` before upserting rows.
