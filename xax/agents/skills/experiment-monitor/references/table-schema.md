# Experiment Log Schema

`experiment_log.csv` uses the following columns:

- `experiment_id`: stable unique id (required).
- `parent_experiment_id`: parent run id if this is a branch.
- `queue_job_id`: queue system id (often same as `experiment_id`).
- `status`: one of `planned`, `queued`, `running`, `completed`, `failed`, `cancelled`, `superseded`.
- `task_key`: task/class identifier.
- `command`: launch command or short CLI invocation.
- `config_path`: config snapshot path.
- `exp_dir`: run output directory (legacy field name in queue payloads).
- `objective_metric`: name of metric to optimize.
- `objective_mode`: `max` or `min`.
- `objective_value`: observed value for completed runs.
- `metrics_json`: JSON object for additional metrics.
- `hypothesis`: expected effect and rationale.
- `change_summary`: concise description of what changed.
- `result_summary`: concise observed outcome.
- `next_action`: follow-up recommendation.
- `started_at`: ISO timestamp.
- `ended_at`: ISO timestamp.
- `created_at`: ISO timestamp when row was created.
- `updated_at`: ISO timestamp when row was last modified.
- `owner`: owner/agent identifier.
- `notes`: free-form notes.

Guidelines:

- Do not delete old rows; preserve history.
- Update `updated_at` on every change.
- Keep `metrics_json` valid JSON.
