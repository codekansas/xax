#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/tmp/xax-ljspeech-autoresearch}"
LOG_PATH="$RUN_DIR/stdout.log"
JSON_LOG="$RUN_DIR/json/log.jsonl"

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

uv run examples/ljspeech.py \
  run_dir="$RUN_DIR" \
  max_steps=600 \
  step_kind=second \
  batch_size=8 \
  gradient_accumulation_steps=1 \
  enable_heavy_eval=true \
  log_heavy_every_n_seconds=300 \
  log_interval_seconds=5 \
  save_every_n_steps=null \
  "$@" \
  2>&1 | tee "$LOG_PATH"

python - <<'PY' "$JSON_LOG"
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing log file: {path}")

last = {}
for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    value = obj.get('value')
    if isinstance(value, dict):
        last.update(value)

required = ['asr_wer_prefix']
missing = [k for k in required if k not in last]
if missing:
    raise SystemExit(f"missing metrics: {missing}")

for key in [
    'asr_wer_prefix',
    'asr_wer',
    'asr_cer',
    'asr_wer_in_domain',
    'asr_wer_prefix_in_domain',
    'asr_wer_gt',
    'semantic_has_eos',
    'semantic_has_eos_in_domain',
    'invalid_q0_token_count',
    'invalid_q0_token_count_in_domain',
    'generated_num_frames',
    'generated_num_frames_in_domain',
]:
    if key in last:
        value = float(last[key])
        if math.isfinite(value):
            print(f"METRIC {key}={value}")
PY
