# Autoresearch: LJSpeech TTS Whisper-match quality after 10 minutes

## Objective
Optimize `examples/ljspeech.py` so that, after **10 minutes of training** on LJSpeech, the generated audio for a fixed prompt is transcribed by Whisper as closely as possible to the prompt text.

We are allowed to change model architecture, optimizer, losses, dataset preparation, tokenization, prompt handling, generation heuristics, and training schedule. We must **not** use a pretrained TTS model. Using pretrained priors for the text LM, Mimi codec, and Whisper evaluator is allowed.

## Metrics
- **Primary**: `asr_wer_prefix` (unitless, lower is better) from the final heavy eval after a 600-second training run. Prefix WER is used because extra trailing words are less important than matching the requested prompt content.
- **Secondary**:
  - `asr_cer`
  - `asr_wer_in_domain`
  - `asr_wer_gt`
  - `semantic_has_eos`
  - `invalid_q0_token_count`
  - `generated_num_frames`
  - wall-clock experiment duration

## How to Run
`./autoresearch.sh` — trains `examples/ljspeech.py` for 600 seconds (`step_kind=second`, `max_steps=600`), then prints `METRIC name=value` lines parsed from the run JSON log.

## Files in Scope
- `examples/ljspeech.py` — main TTS training/eval pipeline, model, losses, dataset prep, generation
- `scripts/extract_latest_metric_from_log.py` — helper parser if needed
- `autoresearch.sh` — benchmark harness for the 10-minute evaluation loop
- `autoresearch.md` — session context and experiment history
- `autoresearch.ideas.md` — backlog of deferred ideas

## Off Limits
- Pretrained TTS / speech-synthesis models
- Unrelated repo files unless needed for this benchmark

## Constraints
- Must keep evaluation based on Whisper transcription closeness to the prompt.
- Must measure after a 10-minute training budget.
- Must not cheat by using a pretrained TTS model.
- Prefer changes that improve fast convergence, not just long-run quality.

## What's Been Tried
- Baseline pending: current two-stage Qwen3 + Mimi-Q0/residual setup with Whisper heavy eval.
- Hypotheses to explore first:
  - shrink/retarget the semantic model so more useful adaptation happens within 10 minutes
  - bias training toward EOS/length stability so generated utterances stay aligned to the prompt
  - use more in-domain text handling and generation caps to reduce drift/repetition
  - adjust stage-1 vs stage-2 optimization balance for better early intelligibility
