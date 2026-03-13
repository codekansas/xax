#!/usr/bin/env python
"""Export TensorBoard audio summaries to WAV files.

This is useful for quickly listening to `xax.Audio` metrics logged to
TensorBoard (e.g. generated TTS samples).
"""

import argparse
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _sanitize_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return "audio"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tag).strip("_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing `tensorboard/`")
    parser.add_argument(
        "--tb-subdir",
        type=str,
        default="tensorboard/heavy",
        help="TensorBoard subdirectory relative to run-dir (default: tensorboard/heavy)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/audio_exports)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional tag filter (repeatable). If provided, only tags containing any filter are exported.",
    )
    parser.add_argument(
        "--max-per-tag",
        type=int,
        default=None,
        help="If set, export at most N most recent events per tag.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    tb_dir = (run_dir / args.tb_subdir).resolve()
    if not tb_dir.exists():
        sys.stderr.write(f"TensorBoard dir not found: {tb_dir}\n")
        return 2

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else (run_dir / "audio_exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    acc = EventAccumulator(str(tb_dir))
    acc.Reload()
    audio_tags = list(acc.Tags().get("audio", []))
    if not audio_tags:
        sys.stderr.write(f"No audio tags found under: {tb_dir}\n")
        return 3

    tag_filters = [t.strip() for t in args.tag if t.strip()]
    exported = 0
    for tag in audio_tags:
        if tag_filters and not any(filt in tag for filt in tag_filters):
            continue
        events = list(acc.Audio(tag))
        if args.max_per_tag is not None and args.max_per_tag > 0:
            events = events[-args.max_per_tag :]
        for event in events:
            safe_tag = _sanitize_tag(tag)
            out_path = output_dir / f"{safe_tag}_step_{event.step}.wav"
            out_path.write_bytes(event.encoded_audio_string)
            exported += 1

    sys.stdout.write(f"exported={exported} output_dir={output_dir}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

