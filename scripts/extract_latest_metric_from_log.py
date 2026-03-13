#!/usr/bin/env python3
"""Extract the latest scalar metric match from a text log file.

This is intended for optimize-loop experiment logs, which may include ANSI
color codes from stdout logging.
"""

import argparse
import re
import sys
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Path to the log file to parse.")
    parser.add_argument("--pattern", type=str, required=True, help="Regex with at least one capture group.")
    parser.add_argument("--group", type=int, default=1, help="Capture group index for the metric value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.log.read_text(encoding="utf-8", errors="ignore")
    clean_text = ANSI_ESCAPE_RE.sub("", text)

    matches = list(re.finditer(args.pattern, clean_text, flags=re.MULTILINE))
    if not matches:
        sys.stderr.write(f"No matches for pattern in {args.log}\n")
        return 1

    try:
        value = matches[-1].group(args.group)
    except IndexError:
        sys.stderr.write(f"Pattern does not provide group {args.group}\n")
        return 2

    sys.stdout.write(f"{value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
