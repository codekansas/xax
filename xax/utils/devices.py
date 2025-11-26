"""Defines utility functions for working with devices."""

import functools
import logging
import re
import subprocess
from typing import Pattern

logger: logging.Logger = logging.getLogger(__name__)

NUMBER_REGEX: Pattern[str] = re.compile(r"[\d\.]+")


@functools.lru_cache(maxsize=None)
def get_num_gpus() -> int:
    command = "nvidia-smi --query-gpu=index --format=csv --format=csv,noheader"

    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            rows = iter(stdout.readline, "")
            return len(list(rows))

    except Exception:
        logger.exception("Caught exception while trying to query `nvidia-smi`")
        return 0


def parse_number(s: str) -> float:
    match = NUMBER_REGEX.search(s)
    if match is None:
        raise ValueError(s)
    return float(match.group())
