"""CLI argument utilities for launcher selection."""

import sys
from dataclasses import dataclass, field
from typing import Literal, get_args

from xax.utils.cli_args import ARGPARSE_DEST_METADATA_KEY, CLI_SHORT_METADATA_KEY, parse_known_args_as

LauncherChoice = Literal["single", "s", "multi", "m", "multi_cpu", "mc", "dataset", "d", "queued", "q"]


@dataclass(frozen=True)
class LauncherCliArgs:
    launcher: LauncherChoice = field(
        default="multi",
        metadata={
            ARGPARSE_DEST_METADATA_KEY: "launcher",
            CLI_SHORT_METADATA_KEY: "l",
        },
    )


def parse_launcher_args(cli_args: list[str]) -> tuple[LauncherCliArgs, list[str]]:
    """Parses launcher-specific CLI args and returns remaining task args."""
    parsed_args, rest = parse_known_args_as(LauncherCliArgs, cli_args)
    if parsed_args.launcher not in get_args(LauncherChoice):
        raise ValueError(f"Invalid launcher choice: {parsed_args.launcher}")
    return parsed_args, rest


def help_requested(use_cli: bool | list[str]) -> bool:
    if not use_cli:
        return False
    args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
    return any(arg in ("-h", "--help") for arg in args)
