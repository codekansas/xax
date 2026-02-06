"""Tests for typed CLI parsing helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from xax.utils.cli_args import ARGPARSE_DEST_METADATA_KEY, parse_args_as, parse_known_args_as


@dataclass(frozen=True)
class _InstallArgs:
    destination: Path = field(metadata={ARGPARSE_DEST_METADATA_KEY: "dest"})
    commit_to_git: bool = False


@dataclass(frozen=True)
class _LauncherArgs:
    launcher: Literal["single", "multi"]


def test_parse_args_as_maps_aliased_dest() -> None:
    parsed = parse_args_as(_InstallArgs, ["--dest", "/tmp/run", "--commit-to-git"])

    assert parsed == _InstallArgs(destination=Path("/tmp/run"), commit_to_git=True)


def test_parse_known_args_as_returns_remaining_args() -> None:
    parsed, rest = parse_known_args_as(_LauncherArgs, ["--launcher", "single", "foo=bar", "abc=123"])

    assert parsed == _LauncherArgs(launcher="single")
    assert rest == ["foo=bar", "abc=123"]


def test_parse_args_as_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="Value 'dataset'"):
        parse_args_as(_LauncherArgs, ["--launcher", "dataset"])
