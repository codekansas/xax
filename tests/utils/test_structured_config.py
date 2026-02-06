"""Tests for structured config interpolation and coercion."""

from dataclasses import dataclass
from typing import Literal

import pytest

from xax.task.base import BaseConfig, BaseTask
from xax.utils.structured_config import merge_config_sources, resolve_interpolations


def test_resolve_interpolations_supports_env_and_dotted_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAX_TEST_LEVEL", raising=False)

    payload = {
        "directories": {"runs": "/tmp/runs"},
        "logging": {"log_level": "${oc.env:XAX_TEST_LEVEL,INFO}"},
        "report_dir": "${directories.runs}/reports",
        "runs_copy": "${directories.runs}",
    }

    resolved = resolve_interpolations(payload)
    assert resolved["logging"] == {"log_level": "INFO"}
    assert resolved["report_dir"] == "/tmp/runs/reports"
    assert resolved["runs_copy"] == "/tmp/runs"


def test_resolve_interpolations_preserves_non_string_full_match() -> None:
    payload = {"values": [1, 2, 3], "values_copy": "${values}"}

    resolved = resolve_interpolations(payload)
    assert resolved["values_copy"] == [1, 2, 3]


@dataclass(kw_only=True)
class _LiteralConfig:
    mode: Literal["train", "eval"] = "train"


def test_merge_config_sources_supports_literal_values() -> None:
    cfg = merge_config_sources(_LiteralConfig, [{"mode": "eval"}])
    assert cfg.mode == "eval"

    with pytest.raises(TypeError, match="literal options"):
        merge_config_sources(_LiteralConfig, [{"mode": "invalid"}])


@dataclass(kw_only=True)
class _TaskConfig(BaseConfig):
    root: str = "/tmp/root"
    output_dir: str = "${root}/outputs"


class _Task(BaseTask[_TaskConfig]):
    pass


def test_base_task_get_config_resolves_interpolations() -> None:
    cfg = _Task.get_config(use_cli=False)
    assert cfg.output_dir == "/tmp/root/outputs"
