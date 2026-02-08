"""Tests for structured config interpolation and coercion."""

from dataclasses import dataclass, fields
from typing import Literal

import pytest

from xax.task.base import BaseConfig, BaseTask
from xax.utils.structured_config import field, merge_config_sources, render_dataclass_help, resolve_interpolations


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


@dataclass(frozen=True)
class _HelpConfig:
    value: int = field(1, help="Value help text")


def test_structured_field_sets_help_metadata() -> None:
    value_field = fields(_HelpConfig)[0]
    assert value_field.metadata["help"] == "Value help text"


@dataclass
class _MutableDefaultConfig:
    values: list[int] = field([1])


def test_structured_field_clones_mutable_defaults() -> None:
    first = _MutableDefaultConfig()
    second = _MutableDefaultConfig()

    first.values.append(2)

    assert second.values == [1]


@dataclass(frozen=True)
class _NestedHelpConfig:
    enabled: bool = field(True, help="Enable this option")


@dataclass(frozen=True)
class _RootHelpConfig:
    learning_rate: float = field(1e-3, help="Learning rate for optimization")
    nested: _NestedHelpConfig = field(_NestedHelpConfig(), help="Nested options")


@dataclass(frozen=True)
class _AlphabeticalHelpConfig:
    zeta: int = field(2, help="Z field")
    alpha: int = field(1, help="A field")


@dataclass(frozen=True)
class _FactoryHelpConfig:
    nested: _NestedHelpConfig = field(default_factory=_NestedHelpConfig, help="Nested options")
    tags: list[str] = field(default_factory=list, help="Run tags")


@dataclass(frozen=True)
class _MutableValueHelpConfig:
    logger_backend: list[str] = field(["stdout", "json", "tensorboard"], help="Logger backends")


def test_render_dataclass_help_uses_tabular_layout() -> None:
    help_text = render_dataclass_help(_RootHelpConfig, prog="train.py", use_color=False)
    assert "Usage: train.py [config.yaml ...] [key=value ...]" in help_text
    assert "Config fields:" in help_text
    assert "┌" in help_text
    assert "field" in help_text
    assert "type" in help_text
    assert "default" in help_text
    assert "description" in help_text
    assert "nested.enabled" in help_text
    assert "Learning rate for" in help_text
    assert "optimization" in help_text


def test_render_dataclass_help_sorts_fields_alphabetically() -> None:
    help_text = render_dataclass_help(_AlphabeticalHelpConfig, prog="train.py", use_color=False)
    assert help_text.index("alpha") < help_text.index("zeta")


def test_render_dataclass_help_hides_factory_defaults() -> None:
    help_text = render_dataclass_help(_FactoryHelpConfig, prog="train.py", use_color=False)
    assert "<factory>" not in help_text
    assert "factory" not in help_text.lower()
    assert "nested" in help_text
    assert "tags" in help_text
    assert "[]" in help_text


def test_render_dataclass_help_shows_mutable_value_defaults() -> None:
    help_text = render_dataclass_help(_MutableValueHelpConfig, prog="train.py", use_color=False)
    assert "['stdout', 'json'," in help_text
    assert "'tensorboard']" in help_text
