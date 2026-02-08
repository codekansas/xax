"""Tests for logging configuration helpers."""

import io
import logging
from types import SimpleNamespace
from typing import Iterator

import pytest

from xax.utils.logging import ColoredFormatter, configure_logging


@pytest.fixture
def _restore_root_logger() -> Iterator[None]:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        yield
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_configure_logging_replaces_plain_console_handlers_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    _restore_root_logger: None,
) -> None:
    root_logger = logging.getLogger()

    plain_stream = io.StringIO()
    plain_handler = logging.StreamHandler(plain_stream)
    plain_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger.addHandler(plain_handler)

    colored_stream = io.StringIO()
    monkeypatch.setattr("xax.utils.logging.sys.stdout", colored_stream)
    monkeypatch.setattr(
        "xax.utils.logging.load_user_config",
        lambda: SimpleNamespace(logging=SimpleNamespace(log_level="INFO", hide_third_party_logs=False)),
    )

    configure_logging()
    configure_logging()

    console_handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]
    assert len(console_handlers) == 1
    assert isinstance(console_handlers[0].formatter, ColoredFormatter)

    root_logger.info("test-message")
    assert "test-message" in colored_stream.getvalue()
    assert plain_stream.getvalue() == ""
