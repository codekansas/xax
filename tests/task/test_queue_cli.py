"""Tests for queue CLI helpers."""

from xax.cli.queue import _service_unit_text


def test_service_unit_has_robust_process_controls() -> None:
    unit_text = _service_unit_text("xax-queue-observer")
    assert "-m xax.cli.main queue start" in unit_text
    assert "-m xax.cli.main queue cleanup" in unit_text
    assert "KillMode=control-group" in unit_text
    assert "ExecStopPost=" in unit_text
    assert "OOMPolicy=continue" in unit_text
    assert "Restart=always" in unit_text
