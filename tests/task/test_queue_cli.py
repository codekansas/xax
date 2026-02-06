"""Tests for queue CLI helpers."""

import subprocess

import pytest

from xax.cli.queue import _service_unit_text, main


def test_service_unit_has_robust_process_controls() -> None:
    unit_text = _service_unit_text("xax-queue-observer")
    assert "-m xax.cli.main queue _observer" in unit_text
    assert "-m xax.cli.main queue cleanup" in unit_text
    assert "KillMode=control-group" in unit_text
    assert "ExecStopPost=" in unit_text
    assert "OOMPolicy=continue" in unit_text
    assert "Restart=always" in unit_text


def test_queue_start_uses_systemd(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr("xax.cli.queue.shutil.which", lambda _: "/usr/bin/systemctl")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert not check
        assert capture_output
        assert text
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("xax.cli.queue.subprocess.run", fake_run)

    with pytest.raises(SystemExit) as system_exit:
        main(["start"])

    assert system_exit.value.code == 0
    assert calls == [["systemctl", "--user", "start", "xax-queue-observer.service"]]


def test_queue_start_fails_without_systemd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("xax.cli.queue.shutil.which", lambda _: None)

    with pytest.raises(SystemExit) as system_exit:
        main(["start"])

    assert system_exit.value.code == 1
