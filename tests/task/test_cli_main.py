"""Tests for the top-level xax CLI dispatcher."""

import pytest

from xax.cli.main import main


def test_xax_cli_help() -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["--help"])
    assert system_exit.value.code == 0


def test_xax_cli_queue_help() -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["queue", "--help"])
    assert system_exit.value.code == 0


def test_xax_cli_edit_config_help() -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["edit-config", "--help"])
    assert system_exit.value.code == 0


def test_xax_cli_rejects_removed_experiment_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as system_exit:
        main(["experiment"])

    assert system_exit.value.code == 2
    assert "Invalid subcommand" in capsys.readouterr().err
