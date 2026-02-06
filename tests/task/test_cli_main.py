"""Tests for the top-level xax CLI dispatcher."""

from pathlib import Path

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


def test_xax_cli_install_skills(tmp_path: Path) -> None:
    destination_agents_dir = tmp_path / ".agents"
    with pytest.raises(SystemExit) as system_exit:
        main(["install-skills", "--dest", str(destination_agents_dir)])
    assert system_exit.value.code == 0
    assert (destination_agents_dir / "skills" / "experiment-monitor" / "SKILL.md").exists()
