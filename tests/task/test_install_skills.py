"""Tests for bundled skill installation."""

from pathlib import Path

import pytest

from xax.cli.install_skills import install_bundled_skills, main


def test_install_bundled_skills_copies_agents_tree(tmp_path: Path) -> None:
    destination_agents_dir = tmp_path / ".agents"

    copied_entry_count = install_bundled_skills(destination_agents_dir)

    assert copied_entry_count >= 1
    skill_root = destination_agents_dir / "skills" / "experiment-monitor"
    assert (skill_root / "SKILL.md").exists()
    assert (skill_root / "scripts" / "upsert_experiment_log.py").exists()
    assert (skill_root / "references" / "workflow.md").exists()
    assert (skill_root / "templates" / "experiment_log.csv").exists()
    assert (skill_root / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_install_bundled_skills_uses_existing_destination(tmp_path: Path) -> None:
    destination_agents_dir = tmp_path / ".agents"
    destination_agents_dir.mkdir(parents=True)
    marker_path = destination_agents_dir / "custom-note.txt"
    marker_path.write_text("keep", encoding="utf-8")

    install_bundled_skills(destination_agents_dir)

    assert marker_path.exists()
    assert marker_path.read_text(encoding="utf-8") == "keep"


def test_install_bundled_skills_commit_to_git_opt_in(tmp_path: Path) -> None:
    destination_agents_dir = tmp_path / ".agents"

    install_bundled_skills(destination_agents_dir, commit_to_git=True)

    skill_root = destination_agents_dir / "skills" / "experiment-monitor"
    assert not (skill_root / ".gitignore").exists()


def test_install_bundled_skills_only_ignores_new_skills(tmp_path: Path) -> None:
    destination_agents_dir = tmp_path / ".agents"
    existing_skill_root = destination_agents_dir / "skills" / "experiment-monitor"
    existing_skill_root.mkdir(parents=True)

    install_bundled_skills(destination_agents_dir)

    assert not (existing_skill_root / ".gitignore").exists()


def test_install_skills_cli_defaults_to_local_agents_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as system_exit:
        main([])

    assert system_exit.value.code == 0
    skill_root = tmp_path / ".agents" / "skills" / "experiment-monitor"
    assert (skill_root / "SKILL.md").exists()
    assert (skill_root / ".gitignore").read_text(encoding="utf-8") == "*\n"
