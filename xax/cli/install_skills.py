"""Installs bundled Codex skills into a local `.agents` directory."""

import shutil
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from xax.utils.cli_args import ARGPARSE_DEST_METADATA_KEY, parse_args_as, render_help_text
from xax.utils.logging import LOG_STATUS, configure_logging


def _write_new_skill_gitignores(destination_agents_dir: Path, existing_skill_names: set[str]) -> int:
    skills_dir = destination_agents_dir / "skills"
    if not skills_dir.exists():
        return 0

    ignored_skill_count = 0
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir() or skill_dir.name in existing_skill_names:
            continue
        gitignore_path = skill_dir / ".gitignore"
        gitignore_path.write_text("*\n", encoding="utf-8")
        ignored_skill_count += 1
    return ignored_skill_count


def install_bundled_skills(destination_agents_dir: Path, *, commit_to_git: bool = False) -> int:
    """Copies packaged `xax/agents/*` contents into the destination directory.

    Args:
        destination_agents_dir: Destination `.agents` directory path.
        commit_to_git: If set, do not write `.gitignore` files in new skill directories.

    Returns:
        Number of top-level entries copied from `xax/agents`.
    """
    destination_agents_dir.mkdir(parents=True, exist_ok=True)
    existing_skill_names: set[str] = set()
    existing_skills_dir = destination_agents_dir / "skills"
    if existing_skills_dir.exists():
        existing_skill_names = {skill_dir.name for skill_dir in existing_skills_dir.iterdir() if skill_dir.is_dir()}

    agents_resource_dir = resources.files("xax").joinpath("agents")
    if not agents_resource_dir.is_dir():
        raise FileNotFoundError("Bundled agents directory is missing from the installed xax package")

    copied_entry_count = 0
    with resources.as_file(agents_resource_dir) as source_agents_dir:
        for source_entry in source_agents_dir.iterdir():
            destination_entry = destination_agents_dir / source_entry.name
            if source_entry.is_dir():
                shutil.copytree(source_entry, destination_entry, dirs_exist_ok=True)
            else:
                destination_entry.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_entry, destination_entry)
            copied_entry_count += 1
    if not commit_to_git:
        _write_new_skill_gitignores(destination_agents_dir, existing_skill_names)
    return copied_entry_count


@dataclass(frozen=True)
class InstallSkillsArgs:
    destination_agents_dir: Path = field(
        default=Path(".agents"),
        metadata={
            ARGPARSE_DEST_METADATA_KEY: "dest",
            "help": "Destination agents directory. Defaults to ./.agents in the current working directory.",
        },
    )
    commit_to_git: bool = field(
        default=False,
        metadata={
            "help": "Allow installed skill directories to be committed; by default, new skills get `.gitignore`.",
        },
    )


def _command_install(args: InstallSkillsArgs) -> int:
    logger = configure_logging(prefix="skills")

    destination_agents_dir = args.destination_agents_dir.expanduser()
    if not destination_agents_dir.is_absolute():
        destination_agents_dir = (Path.cwd() / destination_agents_dir).resolve()

    copied_entry_count = install_bundled_skills(destination_agents_dir, commit_to_git=args.commit_to_git)
    logger.log(
        LOG_STATUS,
        "Installed %d top-level entries into %s (commit_to_git=%s)",
        copied_entry_count,
        destination_agents_dir,
        bool(args.commit_to_git),
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if any(token in ("-h", "--help") for token in argv_list):
        sys.stdout.write(
            render_help_text(
                InstallSkillsArgs,
                prog="xax install-skills",
                description="Install bundled xax Codex skills into a local .agents directory.",
            )
            + "\n"
        )
        raise SystemExit(0)
    parsed_args = parse_args_as(InstallSkillsArgs, argv_list)

    try:
        return_code = int(_command_install(parsed_args))
    except KeyboardInterrupt:
        return_code = 130
    except Exception as error:
        logger = configure_logging(prefix="skills")
        logger.error("Failed to install skills: %s", error)
        return_code = 1
    raise SystemExit(return_code)


if __name__ == "__main__":
    # python -m xax.cli.install_skills
    main()
