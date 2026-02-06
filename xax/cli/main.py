"""Top-level `xax` CLI entrypoint with delegated subcommands."""

import sys
from dataclasses import dataclass
from typing import Callable

from xax.cli import edit_config, install_skills, queue


@dataclass(frozen=True)
class _Subcommand:
    name: str
    help: str
    delegate: Callable[[list[str] | None], None]


SUBCOMMANDS: tuple[_Subcommand, ...] = (
    _Subcommand(name="queue", help="Manage queued jobs and the user systemd observer service", delegate=queue.main),
    _Subcommand(name="install-skills", help="Install bundled Codex skills into .agents", delegate=install_skills.main),
    _Subcommand(name="edit-config", help="Edit checkpoint configs in-place", delegate=edit_config.main),
)


def _build_parser() -> str:
    lines = [
        "Usage: xax <command> [args]",
        "",
        "Top-level CLI for xax utilities.",
        "",
        "Commands:",
    ]
    lines.extend([f"  {subcommand.name:14s} {subcommand.help}" for subcommand in SUBCOMMANDS])
    return "\n".join(lines)


def _exit_code_from_system_exit(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def main(argv: list[str] | None = None) -> None:
    help_text = _build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] in ("-h", "--help"):
        sys.stdout.write(help_text + "\n")
        raise SystemExit(0)

    command = argv_list[0]
    subcommand_argv = argv_list[1:]
    delegate: Callable[[list[str] | None], None] | None = None
    for subcommand in SUBCOMMANDS:
        if subcommand.name == command:
            delegate = subcommand.delegate
            break
    if delegate is None:
        sys.stderr.write(
            f"Invalid subcommand {command!r}. Choose one of: {[subcommand.name for subcommand in SUBCOMMANDS]}\n"
        )
        raise SystemExit(2)

    try:
        delegate(subcommand_argv)
        return_code = 0
    except SystemExit as system_exit:
        return_code = _exit_code_from_system_exit(system_exit.code)

    raise SystemExit(return_code)


if __name__ == "__main__":
    # python -m xax.cli.main
    main()
