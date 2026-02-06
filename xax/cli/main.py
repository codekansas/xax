"""Top-level `xax` CLI entrypoint with delegated subcommands."""

import argparse
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
    _Subcommand(name="queue", help="Manage the local queued launcher observer and jobs", delegate=queue.main),
    _Subcommand(name="install-skills", help="Install bundled Codex skills into .agents", delegate=install_skills.main),
    _Subcommand(name="edit-config", help="Edit checkpoint configs in-place", delegate=edit_config.main),
)


def _build_parser() -> argparse.ArgumentParser:
    subcommand_choices = [subcommand.name for subcommand in SUBCOMMANDS]
    subcommand_help = "\n".join([f"  {subcommand.name:14s} {subcommand.help}" for subcommand in SUBCOMMANDS])
    parser = argparse.ArgumentParser(
        prog="xax",
        description="Top-level CLI for xax utilities.",
        epilog=f"Subcommands:\n{subcommand_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=subcommand_choices, help="Subcommand to run")
    return parser


def _exit_code_from_system_exit(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] in ("-h", "--help"):
        parser.print_help()
        raise SystemExit(0)

    command = argv_list[0]
    subcommand_argv = argv_list[1:]
    delegate: Callable[[list[str] | None], None] | None = None
    for subcommand in SUBCOMMANDS:
        if subcommand.name == command:
            delegate = subcommand.delegate
            break
    if delegate is None:
        parser.error(f"invalid choice: {command!r} (choose from {[subcommand.name for subcommand in SUBCOMMANDS]})")
    assert delegate is not None

    try:
        delegate(subcommand_argv)
        return_code = 0
    except SystemExit as system_exit:
        return_code = _exit_code_from_system_exit(system_exit.code)

    raise SystemExit(return_code)


if __name__ == "__main__":
    # python -m xax.cli.main
    main()
