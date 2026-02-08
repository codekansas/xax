"""Top-level `xax` CLI entrypoint with delegated subcommands."""

import importlib
import sys
from dataclasses import dataclass
from typing import Protocol, cast

from xax.utils.cli_output import CliOutput, get_cli_output


@dataclass(frozen=True)
class _Subcommand:
    name: str
    help: str
    module_name: str


class _SubcommandModule(Protocol):
    def main(self, argv: list[str] | None = None) -> None:
        ...


def _load_subcommand_module(module_name: str) -> _SubcommandModule:
    module = importlib.import_module(module_name)
    main_fn = getattr(module, "main", None)
    if main_fn is None or not callable(main_fn):
        raise RuntimeError(f"Subcommand module {module_name} does not define callable main(argv)")
    return cast(_SubcommandModule, module)


SUBCOMMANDS: tuple[_Subcommand, ...] = (
    _Subcommand(
        name="queue",
        help="Manage queued jobs and the user systemd observer service",
        module_name="xax.cli.queue",
    ),
    _Subcommand(
        name="install-skills",
        help="Install bundled Codex skills into .agents",
        module_name="xax.cli.install_skills",
    ),
    _Subcommand(
        name="edit-config",
        help="Edit checkpoint configs in-place",
        module_name="xax.cli.edit_config",
    ),
)


def _show_help(out: CliOutput) -> None:
    out.plain("Usage: xax <command> [args]")
    out.plain("")
    out.table(
        title="Commands",
        headers=["command", "description"],
        rows=[[subcommand.name, subcommand.help] for subcommand in SUBCOMMANDS],
    )
    out.plain("Run `xax <command> --help` for command usage.")


def _exit_code_from_system_exit(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def main(argv: list[str] | None = None) -> None:
    out = get_cli_output()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] in ("-h", "--help"):
        _show_help(out)
        raise SystemExit(0)

    command = argv_list[0]
    subcommand_argv = argv_list[1:]
    subcommand = next((item for item in SUBCOMMANDS if item.name == command), None)
    if subcommand is None:
        out.error("Invalid subcommand %r.", command)
        out.plain("Choose one of: %s", ", ".join(item.name for item in SUBCOMMANDS))
        raise SystemExit(2)

    try:
        module = _load_subcommand_module(subcommand.module_name)
        module.main(subcommand_argv)
        return_code = 0
    except SystemExit as system_exit:
        return_code = _exit_code_from_system_exit(system_exit.code)

    raise SystemExit(return_code)


if __name__ == "__main__":
    # python -m xax.cli.main
    main()
