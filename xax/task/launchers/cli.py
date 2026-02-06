"""Defines a launcher that can be toggled from typed CLI flags."""

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.task.launchers.dataset import DatasetLauncher
from xax.task.launchers.multi_cpu import MultiCpuLauncher
from xax.task.launchers.multi_device import MultiDeviceLauncher
from xax.task.launchers.queued import QueuedLauncher
from xax.task.launchers.single_device import SingleDeviceLauncher
from xax.utils.cli_args import ARGPARSE_DEST_METADATA_KEY, CLI_SHORT_METADATA_KEY, parse_known_args_as

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


LauncherChoice = Literal["single", "s", "multi", "m", "multi_cpu", "mc", "dataset", "d", "queued", "q"]


@dataclass(frozen=True)
class LauncherCliArgs:
    launcher: LauncherChoice = field(
        default="multi",
        metadata={
            ARGPARSE_DEST_METADATA_KEY: "launcher",
            CLI_SHORT_METADATA_KEY: "l",
        },
    )


def _parse_launcher_args(cli_args: list[str]) -> tuple[LauncherCliArgs, list[str]]:
    parsed_args, rest = parse_known_args_as(LauncherCliArgs, cli_args)
    if parsed_args.launcher not in get_args(LauncherChoice):
        raise ValueError(f"Invalid launcher choice: {parsed_args.launcher}")
    return parsed_args, rest


class CliLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
        parsed_args, cli_args_rest = _parse_launcher_args(args)
        launcher_choice = parsed_args.launcher
        use_cli_next: bool | list[str] = False if not use_cli else cli_args_rest

        match launcher_choice:
            case "single" | "s":
                SingleDeviceLauncher().launch(task, *cfgs, use_cli=use_cli_next)
            case "multi" | "m":
                MultiDeviceLauncher().launch(task, *cfgs, use_cli=use_cli_next)
            case "multi_cpu" | "mc":
                MultiCpuLauncher().launch(task, *cfgs, use_cli=use_cli_next)
            case "dataset" | "d":
                from xax.task.mixins.data_loader import DataloadersMixin  # noqa: PLC0415

                if not issubclass(task, DataloadersMixin):
                    raise ValueError("The task must be a subclass of DataloadersMixin to use the dataset launcher.")
                DatasetLauncher().launch(task, *cfgs, use_cli=use_cli_next)
            case "queued" | "q":
                QueuedLauncher().launch(task, *cfgs, use_cli=use_cli_next)
            case _:
                raise ValueError(f"Invalid launcher choice: {launcher_choice}")
