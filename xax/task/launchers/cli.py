"""Defines a launcher that can be toggled from the command line."""

import argparse
import sys
from typing import TYPE_CHECKING, Literal, get_args

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.task.launchers.multi_device import MultiDeviceLauncher
from xax.task.launchers.single_device import SingleDeviceLauncher

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


LauncherChoice = Literal["single", "multi"]


class CliLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "-l",
            "--launcher",
            choices=get_args(LauncherChoice),
            default="multi",
            help="The launcher to use",
        )
        args, cli_args_rest = parser.parse_known_intermixed_args(args=args)
        launcher_choice: LauncherChoice = args.launcher
        use_cli_next: bool | list[str] = False if not use_cli else cli_args_rest

        match launcher_choice:
            case "single":
                SingleDeviceLauncher().launch(task, *cfgs, use_cli=use_cli_next)
            case "multi":
                MultiDeviceLauncher().launch(task, *cfgs, use_cli=use_cli_next)
            case _:
                raise ValueError(f"Invalid launcher choice: {launcher_choice}")
