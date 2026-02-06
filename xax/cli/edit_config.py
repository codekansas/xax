"""Lets you edit a checkpoint config programmatically."""

import difflib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from xax.task.mixins.checkpointing import load_ckpt
from xax.utils.cli_args import CLI_POSITIONAL_METADATA_KEY, parse_args_as, render_help_text
from xax.utils.structured_config import load_yaml, to_yaml_text
from xax.utils.text import colored, show_info


@dataclass(frozen=True)
class EditConfigArgs:
    ckpt_path: Path = field(metadata={CLI_POSITIONAL_METADATA_KEY: True, "help": "Path to checkpoint tar.gz"})


def _run_edit_config(args: EditConfigArgs) -> None:
    ckpt_path = args.ckpt_path

    # Loads the config from the checkpoint.
    config = load_ckpt(ckpt_path, part="config")
    config_str = to_yaml_text(config, sort_keys=True)

    # Opens the user's preferred editor to edit the config.
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(config_str.encode("utf-8"))
        f.flush()
        subprocess.run([os.environ.get("EDITOR", "vim"), f.name], check=True)

    # Loads the edited config.
    try:
        edited_config = load_yaml(f.name)
        edited_config_str = to_yaml_text(edited_config, sort_keys=True)
    finally:
        os.remove(f.name)

    if edited_config_str == config_str:
        show_info("No changes were made to the config.")
        return

    # Diffs the original and edited configs.
    diff = difflib.ndiff(config_str.splitlines(), edited_config_str.splitlines())
    for line in diff:
        if line.startswith("+ "):
            print(colored(line, "light-green"), flush=True)
        elif line.startswith("- "):
            print(colored(line, "light-red"), flush=True)
        elif line.startswith("? "):
            print(colored(line, "light-cyan"), flush=True)

    # Saves the edited config to the checkpoint.
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(ckpt_path, "r:gz") as src_tar:
            for member in src_tar.getmembers():
                if member.name != "config":  # Skip the old config file
                    src_tar.extract(member, tmp_dir)

        with tarfile.open(ckpt_path, "w:gz") as tar:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmp_dir)
                    tar.add(file_path, arcname=arcname)

            # Add the new config file
            info = tarfile.TarInfo(name="config")
            config_bytes = edited_config_str.encode()
            info.size = len(config_bytes)
            tar.addfile(info, io.BytesIO(config_bytes))


def main(argv: list[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if any(token in ("-h", "--help") for token in argv_list):
        sys.stdout.write(
            render_help_text(
                EditConfigArgs,
                prog="xax edit-config",
                description="Edit checkpoint configs in-place.",
            )
            + "\n"
        )
        raise SystemExit(0)
    parsed_args = parse_args_as(EditConfigArgs, argv_list)
    _run_edit_config(parsed_args)


if __name__ == "__main__":
    # python -m xax.cli.edit_config
    main()
