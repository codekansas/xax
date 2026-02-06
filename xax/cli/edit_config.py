"""Lets you edit a checkpoint config programmatically."""

import argparse
import difflib
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from xax.task.mixins.checkpointing import load_ckpt
from xax.utils.text import colored, show_info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="xax edit-config")
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args(argv)

    # Loads the config from the checkpoint.
    config = load_ckpt(args.ckpt_path, part="config")
    config_str = OmegaConf.to_yaml(config)

    # Opens the user's preferred editor to edit the config.
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(config_str.encode("utf-8"))
        f.flush()
        subprocess.run([os.environ.get("EDITOR", "vim"), f.name], check=True)

    # Loads the edited config.
    try:
        edited_config = OmegaConf.load(f.name)
        edited_config_str = OmegaConf.to_yaml(edited_config, sort_keys=True)
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
        with tarfile.open(args.ckpt_path, "r:gz") as src_tar:
            for member in src_tar.getmembers():
                if member.name != "config":  # Skip the old config file
                    src_tar.extract(member, tmp_dir)

        with tarfile.open(args.ckpt_path, "w:gz") as tar:
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


if __name__ == "__main__":
    # python -m xax.cli.edit_config
    main()
