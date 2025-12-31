"""Functions for managing experiments."""

import contextlib
import datetime
import enum
import functools
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.error
import urllib.request
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Mapping, Self, Sequence, TypeVar, cast
from urllib.parse import urlparse

import git
import requests
from jaxtyping import Array
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from xax.core.conf import get_data_dir, get_pretrained_models_dir, load_user_config
from xax.core.state import State
from xax.utils.text import colored

logger = logging.getLogger(__name__)

# Date format for staging environments.
DATE_FORMAT = "%Y-%m-%d"

USER_AGENT = "xax"

T = TypeVar("T")


class CumulativeTimer:
    """Defines a simple timer to track an average value."""

    def __init__(self) -> None:
        self.steps = 0
        self.elapsed_time = 0.0

    @functools.cached_property
    def start_time(self) -> float:
        return time.time()

    def step(self, steps: int, cur_time: float) -> None:
        if steps != self.steps:
            self.steps = steps
            self.elapsed_time = cur_time - self.start_time

    @property
    def steps_per_second(self) -> float:
        return 0.0 if self.elapsed_time < 1e-4 else self.steps / self.elapsed_time

    @property
    def steps_per_hour(self) -> float:
        return self.steps_per_second * 60 * 60

    @property
    def seconds_per_step(self) -> float:
        return 0.0 if self.steps <= 0 else self.elapsed_time / self.steps

    @property
    def hours_per_step(self) -> float:
        return self.seconds_per_step / (60 * 60)


class IterationTimer:
    """Defines a simple timer to track consecutive values."""

    def __init__(self) -> None:
        self.iteration_time = 0.0
        self.last_time = time.time()

    def step(self, cur_time: float) -> None:
        self.iteration_time = cur_time - self.last_time
        self.last_time = cur_time

    @property
    def iter_seconds(self) -> float:
        return self.iteration_time

    @property
    def iter_hours(self) -> float:
        return self.iter_seconds / (60 * 60)


class StateTimer:
    """Defines a timer for all state information."""

    def __init__(self) -> None:
        self.step_timer = CumulativeTimer()
        self.sample_timer = CumulativeTimer()
        self.iter_timer = IterationTimer()

    def step(self, state: State) -> None:
        cur_time = time.time()
        num_steps = int(state.num_steps.item())
        num_samples = int(state.num_samples.item())
        self.step_timer.step(num_steps, cur_time)
        self.sample_timer.step(num_samples, cur_time)
        self.iter_timer.step(cur_time)

    def log_dict(self) -> dict[str, int | float | tuple[int | float, bool]]:
        return {
            "steps/second": self.step_timer.steps_per_second,
            "samples/second": (self.sample_timer.steps_per_second, True),
            "dt": self.iter_timer.iter_seconds,
        }


class IntervalTicker:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.last_tick_time: float | None = None

    def tick(self, elapsed_time: float) -> bool:
        if self.last_tick_time is None or elapsed_time - self.last_tick_time > self.interval:
            self.last_tick_time = elapsed_time
            return True
        return False


class ContextTimer:
    def __init__(self) -> None:
        self.start_time = 0.0
        self.elapsed_time = 0.0

    def __enter__(self) -> Self:
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.elapsed_time = time.time() - self.start_time


def abs_path(path: str) -> str:
    return str(Path(path).resolve())


OmegaConf.register_new_resolver("xax.abs_path", abs_path, replace=True)


def cpu_count(default: int) -> int:
    if (cpu_count := os.cpu_count()) is not None:
        return cpu_count
    return default


OmegaConf.register_new_resolver("xax.cpu_count", cpu_count, replace=True)


def date_str(_: str) -> str:
    return time.strftime("%Y-%m-%d")


OmegaConf.register_new_resolver("xax.date_str", date_str, replace=True)


def get_random_port(default: int = 1337) -> int:
    try:
        return (hash(time.time()) + random.randint(0, 100000)) % (65_535 - 10_000) + 10_000
    except Exception:
        return default


OmegaConf.register_new_resolver("xax.get_random_port", get_random_port, replace=True)


class NaNError(Exception):
    """Raised when NaNs are detected in the model parameters."""


def diff_configs(
    first: Mapping | Sequence,
    second: Mapping | Sequence,
    prefix: str | None = None,
) -> tuple[list[str], list[str]]:
    """Returns the difference between two configs.

    Args:
        first: The first (original) config
        second: The second (new) config
        prefix: The prefix to check (used for recursion, not main call)

    Returns:
        Two lists of lines describing the diff between the two configs
    """

    def get_diff_string(prefix: str | None, val: Any) -> str:  # noqa: ANN401
        if isinstance(val, (str, float, int)):
            return f"{prefix}={val}"
        return f"{prefix}= ... ({type(val)})"

    def cast_enums(k: Any) -> Any:  # noqa: ANN401
        return k.name if isinstance(k, enum.Enum) else k

    new_first: list[str] = []
    new_second: list[str] = []

    any_config = (ListConfig, DictConfig)

    if isinstance(first, Mapping) and isinstance(second, Mapping):
        first_keys, second_keys = cast(set[str], set(first.keys())), cast(set[str], set(second.keys()))

        # Gets the new keys in each config.
        new_first += [f"{prefix}.{key}" for key in first_keys.difference(second_keys)]
        new_second += [f"{prefix}.{key}" for key in second_keys.difference(first_keys)]

        # Gets the new sub-keys in each config.
        for key in first_keys.intersection(second_keys):
            sub_prefix = key if prefix is None else f"{prefix}.{key}"
            if isinstance(first, DictConfig) and isinstance(second, DictConfig):
                if OmegaConf.is_missing(first, key) or OmegaConf.is_missing(second, key):
                    if not OmegaConf.is_missing(first, key):
                        new_first += [get_diff_string(sub_prefix, first[key])]
                    if not OmegaConf.is_missing(second, key):
                        new_second += [get_diff_string(sub_prefix, second[key])]
            elif isinstance(first[key], any_config) and isinstance(second[key], any_config):
                sub_new_first, sub_new_second = diff_configs(first[key], second[key], prefix=sub_prefix)
                new_first, new_second = new_first + sub_new_first, new_second + sub_new_second
            elif cast_enums(first[key]) != cast_enums(second[key]):
                first_val, second_val = first[key], second[key]
                new_first += [get_diff_string(sub_prefix, first_val)]
                new_second += [get_diff_string(sub_prefix, second_val)]

    elif isinstance(first, Sequence) and isinstance(second, Sequence):
        if len(first) > len(second):
            for i in range(len(second), len(first)):
                new_first += [get_diff_string(prefix, first[i])]
        elif len(second) > len(first):
            for i in range(len(first), len(second)):
                new_second += [get_diff_string(prefix, second[i])]

        for i in range(min(len(first), len(second))):
            sub_prefix = str(i) if prefix is None else f"{prefix}.{i}"
            if isinstance(first[i], any_config) and isinstance(second[i], any_config):
                sub_new_first, sub_new_second = diff_configs(first[i], second[i], prefix=sub_prefix)
                new_first, new_second = new_first + sub_new_first, new_second + sub_new_second
    else:
        new_first += [get_diff_string(prefix, first)]
        new_second += [get_diff_string(prefix, second)]

    return new_first, new_second


def get_diff_string(config_diff: tuple[list[str], list[str]]) -> str | None:
    added_keys, deleted_keys = config_diff
    if not added_keys and not deleted_keys:
        return None
    change_lines: list[str] = []
    change_lines += [f" ↪ {colored('+', 'green')} {added_key}" for added_key in added_keys]
    change_lines += [f" ↪ {colored('-', 'red')} {deleted_key}" for deleted_key in deleted_keys]
    change_summary = "\n".join(change_lines)
    return change_summary


def save_config(config_path: Path, raw_config: DictConfig) -> None:
    if config_path.exists():
        config_diff = diff_configs(raw_config, cast(DictConfig, OmegaConf.load(config_path)))
        diff_string = get_diff_string(config_diff)
        if diff_string is not None:
            logger.warning("Overwriting config %s:\n%s", config_path, diff_string)
            OmegaConf.save(raw_config, config_path)
    else:
        config_path.parent.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(raw_config, config_path)
        logger.info("Saved config to %s", config_path)


def to_markdown_table(config: DictConfig) -> str:
    """Converts a config to a markdown table string.

    Args:
        config: The config to convert to a table.

    Returns:
        The config, formatted as a Markdown string.
    """

    def format_as_string(value: Any) -> str:  # noqa: ANN401
        if isinstance(value, str):
            return value
        if isinstance(value, Array):
            value = value.item()
        if isinstance(value, (int, float)):
            return f"{value:.4g}"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        if isinstance(value, datetime.timedelta):
            return f"{value.total_seconds():.4g}s"
        if value is None:
            return ""
        if value is MISSING:
            return ""
        return str(value)

    def iter_flat(config: dict) -> Iterator[tuple[list[str | None], str]]:
        for key, value in reversed(config.items()):
            if isinstance(value, dict):
                is_first = True
                for sub_key_list, sub_value in iter_flat(value):
                    yield [format_as_string(key) if is_first else None] + sub_key_list, sub_value
                    is_first = False
            elif isinstance(value, (list, tuple)):
                is_first = True
                for i, sub_value in enumerate(value):
                    for sub_key_list, sub_sub_value in iter_flat({f"{i}": sub_value}):
                        yield [format_as_string(key) if is_first else None] + sub_key_list, sub_sub_value
                        is_first = False
            else:
                yield [format_as_string(key)], format_as_string(value)

    config_dict = cast(dict, OmegaConf.to_container(config, resolve=True, throw_on_missing=False, enum_to_str=True))
    config_flat = list(iter_flat(config_dict))

    # Gets rows of strings.
    rows: list[list[str]] = []
    for key_list, value in config_flat:
        row = ["" if key is None else key for key in key_list] + [value]
        rows.append(row)

    # Pads all rows to the same length.
    max_len = max(len(row) for row in rows)
    rows = [row[:-1] + [""] * (max_len - len(row)) + row[-1:] for row in rows]

    # Converts to a markdown table.
    header_str = "| " + " | ".join([f"key_{i}" for i in range(max_len - 1)]) + " | value |"
    header_sep_str = "|-" + "-|-" * (max_len - 1) + "-|"
    rows_str = "\n".join(["| " + " | ".join(row) + " |" for row in rows])
    return "\n".join([header_str, header_sep_str, rows_str])


def stage_environment(obj: object, root: Path) -> None:
    """Stages the current task to a staging directory.

    Args:
        obj: The object with the module to stage.
        root: The root directory to stage to.
    """
    root.mkdir(exist_ok=True, parents=True)

    # Gets the path to the root module. This is done heuristically, so it may
    # not work in all cases, but it should generally work.
    if (mod := inspect.getmodule(obj.__class__)) is None:
        raise RuntimeError(f"Could not find module for task {obj.__class__}!")
    if (spec := mod.__spec__) is None:
        raise RuntimeError(f"Could not find spec for module {mod}!")
    if spec.origin is None:
        raise RuntimeError(f"Could not find origin for spec {spec}!")
    root_mod = spec.name.split(".", 1)[0]
    path_parts = Path(spec.origin).parts[:-1]
    if root_mod not in path_parts:
        raise RuntimeError(f"Could not find root module {root_mod} in path {path_parts}!")
    root_path = Path(*path_parts[: path_parts.index(root_mod) + 1])

    # Gets files to stage.
    fpaths: set[tuple[Path, Path]] = set()
    for module in sys.modules.values():
        if (fpath_str := getattr(module, "__file__", None)) is None:
            continue
        fpath = Path(fpath_str).resolve()
        try:
            rel_fpath = fpath.relative_to(root_path)
            fpaths.add((fpath, rel_fpath))
        except ValueError:
            pass

    # Computes hash of all files and return if it matches the previous hash.
    hashobj = hashlib.md5()
    for fpath, _ in fpaths:
        with open(fpath, "rb") as f:
            while data := f.read(65536):
                hashobj.update(data)
    hashval = hashobj.hexdigest()
    prev_hashval: str | None = None
    hash_file = root / ".hash"
    if hash_file.exists():
        prev_hashval = hash_file.read_text().strip()
    if prev_hashval == hashval:
        return
    hash_file.write_text(hashval)

    # Copies all files to the staging directory.
    if (root / root_mod).exists():
        shutil.rmtree(root / root_mod, ignore_errors=True)
    for fpath, rel_fpath in fpaths:
        new_fpath = root / root_mod / rel_fpath
        new_fpath.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(fpath, new_fpath)


def get_git_state(obj: object) -> str:
    """Gets the state of the Git repo that an object is in as a string.

    Args:
        obj: The object which is in the target Git repo.
        width: The width of the text blocks.

    Returns:
        A nicely-formatted string showing the current task's Git state.
    """
    try:
        task_file = inspect.getfile(type(obj))
        repo = git.Repo(task_file, search_parent_directories=True)
        branch = repo.active_branch
        commit = repo.head.commit
        status = textwrap.indent(str(repo.git.status()), "    ")
        diff = textwrap.indent(str(repo.git.diff(color=False)), "    ")
        return "\n".join(
            [
                f"Path: {task_file}",
                f"Branch: {branch}",
                f"Commit: {commit}",
                "Status:",
                status,
                "Diff:",
                diff,
            ]
        )

    except Exception:
        return traceback.format_exc()


def get_packages_with_versions() -> str:
    """Gets the packages and their versions.

    Returns:
        A dictionary of packages and their versions.
    """
    packages = [(dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()]
    return "\n".join([f"{key}=={version}" for key, version in sorted(packages)])


def get_command_line_string() -> str:
    return " ".join(sys.argv)


def get_environment_variables() -> str:
    return "\n".join([f"{key}={value}" for key, value in sorted(os.environ.items())])


def get_state_file_string(obj: object) -> str:
    return "\n\n".join(
        [
            f"=== Command Line ===\n\n{get_command_line_string()}",
            f"=== Git State ===\n\n{get_git_state(obj)}",
            f"=== Packages ===\n\n{get_packages_with_versions()}",
            f"=== Environment Variables ===\n\n{get_environment_variables()}",
        ]
    )


def get_info_json() -> str:
    return json.dumps(
        {
            "process_id": os.getpid(),
            "job": {
                "start_time": datetime.datetime.now().isoformat(),
            },
        },
        indent=2,
    )


def get_training_code(obj: object) -> str:
    """Gets the text from the file containing the provided object.

    Args:
        obj: The object to get the file from.

    Returns:
        The text from the file containing the object.
    """
    try:
        task_file = inspect.getfile(type(obj))
        with open(task_file, "r") as f:
            return f.read()
    except Exception:
        return traceback.format_exc()


def check_md5(file_path: str | Path, hash_str: str | None, chunk_size: int = 2**16) -> bool:
    """Checks the MD5 of the downloaded file.

    Args:
        file_path: Path to the downloaded file.
        hash_str: Expected MD5 of the file; if None, return True.
        chunk_size: Size of the chunks to read from the file.

    Returns:
        True if the MD5 matches, False otherwise.
    """
    if hash_str is None:
        return True

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest() == hash_str


def check_sha256(file_path: str | Path, hash_str: str | None, chunk_size: int = 2**16) -> bool:
    """Checks the SHA256 of the downloaded file.

    Args:
        file_path: Path to the downloaded file.
        hash_str: Expected SHA256 of the file; if None, return True.
        chunk_size: Size of the chunks to read from the file.

    Returns:
        True if the SHA256 matches, False otherwise.
    """
    if hash_str is None:
        return True

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest() == hash_str


class BaseFileDownloader(ABC):
    """Provides a simple interface for downloading URLs.

    This class is meant to be subclassed to provide different download
    locations. For example, when downloading pretrained models, use the
    :class:`ModelDownloader` class.

    Typically, you should simply use the :func:`ensure_downloaded` function
    to make sure the file is downloaded to the correct location.

    This is adapted in large part from the reference implementation in the
    Torchvision library.

    Parameters:
        url: The URL to download from.
        dnames: The directory names to download to.
        md5: The expected MD5 of the file.
        sha256: The expected SHA256 of the file.
        is_tmp: Whether to download to a temporary directory.
        recheck_hash: Whether to recheck the hash after downloading.
        max_redirect_hops: The maximum number of redirects to follow.
    """

    def __init__(
        self,
        url: str,
        *dnames: str,
        md5: str | None = None,
        sha256: str | None = None,
        is_tmp: bool = False,
        recheck_hash: bool = False,
        max_redirect_hops: int = 3,
    ) -> None:
        super().__init__()

        assert len(dnames) >= 1, "Must provide at least 1 directory name"
        filepath = Path(tempfile.mkdtemp("models")) if is_tmp else self.get_root_directory()
        for dname in dnames:
            filepath = filepath / dname
        (root := filepath.parent).mkdir(parents=True, exist_ok=True)

        self.url = url
        self.filename = filepath.name
        self.root = root
        self.md5 = md5
        self.sha256 = sha256
        self.recheck_hash = recheck_hash
        self.max_redirect_hops = max_redirect_hops

    @abstractmethod
    def get_root_directory(self) -> Path: ...

    @property
    def filepath(self) -> Path:
        return self.root / self.filename

    @property
    def is_downloaded(self) -> bool:
        if not self.filepath.exists():
            return False
        if self.recheck_hash and not self.check_hashes():
            logger.warning("A file was found for %s in %s, but its hashes do not match.", self.url, self.filepath)
            self.filepath.unlink()
            return False
        return True

    def check_hashes(self) -> bool:
        return check_sha256(self.filepath, self.sha256) and check_md5(self.filepath, self.md5)

    def ensure_downloaded(self) -> Path:
        """Ensures the file is downloaded and returns the path to it.

        By default, we only check the hash once when the file is downloaded,
        and we don't bother rechecking unless ``recheck_hash`` is set to True.

        Returns:
            The path to the downloaded file.
        """
        if not self.is_downloaded:
            self.download()
            if not self.check_hashes():
                self.filepath.unlink()
                raise RuntimeError(f"Hashes for {self.filepath} do not match. The corruped file has been deleted.")
        return self.filepath

    def download(self) -> None:
        root = self.root.expanduser()
        root.mkdir(parents=True, exist_ok=True)

        # Expands the redirect chain if needed.
        url = self._get_redirect_url(self.url, max_hops=self.max_redirect_hops)

        # Checks if file is located on Google Drive.
        file_id = self._get_google_drive_file_id(url)
        if file_id is not None:
            return self.download_file_from_google_drive(file_id, root, self.filename)

        # Downloads the file.
        try:
            logger.info("Downloading %s to %s", url, self.filepath)
            self._urlretrieve(url, self.filepath)
        except (urllib.error.URLError, OSError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                logger.warning("Download failed. Trying HTTP instead of HTTPS: %s to %s", url, self.filepath)
                self._urlretrieve(url, self.filepath)
            else:
                raise e

    @classmethod
    def _save_response_content(cls, content: Iterator[bytes], destination: Path) -> None:
        with open(destination, "wb") as fh:
            for chunk in content:
                if not chunk:  # Filter out keep-alive new chunks.
                    continue
                fh.write(chunk)

    @classmethod
    def _urlretrieve(cls, url: str, filename: Path, chunk_size: int = 1024 * 32) -> None:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            cls._save_response_content(iter(lambda: response.read(chunk_size), b""), filename)

    @classmethod
    def _extract_gdrive_api_response(
        cls,
        response: requests.Response,
        chunk_size: int = 32 * 1024,
    ) -> tuple[str | None, Iterator[bytes]]:
        content = response.iter_content(chunk_size)
        first_chunk = None
        while not first_chunk:  # Filter out keep-alive new chunks.
            first_chunk = next(content)
        content = itertools.chain([first_chunk], content)

        try:
            match = re.search("<title>Google Drive - (?P<api_response>.+?)</title>", first_chunk.decode())
            api_response = match["api_response"] if match is not None else None
        except UnicodeDecodeError:
            api_response = None
        return api_response, content

    @classmethod
    def download_file_from_google_drive(cls, file_id: str, root: Path, filename: str | None = None) -> None:
        root = root.expanduser()
        if not filename:
            filename = file_id
        fpath = root / filename
        root.mkdir(parents=True, exist_ok=True)

        url = "https://drive.google.com/uc"
        params = dict(id=file_id, export="download")
        with requests.Session() as session:
            response = session.get(url, params=params, stream=True)

            token: str | None = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value
                    break
            else:
                api_response, content = cls._extract_gdrive_api_response(response)
                token = "t" if api_response == "Virus scan warning" else None

            if token is not None:
                response = session.get(url, params=dict(params, confirm=token), stream=True)
                api_response, content = cls._extract_gdrive_api_response(response)

            if api_response == "Quota exceeded":
                raise RuntimeError(
                    f"The daily quota of the file {filename} is exceeded and it "
                    f"can't be downloaded. This is a limitation of Google Drive "
                    f"and can only be overcome by trying again later."
                )

            cls._save_response_content(content, fpath)

        # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB with only text.
        if os.stat(fpath).st_size < 10 * 1024:
            with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:
                text = fh.read()

                # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
                if re.search(r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)", text):
                    warnings.warn(
                        f"We detected some HTML elements in the downloaded file. "
                        f"This most likely means that the download triggered an unhandled API response by GDrive. "
                        f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "
                        f"the response:\n\n{text}",
                        stacklevel=2,
                    )

    @classmethod
    def _get_google_drive_file_id(cls, url: str) -> str | None:
        parts = urlparse(url)
        if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
            return None
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        if match is None:
            return None
        return match.group("id")

    @classmethod
    def _get_redirect_url(cls, url: str, max_hops: int = 3) -> str:
        initial_url = url
        headers = {"Method": "HEAD", "User-Agent": USER_AGENT}
        for _ in range(max_hops + 1):
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
                if response.url == url or response.url is None:
                    return url
                url = response.url
        raise RecursionError(f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect was {url}.")


class ModelDownloader(BaseFileDownloader):
    def get_root_directory(self) -> Path:
        return get_pretrained_models_dir()


class DataDownloader(BaseFileDownloader):
    def get_root_directory(self) -> Path:
        return get_data_dir()


def get_state_dict_prefix(
    ckpt: dict[str, T],
    prefix: str | None = None,
    suffix: str | None = None,
    regexp: re.Pattern[str] | None = None,
) -> dict[str, T]:
    """Returns the parts of a checkpoint which begin with a prefix.

    Args:
        ckpt: The checkpoint to modify
        prefix: The prefix to clip
        suffix: The suffix to clip
        regexp: The regexp to search for (doesn't modify any keys)

    Returns:
        The modified checkpoint
    """
    if prefix is not None:
        ckpt = {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}
    if suffix is not None:
        ckpt = {k[: -len(suffix)]: v for k, v in ckpt.items() if k.endswith(suffix)}
    if regexp is not None:
        ckpt = {k: v for k, v in ckpt.items() if regexp.match(k)}
    return ckpt


def split_n_items_across_workers(n: int, worker_id: int, num_workers: int) -> tuple[int, int]:
    """Computes offsets for splitting N items across K workers.

    This returns the start and end indices for the items to be processed by the
    given worker. The end index is exclusive.

    Args:
        n: The number of items to process.
        worker_id: The ID of the current worker.
        num_workers: The total number of workers.

    Returns:
        The start and end index for the items in the current worker.
    """
    assert n >= num_workers, f"n ({n}) must be >= num_workers ({num_workers})"
    assert 0 <= worker_id < num_workers, f"worker_id ({worker_id}) must be >= 0 and < num_workers ({num_workers})"

    # The number of items to process per worker.
    items_per_worker = math.ceil(n / num_workers)

    # The start and end indices for the items to process.
    start = worker_id * items_per_worker
    end = min(start + items_per_worker, n)

    return start, end


def num_workers(default: int) -> int:
    max_workers = load_user_config().experiment.max_workers
    if hasattr(os, "sched_getaffinity"):
        try:
            return min(len(os.sched_getaffinity(0)), max_workers)
        except Exception:
            pass
    if (cpu_count := os.cpu_count()) is not None:
        return min(cpu_count, max_workers)
    return min(default, max_workers)


OmegaConf.register_new_resolver("xax.num_workers", num_workers, replace=True)
