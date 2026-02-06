"""Logging utilities."""

import logging
import math
import socket
import sys
from typing import Any

from jaxtyping import Array

from xax.core.conf import load_user_config
from xax.utils.jax import to_scalar
from xax.utils.text import Color, color_parts, colored

# Logging level to show on all ranks.
LOG_INFO_ALL: int = logging.INFO + 1
LOG_DEBUG_ALL: int = logging.DEBUG + 1

# Show as a transient message.
LOG_PING: int = logging.INFO + 2

# Show as a persistent status message.
LOG_STATUS: int = logging.INFO + 3

# Reserved for error summary.
LOG_ERROR_SUMMARY: int = logging.INFO + 4


def format_number(value: int | float, precision: int = 4) -> str:
    if isinstance(value, int):
        return f"{value:,}"  # Add commas to the number
    return f"{value:,.{precision}g}"


def as_str(value: Any, precision: int = 4) -> str:  # noqa: ANN401
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, Array):
        value = to_scalar(value)
    if isinstance(value, (int, float)):
        return format_number(value, precision)
    raise TypeError(f"Unexpected log type: {type(value)}")


class RankFilter(logging.Filter):
    def __init__(self, *, rank: int | None = None) -> None:
        """Logging filter which filters out INFO logs on non-zero ranks.

        Args:
            rank: The current rank
        """
        super().__init__()

        self.rank = rank

        # Log using INFOALL to show on all ranks.
        logging.addLevelName(LOG_INFO_ALL, "INFOALL")
        logging.addLevelName(LOG_DEBUG_ALL, "DEBUGALL")
        logging.addLevelName(LOG_PING, "PING")
        logging.addLevelName(LOG_STATUS, "STATUS")
        logging.addLevelName(LOG_ERROR_SUMMARY, "ERROR_SUMMARY")

        self.log_all_ranks = {
            logging.getLevelName(level)
            for level in (
                LOG_DEBUG_ALL,
                LOG_INFO_ALL,
                LOG_STATUS,
                logging.CRITICAL,
                logging.ERROR,
                logging.WARNING,
            )
        }

        self.log_no_ranks = {logging.getLevelName(level) for level in (LOG_ERROR_SUMMARY,)}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelname in self.log_no_ranks:
            return False
        if self.rank is None or self.rank == 0:
            return True
        if record.levelname in self.log_all_ranks:
            return True
        return False


class ColoredFormatter(logging.Formatter):
    """Defines a custom formatter for displaying logs."""

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS: dict[str, Color] = {
        "WARNING": "yellow",
        "INFOALL": "magenta",
        "INFO": "cyan",
        "DEBUGALL": "grey",
        "DEBUG": "grey",
        "CRITICAL": "yellow",
        "FATAL": "red",
        "ERROR": "red",
        "STATUS": "green",
        "PING": "magenta",
    }

    def __init__(
        self,
        *,
        prefix: str | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        use_color: bool = True,
    ) -> None:
        asc_start, asc_end = color_parts("grey")
        name_start, name_end = color_parts("blue", bold=True)

        message_pre = [
            "{levelname:^19s}",
            asc_start,
            "{asctime}",
            asc_end,
            " [",
            name_start,
            "{name}",
            name_end,
            "]",
        ]
        message_post = [" {message}"]

        if prefix is not None:
            message_pre += [" ", colored(prefix, "magenta", bold=True)]

        if rank is not None or world_size is not None:
            assert rank is not None and world_size is not None
            digits = int(math.log10(world_size) + 1)
            message_pre += [f" [{rank:0{digits}d}/{world_size}]"]
        message = "".join(message_pre + message_post)

        super().__init__(message, style="{", datefmt="%Y-%m-%d %H:%M:%S")

        self.rank = rank
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname

        match levelname:
            case "DEBUG":
                record.levelname = ""
            case "INFOALL":
                record.levelname = "INFO"
            case "DEBUGALL":
                record.levelname = "DEBUG"

        if record.levelname and self.use_color and levelname in self.COLORS:
            record.levelname = colored(record.levelname, self.COLORS[levelname], bold=True)
        return logging.Formatter.format(self, record)


def configure_logging(
    prefix: str | None = None,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    debug: bool | None = None,
) -> logging.Logger:
    """Instantiates logging.

    This captures logs and reroutes them to the Toasts module, which is
    pretty similar to Python logging except that the API is a lot easier to
    interact with.

    Args:
        prefix: An optional prefix to add to the logger
        rank: The current rank, or None if not using multiprocessing
        world_size: The total world size, or None if not using multiprocessing
        debug: Whether to enable debug logging
    """
    if rank is not None or world_size is not None:
        assert rank is not None and world_size is not None
    root_logger = logging.getLogger()

    config = load_user_config().logging

    # Captures warnings from the warnings module.
    logging.captureWarnings(True)

    filter = RankFilter(rank=rank)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColoredFormatter(prefix=prefix, rank=rank, world_size=world_size))
    stream_handler.addFilter(filter)
    root_logger.addHandler(stream_handler)

    if debug is None:
        root_logger.setLevel(logging._nameToLevel[config.log_level])
    else:
        root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Avoid junk logs from other libraries.
    if config.hide_third_party_logs:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("jax").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

    return root_logger


def get_unused_port(default: int | None = None) -> int:
    """Returns an unused port number on the local machine.

    Args:
        default: A default port to try before trying other ports.

    Returns:
        A port number which is currently unused
    """
    if default is not None:
        sock = socket.socket()
        try:
            sock.bind(("", default))
            return default
        except OSError:
            pass
        finally:
            sock.close()

    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


def port_is_busy(port: int) -> int:
    """Checks whether a port is busy.

    Args:
        port: The port to check.

    Returns:
        Whether the port is busy.
    """
    sock = socket.socket()
    try:
        sock.bind(("", port))
        return False
    except OSError:
        return True
    finally:
        sock.close()
