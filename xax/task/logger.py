"""Defines the core logger.

A common problem when quickly prototyping ML models is nicely logging images,
videos, audio, or other data. Additionally, logging on every step can be
overwhelming. This logger implements a number of convenience functions to
take heterogeneous input data and put it into a standard format, which can
then be used by downstream loggers to actually log the data. For example, this
logger will automatically tile multiple images into a single image, add
captions to images, and so on.
"""

import functools
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from types import TracebackType
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Self,
    Sequence,
    TypeVar,
    cast,
)

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import ClosedJaxpr
from jaxtyping import Array
from PIL import Image as PILImage, ImageDraw, ImageFont
from PIL.Image import Image as PILImageType

from xax.core.state import State
from xax.utils.experiments import ContextTimer, IntervalTicker
from xax.utils.logging import LOG_ERROR_SUMMARY, LOG_PING, LOG_STATUS, format_number

logger = logging.getLogger(__name__)

T = TypeVar("T")
LogT = TypeVar("LogT")
Number = int | float | Array | np.ndarray

ChannelSelectMode = Literal["first", "last", "mean"]

DEFAULT_NAMESPACE = "value"
DEFAULT_IMAGE_RESOLUTION = (512, 512)
DEFAULT_IMAGES_RESOLUTION = (256, 256)
DEFAULT_VIDEO_FPS = 30
DEFAULT_HISTOGRAM_BINS = 100

NAMESPACE_STACK: list[str] = []


def standardize_text(text: str, max_line_length: int | None = None, remove_non_ascii: bool = False) -> list[str]:
    """Standardizes a text string to a list of lines.

    Args:
        text: The text to standardize
        max_line_length: If set, truncate lines to this length
        remove_non_ascii: Remove non-ASCII characters if present

    Returns:
        The standardized text lines
    """

    def _chunk_lines(text: str, max_length: int) -> Iterator[str]:
        for i in range(0, len(text), max_length):
            yield text[i : i + max_length]

    if remove_non_ascii:
        text = "".join(char for char in text if ord(char) < 128)
    lines = [re.sub(r"\s+", " ", line) for line in re.split(r"[\n\r]+", text.strip())]
    if max_line_length is not None:
        lines = [subline for line in lines for subline in _chunk_lines(line, max_line_length)]
    return lines


def make_human_viewable_resolution(
    image: PILImageType,
    interpolation: PILImage.Resampling = PILImage.Resampling.LANCZOS,
    trg_res: tuple[int, int] = (512, 512),
) -> PILImageType:
    """Resizes image to human-viewable resolution.

    Args:
        image: The image to resize, with shape (C, H, W)
        interpolation: Interpolation mode to use for image resizing
        trg_res: The target image resolution; the image will be reshaped to
            have approximately the same area as an image with this resolution

    Returns:
        The resized image
    """
    width, height = image.size
    trg_height, trg_width = trg_res
    factor = math.sqrt((trg_height * trg_width) / (height * width))
    new_height, new_width = int(height * factor), int(width * factor)
    return image.resize((new_width, new_height), interpolation)


class namespace_context:  # noqa: N801
    def __init__(self, name: str | None) -> None:
        self._name = name
        self._prev_stack: list[str] | None = None

    def __enter__(self) -> None:
        if self._name is None:
            self._prev_stack = NAMESPACE_STACK[:]
            NAMESPACE_STACK.clear()
        else:
            NAMESPACE_STACK.append(self._name)

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        if self._prev_stack is not None:
            NAMESPACE_STACK[:] = self._prev_stack
        else:
            NAMESPACE_STACK.pop()


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


def ternary_search_optimal_side_counts(height: int, width: int, count: int) -> tuple[int, int]:
    min_factors = [i for i in range(1, math.ceil(math.sqrt(count)) + 1) if count % i == 0]
    max_factors = [i for i in min_factors[::-1] if i * i != count]
    factors = [(i, count // i) for i in min_factors] + [(count // i, i) for i in max_factors]

    lo, hi = 0, len(factors) - 1

    def penalty(i: int) -> float:
        hval, wval = factors[i]
        h, w = hval * height, wval * width
        return -(min(h, w) ** 2)

    # Runs ternary search to minimize penalty.
    while lo < hi - 2:
        lmid, rmid = (lo * 2 + hi) // 3, (lo + hi * 2) // 3
        if penalty(lmid) > penalty(rmid):
            lo = lmid
        else:
            hi = rmid

    # Returns the lowest-penalty configuration.
    mid = (lo + hi) // 2
    plo, pmid, phi = penalty(lo), penalty(mid), penalty(hi)

    if pmid <= plo and pmid <= phi:
        return factors[mid]
    elif plo <= phi:
        return factors[lo]
    else:
        return factors[hi]


def tile_images_different_sizes(images: list[PILImageType], sep: int) -> PILImageType:
    """Tiles a list of images into a single image, even if they have different sizes.

    Args:
        images: The images to tile.
        sep: The separation between adjacent images.

    Returns:
        The tiled image.
    """
    total_width, max_height = sum(image.width for image in images), max(image.height for image in images)
    tiled = PILImage.new("RGB", (total_width + (len(images) - 1) * sep, max_height))
    x = 0
    for image in images:
        tiled.paste(image, (x, 0))
        x += image.width + sep
    return tiled


def tile_images(images: list[PILImageType], sep: int = 0) -> PILImageType:
    """Tiles a list of images into a single image.

    Args:
        images: The images to tile.
        sep: The separation between adjacent images.

    Returns:
        The tiled image.
    """
    if not images:
        return PILImage.new("RGB", (0, 0))

    # Gets the optimal side counts.
    height, width = images[0].height, images[0].width
    if not all(image.size == images[0].size for image in images):
        return tile_images_different_sizes(images, sep)

    hside, wside = ternary_search_optimal_side_counts(height, width, len(images))

    # Tiles the images.
    tiled = PILImage.new("RGB", (wside * width + (wside - 1) * sep, hside * height + (hside - 1) * sep))
    for i, image in enumerate(images):
        x, y = i % wside, i // wside
        tiled.paste(image, (x * (width + sep), y * (height + sep)))

    return tiled


def as_numpy(array: Array | np.ndarray) -> np.ndarray:
    """Convert a JAX array or numpy array to numpy array."""
    if isinstance(array, np.ndarray):
        return array
    array = jax.device_get(array)
    if jax.dtypes.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.float32)
    elif jax.dtypes.issubdtype(array.dtype, jnp.integer):
        array = array.astype(jnp.int32)
    elif jax.dtypes.issubdtype(array.dtype, jnp.bool_):
        array = array.astype(jnp.bool_)
    return np.array(array)


def as_numpy_opt(array: Array | np.ndarray | None) -> np.ndarray | None:
    """Convert an optional JAX array or numpy array to numpy array."""
    if array is None:
        return None
    return as_numpy(array)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Scalar:
    value: Array
    secondary: bool = field(default=False)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Distribution:
    mean: Array
    std: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Histogram:
    value: Array
    bins: int = field(default=DEFAULT_HISTOGRAM_BINS)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Tokens:
    value: Array
    secondary: bool = field(default=False)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Image:
    image: Array
    target_resolution: tuple[int, int] = field(default=DEFAULT_IMAGE_RESOLUTION)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Images:
    images: Array
    max_images: int | None = field(default=None)
    target_resolution: tuple[int, int] = field(default=DEFAULT_IMAGES_RESOLUTION)
    sep: int = field(default=0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LabeledImages:
    images: Array
    labels: Array
    max_images: int | None = field(default=None)
    max_line_length: int | None = field(default=None)
    max_num_lines: int | None = field(default=None)
    target_resolution: tuple[int, int] = field(default=DEFAULT_IMAGE_RESOLUTION)
    line_spacing: int = field(default=2)
    centered: bool = field(default=True)
    sep: int = field(default=0)

    def __post_init__(self) -> None:
        chex.assert_shape(self.labels, (self.images.shape[0],))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Video:
    video: Array
    fps: int = field(default=DEFAULT_VIDEO_FPS)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Mesh:
    vertices: Array
    colors: Array | None = field(default=None)
    faces: Array | None = field(default=None)
    config_dict: dict[str, Any] | None = field(default=None)


Metric = Scalar | Distribution | Histogram | Tokens | Image | Images | LabeledImages | Video | Mesh


@dataclass(kw_only=True)
class LogString:
    value: str
    secondary: bool


@dataclass(kw_only=True)
class LogImage:
    image: PILImageType


@dataclass(kw_only=True)
class LogVideo:
    """Container for video data and metadata.

    Attributes:
        frames: Video frames as a numpy array of shape (T,H,W,C)
        fps: Frames per second
    """

    frames: np.ndarray
    fps: int


@dataclass(kw_only=True)
class LogScalar:
    value: Number
    secondary: bool


@dataclass(kw_only=True)
class LogDistribution:
    mean: Number
    std: Number


@dataclass(kw_only=True)
class LogHistogram:
    min: Number
    max: Number
    num: int
    sum: Number
    sum_squares: Number
    bucket_limits: list[float]
    bucket_counts: list[int]


@dataclass(kw_only=True)
class LogMesh:
    vertices: np.ndarray
    colors: np.ndarray | None
    faces: np.ndarray | None
    config_dict: dict[str, Any] | None  # noqa: ANN401


@dataclass(kw_only=True)
class LogGraph:
    computation: ClosedJaxpr


@dataclass(kw_only=True)
class LogLine:
    state: State
    heavy: bool
    scalars: dict[str, dict[str, LogScalar]]
    distributions: dict[str, dict[str, LogDistribution]]
    histograms: dict[str, dict[str, LogHistogram]]
    strings: dict[str, dict[str, LogString]]
    images: dict[str, dict[str, LogImage]]
    videos: dict[str, dict[str, LogVideo]]
    meshes: dict[str, dict[str, LogMesh]]


@dataclass(kw_only=True)
class LogErrorSummary:
    message: str


@dataclass(kw_only=True)
class LogError:
    message: str
    location: str | None = None

    @property
    def message_with_location(self) -> str:
        message = self.message
        if self.location is not None:
            message += f" ({self.location})"
        return message


@dataclass(kw_only=True)
class LogStatus:
    message: str
    created: float
    filename: str | None = None
    lineno: int | None = None


@dataclass(kw_only=True)
class LogPing:
    message: str
    created: float
    filename: str | None = None
    lineno: int | None = None


def get_image(image: np.ndarray | Array | PILImageType, target_resolution: tuple[int, int] | None = None) -> LogImage:
    if not isinstance(image, (np.ndarray, Array, PILImageType)):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if isinstance(image, Array):
        image = as_numpy(image)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.ndim != 3:
            raise RuntimeError(f"Expected image to have shape HW, HWC, or CHW, got {image.shape}")

        # Normalizes the image and converts to integer.
        if np.issubdtype(image.dtype, np.floating):
            image = (normalize(image) * 255).round().astype(np.uint8)
        elif image.dtype == np.uint8:
            pass
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        # Converts to a PIL image.
        if image.shape[-1] == 1:
            image = PILImage.fromarray(image[..., 0])
        elif image.shape[-1] == 3:
            image = PILImage.fromarray(image)
        elif image.shape[0] == 1:
            image = PILImage.fromarray(image[0])
        elif image.shape[0] == 3:
            image = PILImage.fromarray(image.transpose(1, 2, 0))
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    if target_resolution is not None:
        image = make_human_viewable_resolution(image, trg_res=target_resolution)
    return LogImage(image=image)


def image_with_text(
    image: PILImageType,
    text: list[str],
    max_num_lines: int | None,
    line_spacing: int,
    centered: bool,
) -> LogImage:
    """Adds a text label to an image.

    Args:
        image: The image to label, with shape (C, H, W)
        text: The text label for the image
        max_num_lines: The number of lines of spacing to add to the bottom
            of the image
        line_spacing: The spacing between adjacent lines
        centered: If set, center the text labels, otherwise align to the left

    Returns:
        The image with a text label
    """
    if not text:
        return LogImage(image=image)
    if max_num_lines is None:
        max_num_lines = len(text)
    else:
        text = text[:max_num_lines]
    width, height = image.size
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont = ImageFont.load_default()
    _, _, _, line_height = font.getbbox(text[0])
    new_width, new_height = width, int(height + line_spacing + max_num_lines * (line_height + line_spacing))
    padded_image = PILImage.new(image.mode, (new_width, new_height), 255)
    padded_image.paste(image, (0, 0))
    drawer = ImageDraw.Draw(padded_image)
    for i, text_line in enumerate(text):
        text_line_top = height + line_spacing + i * (line_height + line_spacing)
        if centered:
            _, _, line_width, _ = font.getbbox(text_line)
            text_line_left = (width - line_width) / 2
            drawer.text((text_line_left, text_line_top), text_line, font=font, fill=0)
        else:
            drawer.text((line_spacing, text_line_top), text_line, font=font, fill=0)
    return LogImage(image=padded_image)


def get_video(video: np.ndarray | Array, fps: int = 30) -> LogVideo:
    """Converts video data to standard format.

    Args:
        video: The video frames. Can be:
            - A numpy array of shape (T, H, W, C) or (T, C, H, W)
            - A JAX array of shape (T, H, W, C) or (T, C, H, W)
        fps: Frames per second

    Returns:
        LogVideo containing standardized video frames
    """
    if isinstance(video, Array):
        video = as_numpy(video)

    if not isinstance(video, np.ndarray):
        raise ValueError(f"Unsupported video type: {type(video)}")

    # Handle different dimension orderings
    if video.ndim != 4:
        raise ValueError(f"Expected video array of shape (T, H, W, C) or (T, C, H, W), got shape {video.shape}")

    if video.shape[1] == 3:  # (T,C,H,W) format
        video = video.transpose(0, 2, 3, 1)

    # Normalize and convert to uint8 if needed
    if np.issubdtype(video.dtype, np.floating):
        video = (normalize(video) * 255).round().astype(np.uint8)
    elif video.dtype != np.uint8:
        raise ValueError(f"Unsupported video dtype: {video.dtype}")

    return LogVideo(frames=video, fps=fps)


class LoggerImpl(ABC):
    def __init__(self, log_interval_seconds: float = 1.0) -> None:
        """Defines some default behavior for loggers.

        Every logger needs to implement the ``write`` function, which handles
        actually writing the logs to wherever they needs to go. The basic
        class implements a simple interval-based logging scheme to avoid
        writing too many log lines.

        Args:
            log_interval_seconds: The interval between successive log lines.
        """
        super().__init__()

        self.ticker = {k: IntervalTicker(log_interval_seconds) for k in [True, False]}

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def write(self, line: LogLine) -> None:
        """Handles writing the current log line.

        Args:
            line: The line to write.
        """

    @abstractmethod
    def write_error_summary(self, error_summary: LogErrorSummary) -> None:
        """Handles writing an error summary.

        Args:
            error_summary: The error summary to write.
        """

    @abstractmethod
    def write_error(self, error: LogError) -> None:
        """Handles writing an error line.

        Args:
            error: The error information to write.
        """

    @abstractmethod
    def write_status(self, status: LogStatus) -> None:
        """Handles writing a status line.

        Args:
            status: The status to write.
        """

    @abstractmethod
    def write_ping(self, ping: LogPing) -> None:
        """Handles writing a ping line.

        Args:
            ping: The ping to write.
        """

    @abstractmethod
    def log_file(self, name: str, contents: str) -> None:
        """Logs a large text file.

        Args:
            name: The name of the file.
            contents: The contents of the file.
        """

    def should_log(self, state: State, heavy: bool) -> bool:
        """Function that determines if the logger should log the current step.

        Args:
            state: The current step's state.
            heavy: If the current step is heavy.

        Returns:
            If the logger should log the current step.
        """
        elapsed_time = state.elapsed_time_s.item()
        should_log = self.ticker[heavy].tick(elapsed_time)
        return should_log


class ToastHandler(logging.Handler):
    def __init__(self, logger: "Logger") -> None:
        super().__init__()

        self.logger = logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno == LOG_ERROR_SUMMARY:
                self.logger.write_error_summary(record.getMessage())
            elif record.levelno == LOG_STATUS:
                self.logger.write_status(record.getMessage(), record.filename, record.lineno)
            elif record.levelno in (LOG_PING, logging.WARNING):
                self.logger.write_ping(record.getMessage(), record.filename, record.lineno)
            elif record.levelno in (logging.ERROR, logging.CRITICAL, logging.WARNING):
                self.logger.write_error(record.getMessage(), f"{record.filename}:{record.lineno}")
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def add_for_logger(self, logger: logging.Logger) -> None:
        # Removes existing ToastHandler.
        handlers_to_remove = []
        for handler in logger.handlers:
            if isinstance(handler, ToastHandler):
                handlers_to_remove.append(handler)
        for handler in handlers_to_remove:
            logger.removeHandler(handler)

        # Adds the new ToastHandler.
        logger.addHandler(self)


class Logger:
    """Defines an intermediate container which holds values to log somewhere else."""

    def __init__(self, default_namespace: str = DEFAULT_NAMESPACE) -> None:
        self.scalars: dict[str, dict[str, Callable[[], LogScalar]]] = defaultdict(dict)
        self.distributions: dict[str, dict[str, Callable[[], LogDistribution]]] = defaultdict(dict)
        self.histograms: dict[str, dict[str, Callable[[], LogHistogram]]] = defaultdict(dict)
        self.strings: dict[str, dict[str, Callable[[], LogString]]] = defaultdict(dict)
        self.images: dict[str, dict[str, Callable[[], LogImage]]] = defaultdict(dict)
        self.videos: dict[str, dict[str, Callable[[], LogVideo]]] = defaultdict(dict)
        self.meshes: dict[str, dict[str, Callable[[], LogMesh]]] = defaultdict(dict)
        self.default_namespace = default_namespace
        self.loggers: list[LoggerImpl] = []

        # Registers a logging handler to route log messages to the logger.
        root_logger = logging.getLogger()
        ToastHandler(self).add_for_logger(root_logger)

        # Flag when the logger is active.
        self.active = False

    def add_logger(self, *logger: LoggerImpl) -> None:
        """Add the logger, so that it gets called when `write` is called.

        Args:
            logger: The logger to add.
        """
        self.loggers.extend(logger)

    def pack(self, state: State, heavy: bool) -> LogLine:
        return LogLine(
            state=state,
            heavy=heavy,
            scalars={k: {kk: v() for kk, v in v.items()} for k, v in self.scalars.items()},
            distributions={k: {kk: v() for kk, v in v.items()} for k, v in self.distributions.items()},
            histograms={k: {kk: v() for kk, v in v.items()} for k, v in self.histograms.items()},
            strings={k: {kk: v() for kk, v in v.items()} for k, v in self.strings.items()},
            images={k: {kk: v() for kk, v in v.items()} for k, v in self.images.items()},
            videos={k: {kk: v() for kk, v in v.items()} for k, v in self.videos.items()},
            meshes={k: {kk: v() for kk, v in v.items()} for k, v in self.meshes.items()},
        )

    def clear(self) -> None:
        self.scalars.clear()
        self.distributions.clear()
        self.histograms.clear()
        self.strings.clear()
        self.images.clear()
        self.videos.clear()
        self.meshes.clear()

    def write(self, state: State, heavy: bool) -> None:
        """Writes the current step's logging information.

        Args:
            state: The current step's state.
            heavy: If the current step is heavy.
        """
        should_log = [lg.should_log(state, heavy) for lg in self.loggers]
        if not any(should_log):
            self.clear()
            return
        line = self.pack(state, heavy)
        self.clear()
        for lg in (lg for lg, should_log in zip(self.loggers, should_log, strict=False) if should_log):
            lg.write(line)

    def write_error_summary(self, error_summary: str) -> None:
        for logger in self.loggers:
            logger.write_error_summary(LogErrorSummary(message=error_summary))

    def write_error(self, message: str, location: str | None = None) -> None:
        for logger in self.loggers:
            logger.write_error(LogError(message=message, location=location))

    def write_status(
        self,
        message: str,
        filename: str | None = None,
        lineno: int | None = None,
        created: float | None = None,
    ) -> None:
        status = LogStatus(
            message=message,
            created=time.time() if created is None else created,
            filename=filename,
            lineno=lineno,
        )
        for logger in self.loggers:
            logger.write_status(status)

    def write_ping(
        self,
        message: str,
        filename: str | None = None,
        lineno: int | None = None,
        created: float | None = None,
    ) -> None:
        ping = LogPing(
            message=message,
            created=time.time() if created is None else created,
            filename=filename,
            lineno=lineno,
        )
        for logger in self.loggers:
            logger.write_ping(ping)

    def resolve_namespace(self, namespace: str | None = None) -> str:
        return "_".join([self.default_namespace if namespace is None else namespace] + NAMESPACE_STACK)

    def log_metric(
        self,
        key: str,
        value: Metric,
        *,
        namespace: str | None = None,
        decode_tokens: Callable[[Array], str] | None = None,
    ) -> None:
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        if isinstance(value, Scalar):
            self.log_scalar(key, value.value, secondary=value.secondary, namespace=namespace)
        elif isinstance(value, Distribution):
            self.log_distribution(key, (value.mean, value.std), namespace=namespace)
        elif isinstance(value, Histogram):
            self.log_histogram(key, value.value, bins=value.bins, namespace=namespace)
        elif isinstance(value, Tokens):
            if decode_tokens is None:
                raise ValueError("decode_tokens must be provided when logging Tokens")
            value_str = decode_tokens(value.value)
            self.log_string(key, value_str, namespace=namespace, secondary=value.secondary)
        elif isinstance(value, Image):
            self.log_image(key, value.image, namespace=namespace, target_resolution=value.target_resolution)
        elif isinstance(value, Images):
            self.log_images(
                key,
                value.images,
                namespace=namespace,
                target_resolution=value.target_resolution,
                sep=value.sep,
            )
        elif isinstance(value, LabeledImages):
            self.log_labeled_images(
                key,
                (value.images, [format_number(label) for label in value.labels.tolist()]),
                namespace=namespace,
                max_images=value.max_images,
                max_line_length=value.max_line_length,
                max_num_lines=value.max_num_lines,
                target_resolution=value.target_resolution,
                line_spacing=value.line_spacing,
                centered=value.centered,
                sep=value.sep,
            )
        elif isinstance(value, Video):
            self.log_video(key, value.video, namespace=namespace, fps=value.fps)
        elif isinstance(value, Mesh):
            self.log_mesh(
                key,
                value.vertices,
                colors=value.colors,
                faces=value.faces,
                config_dict=value.config_dict,
                namespace=namespace,
            )
        else:
            raise ValueError(f"Unsupported log type: {type(value)}")

    def log_scalar(
        self,
        key: str,
        value: Callable[[], Number] | Number,
        *,
        namespace: str | None = None,
        secondary: bool = False,
    ) -> None:
        """Logs a scalar value.

        Args:
            key: The key being logged
            value: The scalar value being logged
            namespace: An optional logging namespace
            secondary: If set, treat this as a secondary value (meaning, it is
                less important than other values, and some downstream loggers
                will not display it)
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        if isinstance(value, jnp.ndarray):
            assert value.ndim == 0, f"Scalar must be a 0D array, got shape {value.shape}"

        @functools.lru_cache(maxsize=None)
        def scalar_future() -> LogScalar:
            with ContextTimer() as timer:
                value_concrete = value() if callable(value) else value
            logger.debug("Scalar Key: %s, Time: %s", key, timer.elapsed_time)
            return LogScalar(value=value_concrete, secondary=secondary)

        self.scalars[namespace][key] = scalar_future

    def log_distribution(
        self,
        key: str,
        value: Callable[[], tuple[Number, Number]] | tuple[Number, Number],
        *,
        namespace: str | None = None,
    ) -> None:
        """Logs a distribution value.

        Args:
            key: The key being logged
            value: The distribution value being logged, a tuple of (mean, std)
            namespace: An optional logging namespace
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def distribution_future() -> LogDistribution:
            with ContextTimer() as timer:
                value_concrete = value() if callable(value) else value
            logger.debug("Distribution Key: %s, Time: %s", key, timer.elapsed_time)
            mean, std = value_concrete
            return LogDistribution(mean=mean, std=std)

        self.distributions[namespace][key] = distribution_future

    def log_histogram(
        self,
        key: str,
        value: Callable[[], np.ndarray | Array] | np.ndarray | Array,
        *,
        bins: int = 100,
        namespace: str | None = None,
    ) -> None:
        """Logs a histogram value.

        Args:
            key: The key being logged
            value: The histogram value being logged
            bins: The number of bins to use for the histogram
            namespace: An optional logging namespace
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def histogram_future() -> LogHistogram:
            with ContextTimer() as timer:
                values = value() if callable(value) else value
                values = values.reshape(-1)  # Must be flat.

                if isinstance(values, Array):
                    counts, limits = jnp.histogram(values, bins=bins)
                    counts, limits = as_numpy(counts), as_numpy(limits)
                elif isinstance(values, np.ndarray):
                    counts, limits = np.histogram(values, bins=bins)
                else:
                    raise ValueError(f"Unsupported histogram type: {type(values)}")

                histogram_values = LogHistogram(
                    min=float(values.min()),
                    max=float(values.max()),
                    num=int(values.size),
                    sum=float(values.sum()),
                    sum_squares=float(values.dot(values)),
                    bucket_limits=cast(list[float], limits[1:].tolist()),
                    bucket_counts=cast(list[int], counts.tolist()),
                )

            logger.debug("Histogram Key: %s, Time: %s", key, timer.elapsed_time)
            return histogram_values

        self.histograms[namespace][key] = histogram_future

    def log_histogram_raw(
        self,
        key: str,
        counts: Array | np.ndarray,
        limits: Array | np.ndarray,
        minv: Number | None = None,
        maxv: Number | None = None,
        sumv: Number | None = None,
        sum_squaresv: Number | None = None,
        *,
        namespace: str | None = None,
    ) -> None:
        """Logs a histogram from raw counts and limits.

        Args:
            key: The key being logged
            counts: The counts of the histogram
            limits: The limits of the histogram
            minv: The minimum value of the histogram
            maxv: The maximum value of the histogram
            sumv: The sum of the histogram
            sum_squaresv: The sum of the squares of the histogram
            namespace: An optional logging namespace
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def histogram_future() -> LogHistogram:
            with ContextTimer() as timer:
                counts_np = (as_numpy(counts) if isinstance(counts, Array) else counts).astype(int)
                limits_np = (as_numpy(limits) if isinstance(limits, Array) else limits).astype(float)

                minv_ = counts_np.min() if minv is None else minv
                maxv_ = counts_np.max() if maxv is None else maxv
                sumv_ = counts_np.sum() if sumv is None else sumv
                sum_squaresv_ = counts_np.dot(counts_np) if sum_squaresv is None else sum_squaresv

                histogram_values = LogHistogram(
                    min=float(minv_),
                    max=float(maxv_),
                    num=int(counts_np.size),
                    sum=float(sumv_),
                    sum_squares=float(sum_squaresv_),
                    bucket_limits=cast(list[float], limits_np.tolist()),
                    bucket_counts=cast(list[int], counts_np.tolist()),
                )

            logger.debug("Raw Histogram Key: %s, Time: %s", key, timer.elapsed_time)
            return histogram_values

        self.histograms[namespace][key] = histogram_future

    def log_string(
        self,
        key: str,
        value: Callable[[], str] | str,
        *,
        namespace: str | None = None,
        secondary: bool = False,
    ) -> None:
        """Logs a string value.

        Args:
            key: The key being logged
            value: The string value being logged
            namespace: An optional logging namespace
            secondary: If set, treat this as a secondary value (meaning, it is
                less important than other values, and some downstream loggers
                will not display it)
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def value_future() -> LogString:
            value_concrete = value() if callable(value) else value
            return LogString(value=value_concrete, secondary=secondary)

        self.strings[namespace][key] = value_future

    def log_image(
        self,
        key: str,
        value: Callable[[], np.ndarray | Array | PILImageType] | np.ndarray | Array | PILImageType,
        *,
        namespace: str | None = None,
        target_resolution: tuple[int, int] | None = DEFAULT_IMAGE_RESOLUTION,
    ) -> None:
        """Logs an image.

        Args:
            key: The key being logged
            value: The image being logged
            namespace: An optional logging namespace
            target_resolution: The target resolution for each image; if None,
                don't resample the images
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def image_future() -> LogImage:
            with ContextTimer() as timer:
                value_concrete = value() if callable(value) else value
                try:
                    image = get_image(value_concrete, target_resolution)
                except Exception as e:
                    raise ValueError(f"Error with image {namespace}/{key}") from e
            logger.debug("Image Key: %s, Time: %s", key, timer.elapsed_time)
            return image

        self.images[namespace][key] = image_future

    def log_labeled_image(
        self,
        key: str,
        value: (
            Callable[[], tuple[np.ndarray | Array | PILImageType, str]] | tuple[np.ndarray | Array | PILImageType, str]
        ),
        *,
        namespace: str | None = None,
        max_line_length: int | None = None,
        max_num_lines: int | None = None,
        target_resolution: tuple[int, int] | None = DEFAULT_IMAGE_RESOLUTION,
        line_spacing: int = 2,
        centered: bool = True,
    ) -> None:
        """Logs an image with a label.

        Args:
            key: The key being logged
            value: The image and label being logged
            namespace: An optional logging namespace
            max_line_length: The maximum line length for the label
            max_num_lines: The number of lines of spacing to add to the bottom
                of the image
            target_resolution: The target resolution for each image; if None,
                don't resample the images
            line_spacing: The spacing between adjacent lines
            centered: If set, center the text labels, otherwise align to the left
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def image_future() -> LogImage:
            with ContextTimer() as timer:
                image, label = value() if callable(value) else value
                try:
                    image_concrete = get_image(image, target_resolution)
                except Exception as e:
                    raise ValueError(f"Error with image {namespace}/{key}") from e

                image_value = image_with_text(
                    image_concrete.image,
                    standardize_text(label, max_line_length),
                    max_num_lines=max_num_lines,
                    line_spacing=line_spacing,
                    centered=centered,
                )

            logger.debug("Labeled Image Key: %s, Time: %s", key, timer.elapsed_time)
            return image_value

        self.images[namespace][key] = image_future

    def log_images(
        self,
        key: str,
        value: (
            Callable[[], Sequence[np.ndarray | Array | PILImageType] | np.ndarray | Array]
            | Sequence[np.ndarray | Array | PILImageType]
            | np.ndarray
            | Array
        ),
        *,
        namespace: str | None = None,
        max_images: int | None = None,
        target_resolution: tuple[int, int] | None = DEFAULT_IMAGES_RESOLUTION,
        sep: int = 0,
    ) -> None:
        """Logs a set of images.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images being logged
            namespace: An optional logging namespace
            max_images: The maximum number of images to show; extra images
                are clipped
            target_resolution: The target resolution for each image; if None,
                don't resample the images
            sep: An optional separation amount between adjacent images
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def images_future() -> LogImage:
            with ContextTimer() as timer:
                images = value() if callable(value) else value
                if max_images is not None:
                    images = images[:max_images]
                if isinstance(images, Array):
                    images = as_numpy(images)
                if isinstance(images, Sequence):
                    images = list(images)
                try:
                    images_concrete = [get_image(image, target_resolution) for image in images]
                except Exception as e:
                    raise ValueError(f"Error with image {namespace}/{key}") from e
                tiled = tile_images([img.image for img in images_concrete], sep)

            logger.debug("Images Key: %s, Time: %s", key, timer.elapsed_time)
            return LogImage(image=tiled)

        self.images[namespace][key] = images_future

    def log_labeled_images(
        self,
        key: str,
        value: (
            Callable[[], tuple[Sequence[np.ndarray | Array | PILImageType] | np.ndarray | Array, Sequence[str]]]
            | tuple[Sequence[np.ndarray | Array | PILImageType] | np.ndarray | Array, Sequence[str]]
        ),
        *,
        namespace: str | None = None,
        max_images: int | None = None,
        max_line_length: int | None = None,
        max_num_lines: int | None = None,
        target_resolution: tuple[int, int] | None = (256, 256),
        line_spacing: int = 2,
        centered: bool = True,
        sep: int = 0,
    ) -> None:
        """Logs a set of images with labels.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images and labels being logged
            namespace: An optional logging namespace
            max_images: The maximum number of images to show; extra images
                are clipped
            max_line_length: The maximum line length for the label
            max_num_lines: The number of lines of spacing to add to the bottom
                of the image
            target_resolution: The target resolution for each image; if None,
                don't resample the images
            line_spacing: The spacing between adjacent lines
            centered: If set, center the text labels, otherwise align to the left
            sep: An optional separation amount between adjacent images
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def images_future() -> LogImage:
            with ContextTimer() as timer:
                images, labels = value() if callable(value) else value
                if max_images is not None:
                    images = images[:max_images]
                    labels = labels[:max_images]
                try:
                    images_concrete = [get_image(image, target_resolution) for image in images]
                    labeled = [
                        image_with_text(
                            img.image,
                            standardize_text(label, max_line_length),
                            max_num_lines=max_num_lines,
                            line_spacing=line_spacing,
                            centered=centered,
                        )
                        for img, label in zip(images_concrete, labels, strict=True)
                    ]
                    tiled = tile_images([img.image for img in labeled], sep)
                except Exception as e:
                    raise ValueError(f"Error with image {namespace}/{key}") from e

            logger.debug("Labeled Images Key: %s, Time: %s", key, timer.elapsed_time)
            return LogImage(image=tiled)

        self.images[namespace][key] = images_future

    def log_video(
        self,
        key: str,
        value: Callable[[], np.ndarray | Array] | np.ndarray | Array,
        *,
        fps: int = DEFAULT_VIDEO_FPS,
        namespace: str | None = None,
    ) -> None:
        """Logs a video.

        Args:
            key: The key being logged
            value: The video frames. Can be:
                - A numpy array of shape (T,H,W,C) or (T,C,H,W)
                - A JAX array of shape (T,H,W,C) or (T,C,H,W)
            fps: Frames per second
            namespace: An optional logging namespace
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def video_future() -> LogVideo:
            with ContextTimer() as timer:
                video = get_video(value() if callable(value) else value, fps=fps)

            logger.debug("Video Key: %s, Time: %s", key, timer.elapsed_time)
            return video

        self.videos[namespace][key] = video_future

    def log_mesh(
        self,
        key: str,
        vertices: np.ndarray | Array | Callable[[], np.ndarray | Array],
        colors: np.ndarray | Array | None | Callable[[], np.ndarray | Array | None] = None,
        faces: np.ndarray | Array | None | Callable[[], np.ndarray | Array | None] = None,
        config_dict: dict[str, Any] | None = None,
        *,
        namespace: str | None = None,
    ) -> None:
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def mesh_future() -> LogMesh:
            with ContextTimer() as timer:
                # Get the raw values
                vertices_val = vertices() if callable(vertices) else vertices
                colors_val = colors() if callable(colors) else colors
                faces_val = faces() if callable(faces) else faces

                # Convert to numpy arrays with proper type handling
                vertices_np = as_numpy(vertices_val)
                colors_np = as_numpy_opt(colors_val)
                faces_np = as_numpy_opt(faces_val)

                # Checks vertices shape.
                if vertices_np.ndim == 2:
                    vertices_np = vertices_np[None]
                if vertices_np.shape[-1] != 3 or vertices_np.ndim != 3:
                    raise ValueError("Vertices must have shape (N, 3) or (B, N, 3)")

                # Checks colors shape.
                if colors_np is not None:
                    if colors_np.ndim == 2:
                        colors_np = colors_np[None]
                    if colors_np.shape[-1] != 3 or colors_np.ndim != 3:
                        raise ValueError("Colors must have shape (N, 3) or (B, N, 3)")

                # Checks faces shape.
                if faces_np is not None:
                    if faces_np.ndim == 2:
                        faces_np = faces_np[None]
                    if faces_np.shape[-1] != 3 or faces_np.ndim != 3:
                        raise ValueError("Faces must have shape (N, 3) or (B, N, 3)")

                # Ensures colors dtype is uint8.
                if colors_np is not None:
                    if colors_np.dtype != np.uint8:
                        colors_np = (colors_np * 255).astype(np.uint8)

                # Ensures faces dtype is int32.
                if faces_np is not None:
                    if faces_np.dtype != np.int32:
                        faces_np = faces_np.astype(np.int32)

            logger.debug("Mesh Key: %s, Time: %s", key, timer.elapsed_time)
            return LogMesh(
                vertices=vertices_np,
                colors=colors_np,
                faces=faces_np,
                config_dict=config_dict,
            )

        self.meshes[namespace][key] = mesh_future

    def log_file(self, name: str, contents: str) -> None:
        for logger in self.loggers:
            logger.log_file(name, contents)

    def __enter__(self) -> Self:
        self.active = True
        for logger in self.loggers:
            logger.start()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        for logger in self.loggers:
            logger.stop()
        self.active = False
