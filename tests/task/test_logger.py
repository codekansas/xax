"""Runs tests on the logger module."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from PIL import Image
from PIL.Image import Image as PILImage

import xax


class DummyLogger(xax.LoggerImpl):
    def __init__(self) -> None:
        super().__init__()

        self._line: xax.LogLine | None = None

    @property
    def line(self) -> xax.LogLine:
        assert self._line is not None
        return self._line

    def write(self, line: xax.LogLine) -> None:
        self._line = line

    def clear(self) -> None:
        self._line = None

    def should_log(self, state: xax.State, heavy: bool) -> bool:
        return True

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def log_file(self, name: str, contents: str) -> None:
        pass

    def log_error(self, error: xax.LogError) -> None:
        pass

    def log_error_summary(self, summary: xax.LogErrorSummary) -> None:
        pass

    def log_status(self, status: xax.LogStatus) -> None:
        pass

    def log_ping(self, ping: xax.LogPing) -> None:
        pass

    def write_error(self, error: xax.LogError) -> None:
        pass

    def write_error_summary(self, error_summary: xax.LogErrorSummary) -> None:
        pass

    def write_status(self, status: xax.LogStatus) -> None:
        pass

    def write_ping(self, ping: xax.LogPing) -> None:
        pass


@pytest.mark.parametrize(
    "image",
    [
        np.random.random((32, 32, 3)),
        np.random.random((32, 32, 1)),
        np.random.random((3, 32, 32)),
        np.random.random((32, 32)),
        jnp.array(np.random.random((32, 32, 3))),
        jnp.array(np.random.random((1, 32, 32))),
        Image.new("RGB", (32, 32)),
        Image.new("L", (32, 32)),
        np.array(Image.new("L", (32, 32))),
    ],
)
def test_log_image(image: np.ndarray | Array | PILImage) -> None:
    with xax.Logger() as logger:
        dummy_logger = DummyLogger()
        logger.add_logger(dummy_logger)

        # Logs the image.
        logger.log_image("test", image, target_resolution=(32, 32))
        logger.write(xax.State.init_state(), heavy=False)
        image = dummy_logger.line.images["value"]["test"].image
        dummy_logger.clear()
        assert image.size == (32, 32)

        # Logs the image with a caption.
        logger.log_labeled_image("test", (image, "caption\ncaption"), target_resolution=(32, 32))
        logger.write(xax.State.init_state(), heavy=False)
        image = dummy_logger.line.images["value"]["test"].image
        dummy_logger.clear()
        assert image.size > (32, 32)


@pytest.mark.parametrize(
    "images",
    [
        np.random.random((7, 32, 32, 3)),
        np.random.random((7, 32, 32, 1)),
        np.random.random((7, 3, 32, 32)),
        np.random.random((7, 32, 32)),
        jnp.array(np.random.random((7, 32, 32, 3))),
        jnp.array(np.random.random((7, 1, 32, 32))),
        [Image.new("RGB", (32, 32))] * 7,
        [Image.new("L", (32, 32))] * 7,
        np.array(Image.new("L", (32, 32)))[None].repeat(7, axis=0),
    ],
)
def test_log_images(images: np.ndarray | Array | list[PILImage]) -> None:
    with xax.Logger() as logger:
        dummy_logger = DummyLogger()
        logger.add_logger(dummy_logger)

        # Logs the images.
        logger.log_images("test", images, target_resolution=(32, 32), max_images=6)
        logger.write(xax.State.init_state(), heavy=False)
        image = dummy_logger.line.images["value"]["test"].image
        dummy_logger.clear()
        assert np.prod(image.size) == 6 * 32 * 32

        # Logs the images with captions.
        logger.log_labeled_images(
            "test",
            (images, ["caption\ncaption"] * 7),
            target_resolution=(32, 32),
            max_images=6,
        )
        logger.write(xax.State.init_state(), heavy=False)
        image = dummy_logger.line.images["value"]["test"].image
        dummy_logger.clear()
        assert np.prod(image.size) > 6 * 32 * 32
