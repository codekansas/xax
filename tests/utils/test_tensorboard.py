"""Tests for TensorBoard utility functions."""

import tempfile
from pathlib import Path

import numpy as np
import PIL.Image

from xax.utils.tensorboard import TensorboardWriter, TensorboardWriters, make_histogram


def test_make_histogram_basic() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hist = make_histogram(values, bins="auto")
    assert hist.min == 1.0
    assert hist.max == 5.0
    assert hist.num == 5
    assert hist.sum == 15.0


def test_make_histogram_with_max_bins() -> None:
    values = np.random.randn(1000)
    hist = make_histogram(values, bins="auto", max_bins=10)
    # Histogram should be created successfully (max_bins is approximate)
    assert len(hist.bucket_limit) > 0
    assert hist.num == 1000


def test_tensorboard_writer_add_scalar() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)
        writer.add_scalar("test/loss", 0.5, global_step=1)
        writer.pb_writer.flush()

        # Check that event file was created
        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_scalar_old_style() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)
        writer.add_scalar("test/loss", 0.5, global_step=1, new_style=False)
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_image() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)

        # Create a simple test image
        image = PIL.Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        writer.add_image("test/image", image, global_step=1)
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_video() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)

        # Create a simple test video (T, H, W, C)
        video = np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8)
        writer.add_video("test/video", video, fps=10, global_step=1)
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_audio_mono() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)

        # Create mono audio (T,)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
        writer.add_audio("test/audio", audio, sample_rate=44100, global_step=1)
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_audio_stereo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)

        # Create stereo audio (C, T)
        t = np.linspace(0, 1, 44100)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        audio = np.stack([left, right], axis=0)  # (2, T)
        writer.add_audio("test/audio_stereo", audio, sample_rate=44100, global_step=1)
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_text() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)
        writer.add_text("test/text", "Hello, TensorBoard!", global_step=1)
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_histogram() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)
        values = np.random.randn(1000)
        writer.add_histogram("test/histogram", values, global_step=1)
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writer_add_histogram_raw() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)
        writer.add_histogram_raw(
            "test/histogram_raw",
            min=0.0,
            max=10.0,
            num=100,
            sum=500.0,
            sum_squares=3000.0,
            bucket_limits=[2.0, 4.0, 6.0, 8.0, 10.0],
            bucket_counts=[20, 25, 30, 15, 10],
            global_step=1,
        )
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_tensorboard_writers_light_heavy() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writers = TensorboardWriters(tmpdir)

        # Write to light writer
        light_writer = writers.writer(heavy=False)
        light_writer.add_scalar("train/loss", 0.5, global_step=1)
        light_writer.pb_writer.flush()

        # Write to heavy writer
        heavy_writer = writers.writer(heavy=True)
        heavy_writer.add_scalar("train/accuracy", 0.9, global_step=1)
        heavy_writer.pb_writer.flush()

        # Check both directories were created
        light_dir = Path(tmpdir) / "light"
        heavy_dir = Path(tmpdir) / "heavy"
        assert light_dir.exists()
        assert heavy_dir.exists()
        assert len(list(light_dir.glob("events.out.tfevents.*"))) == 1
        assert len(list(heavy_dir.glob("events.out.tfevents.*"))) == 1


def test_tensorboard_writer_add_mesh() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorboardWriter(tmpdir)

        # Create simple mesh data (triangle) with batch dimension
        # Shape should be (B, N, 3) for batch size B and N vertices
        vertices = np.array([[[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]], dtype=np.float32)  # (1, 3, 3)
        faces = np.array([[[0, 1, 2]]], dtype=np.int32)  # (1, 1, 3)
        colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)  # (1, 3, 3)

        writer.add_mesh(
            "test/mesh",
            vertices=vertices,
            faces=faces,
            colors=colors,
            config_dict=None,
            global_step=1,
        )
        writer.pb_writer.flush()

        event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1
