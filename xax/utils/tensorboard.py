"""Defines utility functions for interfacing with Tensorboard."""

import functools
import io
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import PIL.Image
from PIL.Image import Image as PILImage
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.event_pb2 import Event, TaggedRunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.summary_pb2 import (
    HistogramProto,
    Summary,
    SummaryMetadata,
)
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.mesh import metadata as mesh_metadata
from tensorboard.plugins.mesh.plugin_data_pb2 import MeshPluginData
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData
from tensorboard.summary.writer.event_file_writer import EventFileWriter

ImageShape = Literal["HWC", "CHW", "HW", "NHWC", "NCHW", "NHW"]


def make_histogram(values: np.ndarray, bins: str | np.ndarray, max_bins: int | None = None) -> "HistogramProto":  # type: ignore[valid-type]
    """Convert values into a histogram proto using logic from histogram.cc.

    Args:
        values: Input values to create histogram from
        bins: Bin specification (string like 'auto' or array of bin edges)
        max_bins: Optional maximum number of bins

    Returns:
        HistogramProto containing the histogram data
    """
    if values.size == 0:
        raise ValueError("The input has no element.")
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)

    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(
                counts,
                pad_width=[[0, subsampling - subsampling_remainder]],
                mode="constant",
                constant_values=0,
            )
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:
    cum_counts = np.cumsum(np.greater(counts, 0))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1

    # TensorBoard only includes the right bin limits. To still have the leftmost limit
    # included, we include an empty bin left.
    counts = counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start : end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError("The histogram is empty, please file a bug report.")

    sum_sq = values.dot(values)
    return HistogramProto(
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limit=limits.tolist(),
        bucket=counts.tolist(),
    )


def _get_json_config(config_dict: dict[str, Any] | None) -> str:
    json_config = "{}"
    if config_dict is not None:
        json_config = json.dumps(config_dict, sort_keys=True)
    return json_config


def make_mesh_summary(
    tag: str,
    vertices: np.ndarray,
    colors: np.ndarray | None,
    faces: np.ndarray | None,
    config_dict: dict[str, Any] | None,
    display_name: str | None = None,
    description: str | None = None,
) -> "Summary":  # type: ignore[valid-type]
    json_config = _get_json_config(config_dict)

    summaries = []
    tensors = [
        (vertices, MeshPluginData.VERTEX),  # type: ignore[attr-defined]
        (faces, MeshPluginData.FACE),  # type: ignore[attr-defined]
        (colors, MeshPluginData.COLOR),  # type: ignore[attr-defined]
    ]
    # Filter out None tensors and explicitly type the list
    valid_tensors = [(t, content_type) for t, content_type in tensors if t is not None]
    components = mesh_metadata.get_components_bitmask([content_type for (_, content_type) in valid_tensors])

    for tensor, content_type in valid_tensors:
        tensor_metadata = mesh_metadata.create_summary_metadata(
            tag,
            display_name,
            content_type,
            components,
            tensor.shape,  # Safe now since tensor is not None
            description,
            json_config=json_config,
        )

        tensor_proto = TensorProto(
            dtype="DT_FLOAT",
            float_val=tensor.reshape(-1).tolist(),  # Safe now since tensor is not None
            tensor_shape=TensorShapeProto(
                dim=[
                    TensorShapeProto.Dim(size=tensor.shape[0]),  # type: ignore[attr-defined]
                    TensorShapeProto.Dim(size=tensor.shape[1]),  # type: ignore[attr-defined]
                    TensorShapeProto.Dim(size=tensor.shape[2]),  # type: ignore[attr-defined]
                ]
            ),
        )

        tensor_summary = Summary.Value(  # type: ignore[attr-defined]
            tag=mesh_metadata.get_instance_name(tag, content_type),
            tensor=tensor_proto,
            metadata=tensor_metadata,
        )

        summaries.append(tensor_summary)

    return Summary(value=summaries)


class TensorboardProtobufWriter:
    def __init__(
        self,
        log_directory: str | Path,
        max_queue_size: int = 10,
        flush_seconds: float = 120.0,
        filename_suffix: str = "",
    ) -> None:
        super().__init__()

        self.log_directory = Path(log_directory)
        self.max_queue_size = max_queue_size
        self.flush_seconds = flush_seconds
        self.filename_suffix = filename_suffix

    @functools.cached_property
    def event_writer(self) -> EventFileWriter:
        return EventFileWriter(
            logdir=str(self.log_directory),
            max_queue_size=self.max_queue_size,
            flush_secs=self.flush_seconds,
            filename_suffix=self.filename_suffix,
        )

    def add_event(
        self,
        event: "Event",  # type: ignore[valid-type]
        step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(
        self,
        summary: "Summary",  # type: ignore[valid-type]
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        event = Event(summary=summary)
        self.add_event(event, step=global_step, walltime=walltime)

    def add_graph(
        self,
        graph: "GraphDef",  # type: ignore[valid-type]
        run_metadata: "RunMetadata | None" = None,  # type: ignore[valid-type]
        walltime: float | None = None,
    ) -> None:
        event = Event(graph_def=graph.SerializeToString())
        self.add_event(event, walltime=walltime)
        if run_metadata is not None:
            trm = TaggedRunMetadata(tag="step1", run_metadata=run_metadata.SerializeToString())
            event = Event(tagged_run_metadata=trm)
            self.add_event(event, walltime=walltime)

    def flush(self) -> None:
        self.event_writer.flush()

    def close(self) -> None:
        self.event_writer.close()


class TensorboardWriter:
    """Defines a class for writing artifacts to Tensorboard.

    Parameters:
        log_directory: The directory to write logs to.
        max_queue_size: The maximum queue size.
        flush_seconds: How often to flush logs.
        filename_suffix: The filename suffix to use.
    """

    def __init__(
        self,
        log_directory: str | Path,
        max_queue_size: int = 10,
        flush_seconds: float = 120.0,
        filename_suffix: str = "",
    ) -> None:
        super().__init__()

        self.pb_writer = TensorboardProtobufWriter(
            log_directory=log_directory,
            max_queue_size=max_queue_size,
            flush_seconds=flush_seconds,
            filename_suffix=filename_suffix,
        )

    def add_scalar(
        self,
        tag: str,
        value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = True,
        double_precision: bool = False,
    ) -> None:
        if new_style:
            self.pb_writer.add_summary(
                Summary(
                    value=[
                        Summary.Value(  # type: ignore[attr-defined]
                            tag=tag,
                            tensor=(
                                TensorProto(double_val=[value], dtype="DT_DOUBLE")
                                if double_precision
                                else TensorProto(float_val=[value], dtype="DT_FLOAT")
                            ),
                            metadata=SummaryMetadata(
                                plugin_data=SummaryMetadata.PluginData(  # type: ignore[attr-defined]
                                    plugin_name="scalars",
                                ),
                            ),
                        )
                    ],
                ),
                global_step=global_step,
                walltime=walltime,
            )
        else:
            self.pb_writer.add_summary(
                Summary(
                    value=[
                        Summary.Value(  # type: ignore[attr-defined]
                            tag=tag,
                            simple_value=value,
                        ),
                    ],
                ),
                global_step=global_step,
                walltime=walltime,
            )

    def add_image(
        self,
        tag: str,
        value: PILImage,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        output = io.BytesIO()
        value.convert("RGB").save(output, format="PNG")
        image_string = output.getvalue()
        output.close()

        self.pb_writer.add_summary(
            Summary(
                value=[
                    Summary.Value(  # type: ignore[attr-defined]
                        tag=tag,
                        image=Summary.Image(  # type: ignore[attr-defined]
                            height=value.height,
                            width=value.width,
                            colorspace=3,  # RGB
                            encoded_image_string=image_string,
                        ),
                    ),
                ],
            ),
            global_step=global_step,
            walltime=walltime,
        )

    def add_video(
        self,
        tag: str,
        value: np.ndarray,
        global_step: int | None = None,
        walltime: float | None = None,
        fps: int = 30,
    ) -> None:
        assert value.ndim == 4, "Video must be 4D array (T, H, W, C)"

        images = [PIL.Image.fromarray(frame).convert("RGB") for frame in value]
        width, height = images[0].size
        big_image = PIL.Image.new("RGB", (width, height * len(images)))
        for i, im in enumerate(images):
            big_image.paste(im, (0, i * height))

        quantized_big = big_image.quantize(method=PIL.Image.Quantize.MAXCOVERAGE, dither=PIL.Image.Dither.NONE)
        palette = quantized_big.getpalette()

        processed = []
        for im in images:
            q = im.quantize(
                method=PIL.Image.Quantize.MAXCOVERAGE,
                palette=quantized_big,
                dither=PIL.Image.Dither.NONE,
            )
            processed.append(q)

        if palette is not None:
            palette[0:3] = [255, 255, 255]
            for im in processed:
                im.putpalette(palette)

        # Create temporary file for GIF
        temp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        try:
            processed[0].save(
                temp_file.name,
                save_all=True,
                append_images=processed[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            with open(temp_file.name, "rb") as f:
                video_string = f.read()

        finally:
            # Clean up temporary file
            try:
                os.remove(temp_file.name)
            except OSError:
                pass

        # Add to summary
        self.pb_writer.add_summary(
            Summary(
                value=[
                    Summary.Value(  # type: ignore[attr-defined]
                        tag=tag,
                        image=Summary.Image(  # type: ignore[attr-defined]
                            height=value.shape[1],
                            width=value.shape[2],
                            colorspace=value.shape[3],
                            encoded_image_string=video_string,
                        ),
                    ),
                ],
            ),
            global_step=global_step,
            walltime=walltime,
        )

    def add_audio(
        self,
        tag: str,
        value: np.ndarray,
        sample_rate: int,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Add audio data to TensorBoard.

        Args:
            tag: Name for the audio
            value: Audio samples as float32 array of shape (T,) or (C, T)
            sample_rate: Audio sample rate in Hz
            global_step: Training step
            walltime: Wall clock time
        """
        # Ensure we have (T,) or (C, T) format
        if value.ndim == 1:
            num_channels = 1
            length_frames = value.shape[0]
            audio_data = value
        elif value.ndim == 2:
            num_channels = value.shape[0]
            length_frames = value.shape[1]
            # Interleave channels for WAV format: (C, T) -> (T, C) -> flatten
            audio_data = value.T.flatten()
        else:
            raise ValueError(f"Expected audio array of shape (T,) or (C, T), got shape {value.shape}")

        # Convert to int16 for WAV encoding
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)

        # Encode as WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        encoded_audio = wav_buffer.getvalue()

        self.pb_writer.add_summary(
            Summary(
                value=[
                    Summary.Value(  # type: ignore[attr-defined]
                        tag=tag,
                        audio=Summary.Audio(  # type: ignore[attr-defined]
                            sample_rate=float(sample_rate),
                            num_channels=num_channels,
                            length_frames=length_frames,
                            encoded_audio_string=encoded_audio,
                            content_type="audio/wav",
                        ),
                    ),
                ],
            ),
            global_step=global_step,
            walltime=walltime,
        )

    def add_text(
        self,
        tag: str,
        value: str,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        self.pb_writer.add_summary(
            Summary(
                value=[
                    Summary.Value(  # type: ignore[attr-defined]
                        tag=tag + "/text_summary",
                        metadata=SummaryMetadata(
                            plugin_data=SummaryMetadata.PluginData(  # type: ignore[attr-defined]
                                plugin_name="text", content=TextPluginData(version=0).SerializeToString()
                            ),
                        ),
                        tensor=TensorProto(
                            dtype="DT_STRING",
                            string_val=[value.encode(encoding="utf_8")],
                            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),  # type: ignore[attr-defined]
                        ),
                    ),
                ],
            ),
            global_step=global_step,
            walltime=walltime,
        )

    def add_histogram(
        self,
        tag: str,
        values: np.ndarray,
        global_step: int | None = None,
        bins: str | np.ndarray = "auto",
        walltime: float | None = None,
        max_bins: int | None = None,
    ) -> None:
        hist = make_histogram(values.astype(float), bins, max_bins)
        self.pb_writer.add_summary(
            Summary(value=[Summary.Value(tag=tag, histo=hist)]),  # type: ignore[attr-defined]
            global_step=global_step,
            walltime=walltime,
        )

    def add_histogram_raw(
        self,
        tag: str,
        min: float | np.ndarray,
        max: float | np.ndarray,
        num: int | np.ndarray,
        sum: float | np.ndarray,
        sum_squares: float | np.ndarray,
        bucket_limits: list[float | np.ndarray],
        bucket_counts: list[int | np.ndarray],
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Add histogram with raw data to summary.

        Args:
            tag: Data identifier
            min: Min value
            max: Max value
            num: Number of values
            sum: Sum of all values
            sum_squares: Sum of squares for all values
            bucket_limits: Upper value per bucket
            bucket_counts: Number of values per bucket
            global_step: Global step value to record
            walltime: Optional override default walltime
        """
        if len(bucket_limits) != len(bucket_counts):
            raise ValueError("len(bucket_limits) != len(bucket_counts)")

        # Convert numpy arrays to Python types
        hist = HistogramProto(
            min=float(min),
            max=float(max),
            num=int(num),
            sum=float(sum),
            sum_squares=float(sum_squares),
            bucket_limit=[float(x) for x in bucket_limits],
            bucket=[int(x) for x in bucket_counts],
        )
        self.pb_writer.add_summary(
            Summary(value=[Summary.Value(tag=tag, histo=hist)]),  # type: ignore[attr-defined]
            global_step=global_step,
            walltime=walltime,
        )

    def add_gaussian_distribution(
        self,
        tag: str,
        mean: float | np.ndarray,
        std: float | np.ndarray,
        bins: int = 16,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Add a Gaussian distribution to the summary.

        Args:
            tag: Data identifier
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution
            bins: Number of bins to use for the histogram
            global_step: Global step value to record
            walltime: Optional override default walltime
        """
        # Convert numpy arrays to Python types
        mean = float(mean)
        std = float(std)

        # Create bin edges spanning ±3 standard deviations
        bin_edges = np.linspace(mean - 3 * std, mean + 3 * std, bins + 1)

        # Calculate the probability density for each bin
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        gaussian_pdf = np.exp(-0.5 * ((bin_centers - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        # Scale the PDF to represent counts
        num_samples = bins * 1000
        bucket_counts = (gaussian_pdf * num_samples * (bin_edges[1] - bin_edges[0])).astype(int)

        # Ensure we have at least one count per bin for visualization
        bucket_counts = np.maximum(bucket_counts, 1)

        # Calculate actual statistics based on the discretized distribution
        total_counts = float(bucket_counts.sum())
        weighted_sum = float((bin_centers * bucket_counts).sum())
        weighted_sum_squares = float((bin_centers**2 * bucket_counts).sum())

        # Convert bin edges to list of floats explicitly
        bucket_limits: list[float | np.ndarray] = [float(x) for x in bin_edges[1:]]

        self.add_histogram_raw(
            tag=tag,
            min=float(bin_edges[0]),
            max=float(bin_edges[-1]),
            num=int(total_counts),
            sum=weighted_sum,
            sum_squares=weighted_sum_squares,
            bucket_limits=bucket_limits,  # Now properly typed
            bucket_counts=bucket_counts.tolist(),
            global_step=global_step,
            walltime=walltime,
        )

    def add_mesh(
        self,
        tag: str,
        vertices: np.ndarray,
        colors: np.ndarray | None,
        faces: np.ndarray | None,
        config_dict: dict[str, Any] | None,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        self.pb_writer.add_summary(
            make_mesh_summary(tag, vertices, colors, faces, config_dict),
            global_step=global_step,
            walltime=walltime,
        )


class TensorboardWriterKwargs(TypedDict):
    max_queue_size: int
    flush_seconds: float
    filename_suffix: str


class TensorboardWriters:
    def __init__(
        self,
        log_directory: str | Path,
        max_queue_size: int = 10,
        flush_seconds: float = 120.0,
        filename_suffix: str = "",
    ) -> None:
        super().__init__()

        self.log_directory = Path(log_directory)

        self.kwargs: TensorboardWriterKwargs = {
            "max_queue_size": max_queue_size,
            "flush_seconds": flush_seconds,
            "filename_suffix": filename_suffix,
        }

    @functools.cached_property
    def light_writer(self) -> TensorboardWriter:
        return TensorboardWriter(self.log_directory / "light", **self.kwargs)

    @functools.cached_property
    def heavy_writer(self) -> TensorboardWriter:
        return TensorboardWriter(self.log_directory / "heavy", **self.kwargs)

    def writer(self, heavy: bool) -> TensorboardWriter:
        return self.heavy_writer if heavy else self.light_writer
