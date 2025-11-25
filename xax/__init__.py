"""Defines the top-level xax API.

This package is structured so that all the important stuff can be accessed
without having to dig around through the internals. This is done by lazily
importing the module by name.

This file can be maintained by updating the imports at the bottom of the file
and running the update script:

.. code-block:: bash

    python -m scripts.update_api --inplace
"""

__version__ = "0.4.4"

# This list shouldn't be modified by hand; instead, run the update script.
__all__ = [
    "UserConfig",
    "field",
    "get_data_dir",
    "get_pretrained_models_dir",
    "get_run_dir",
    "load_user_config",
    "State",
    "AttentionCache",
    "AttentionCacheDict",
    "CrossAttentionBlock",
    "SelfAttentionBlock",
    "Transformer",
    "TransformerBlock",
    "TransformerCache",
    "TransformerStack",
    "Categorical",
    "Distribution",
    "MixtureOfGaussians",
    "Normal",
    "FourierEmbeddings",
    "IdentityPositionalEmbeddings",
    "LearnedPositionalEmbeddings",
    "RotaryEmbeddings",
    "SinusoidalEmbeddings",
    "apply_rotary_embeddings",
    "cast_embedding_kind",
    "fourier_embeddings",
    "get_positional_embeddings",
    "get_rotary_embeddings",
    "rotary_embeddings",
    "cubic_bezier_interpolation",
    "euler_to_quat",
    "get_projected_gravity_vector_from_quat",
    "normalize",
    "quat_mul",
    "quat_to_euler",
    "quat_to_rotmat",
    "quat_to_yaw",
    "rotate_vector_by_quat",
    "rotation6d_to_rotation_matrix",
    "rotation_matrix_to_quat",
    "rotation_matrix_to_rotation6d",
    "cross_entropy",
    "cast_norm_type",
    "dynamic_time_warping",
    "get_norm",
    "is_master",
    "BaseSSMBlock",
    "DiagSSMBlock",
    "SSM",
    "SSMBlock",
    "BaseLauncher",
    "CliLauncher",
    "MultiDeviceLauncher",
    "SingleDeviceLauncher",
    "LogDistribution",
    "LogError",
    "LogErrorSummary",
    "LogGraph",
    "LogHistogram",
    "LogImage",
    "LogLine",
    "LogMesh",
    "LogPing",
    "LogScalar",
    "LogStatus",
    "LogVideo",
    "Logger",
    "LoggerImpl",
    "CallbackLogger",
    "JsonLogger",
    "StateLogger",
    "StdoutLogger",
    "TensorboardLogger",
    "load_ckpt",
    "CPUStatsOptions",
    "DataloadersConfig",
    "DataloadersMixin",
    "GPUStatsOptions",
    "ParallelConfig",
    "ParallelMixin",
    "StepContext",
    "InitParams",
    "Script",
    "ScriptConfig",
    "Config",
    "SupervisedConfig",
    "SupervisedTask",
    "Task",
    "collate",
    "collate_non_null",
    "breakpoint_if_nonfinite",
    "get_named_leaves",
    "log_if_nonfinite",
    "BaseFileDownloader",
    "ContextTimer",
    "CumulativeTimer",
    "DataDownloader",
    "IntervalTicker",
    "IterationTimer",
    "MinGradScaleError",
    "ModelDownloader",
    "NaNError",
    "StateTimer",
    "TrainingFinishedError",
    "check_md5",
    "check_sha256",
    "cpu_count",
    "date_str",
    "diff_configs",
    "get_git_state",
    "get_random_port",
    "get_state_dict_prefix",
    "get_training_code",
    "save_config",
    "stage_environment",
    "to_markdown_table",
    "grad",
    "jit",
    "scan",
    "vmap",
    "save_jaxpr_dot",
    "ColoredFormatter",
    "configure_logging",
    "one_hot",
    "partial_flatten",
    "worker_chunk",
    "profile",
    "compute_nan_ratio",
    "diff_pytree",
    "flatten_array",
    "flatten_pytree",
    "get_pytree_mapping",
    "get_pytree_param_count",
    "pytree_has_nans",
    "reshuffle_pytree",
    "reshuffle_pytree_along_dims",
    "reshuffle_pytree_independently",
    "slice_array",
    "slice_pytree",
    "tuple_insert",
    "update_pytree",
    "TextBlock",
    "camelcase_to_snakecase",
    "colored",
    "format_datetime",
    "format_timedelta",
    "highlight_exception_message",
    "is_interactive_session",
    "outlined",
    "render_text_blocks",
    "show_error",
    "show_info",
    "show_warning",
    "snakecase_to_camelcase",
    "uncolored",
    "wrapped",
    "FrozenDict",
    "freeze_dict",
    "HashableArray",
    "hashable_array",
]

__all__ += [
    "Batch",
    "CollateMode",
    "EmbeddingKind",
    "ActivationFunction",
    "DTYPE",
    "LOG_ERROR_SUMMARY",
    "LOG_PING",
    "LOG_STATUS",
    "NormType",
    "Output",
    "RawConfigType",
]

import os
import shutil
from typing import TYPE_CHECKING

# Sets some useful XLA flags.
xla_flags: list[str] = []
if "XLA_FLAGS" in os.environ:
    xla_flags.append(os.environ["XLA_FLAGS"])

# If Nvidia GPU is detected (meaning, is `nvidia-smi` available?), disable
# Triton GEMM kernels. See https://github.com/NVIDIA/JAX-Toolbox
if shutil.which("nvidia-smi") is not None:
    xla_flags += [
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_triton_gemm=false",
    ]
os.environ["XLA_FLAGS"] = " ".join(xla_flags)

# If this flag is set, eagerly imports the entire package (not recommended).
IMPORT_ALL = int(os.environ.get("XAX_IMPORT_ALL", "0")) != 0

del os, shutil, xla_flags

# This dictionary is auto-generated and shouldn't be modified by hand; instead,
# run the update script.
NAME_MAP: dict[str, str] = {
    "UserConfig": "core.conf",
    "field": "core.conf",
    "get_data_dir": "core.conf",
    "get_pretrained_models_dir": "core.conf",
    "get_run_dir": "core.conf",
    "load_user_config": "core.conf",
    "State": "core.state",
    "AttentionCache": "nn.attention",
    "AttentionCacheDict": "nn.attention",
    "CrossAttentionBlock": "nn.attention",
    "SelfAttentionBlock": "nn.attention",
    "Transformer": "nn.attention",
    "TransformerBlock": "nn.attention",
    "TransformerCache": "nn.attention",
    "TransformerStack": "nn.attention",
    "Categorical": "nn.distributions",
    "Distribution": "nn.distributions",
    "MixtureOfGaussians": "nn.distributions",
    "Normal": "nn.distributions",
    "FourierEmbeddings": "nn.embeddings",
    "IdentityPositionalEmbeddings": "nn.embeddings",
    "LearnedPositionalEmbeddings": "nn.embeddings",
    "RotaryEmbeddings": "nn.embeddings",
    "SinusoidalEmbeddings": "nn.embeddings",
    "apply_rotary_embeddings": "nn.embeddings",
    "cast_embedding_kind": "nn.embeddings",
    "fourier_embeddings": "nn.embeddings",
    "get_positional_embeddings": "nn.embeddings",
    "get_rotary_embeddings": "nn.embeddings",
    "rotary_embeddings": "nn.embeddings",
    "cubic_bezier_interpolation": "nn.geom",
    "euler_to_quat": "nn.geom",
    "get_projected_gravity_vector_from_quat": "nn.geom",
    "normalize": "nn.geom",
    "quat_mul": "nn.geom",
    "quat_to_euler": "nn.geom",
    "quat_to_rotmat": "nn.geom",
    "quat_to_yaw": "nn.geom",
    "rotate_vector_by_quat": "nn.geom",
    "rotation6d_to_rotation_matrix": "nn.geom",
    "rotation_matrix_to_quat": "nn.geom",
    "rotation_matrix_to_rotation6d": "nn.geom",
    "cross_entropy": "nn.losses",
    "cast_norm_type": "nn.metrics",
    "dynamic_time_warping": "nn.metrics",
    "get_norm": "nn.metrics",
    "is_master": "nn.parallel",
    "BaseSSMBlock": "nn.ssm",
    "DiagSSMBlock": "nn.ssm",
    "SSM": "nn.ssm",
    "SSMBlock": "nn.ssm",
    "BaseLauncher": "task.launchers.base",
    "CliLauncher": "task.launchers.cli",
    "MultiDeviceLauncher": "task.launchers.multi_device",
    "SingleDeviceLauncher": "task.launchers.single_device",
    "LogDistribution": "task.logger",
    "LogError": "task.logger",
    "LogErrorSummary": "task.logger",
    "LogGraph": "task.logger",
    "LogHistogram": "task.logger",
    "LogImage": "task.logger",
    "LogLine": "task.logger",
    "LogMesh": "task.logger",
    "LogPing": "task.logger",
    "LogScalar": "task.logger",
    "LogStatus": "task.logger",
    "LogVideo": "task.logger",
    "Logger": "task.logger",
    "LoggerImpl": "task.logger",
    "CallbackLogger": "task.loggers.callback",
    "JsonLogger": "task.loggers.json",
    "StateLogger": "task.loggers.state",
    "StdoutLogger": "task.loggers.stdout",
    "TensorboardLogger": "task.loggers.tensorboard",
    "load_ckpt": "task.mixins.checkpointing",
    "CPUStatsOptions": "task.mixins.cpu_stats",
    "DataloadersConfig": "task.mixins.data_loader",
    "DataloadersMixin": "task.mixins.data_loader",
    "GPUStatsOptions": "task.mixins.gpu_stats",
    "ParallelConfig": "task.mixins.parallel",
    "ParallelMixin": "task.mixins.parallel",
    "StepContext": "task.mixins.step_wrapper",
    "InitParams": "task.mixins.train",
    "Script": "task.script",
    "ScriptConfig": "task.script",
    "Config": "task.task",
    "SupervisedConfig": "task.task",
    "SupervisedTask": "task.task",
    "Task": "task.task",
    "collate": "utils.data.collate",
    "collate_non_null": "utils.data.collate",
    "breakpoint_if_nonfinite": "utils.debugging",
    "get_named_leaves": "utils.debugging",
    "log_if_nonfinite": "utils.debugging",
    "BaseFileDownloader": "utils.experiments",
    "ContextTimer": "utils.experiments",
    "CumulativeTimer": "utils.experiments",
    "DataDownloader": "utils.experiments",
    "IntervalTicker": "utils.experiments",
    "IterationTimer": "utils.experiments",
    "MinGradScaleError": "utils.experiments",
    "ModelDownloader": "utils.experiments",
    "NaNError": "utils.experiments",
    "StateTimer": "utils.experiments",
    "TrainingFinishedError": "utils.experiments",
    "check_md5": "utils.experiments",
    "check_sha256": "utils.experiments",
    "cpu_count": "utils.experiments",
    "date_str": "utils.experiments",
    "diff_configs": "utils.experiments",
    "get_git_state": "utils.experiments",
    "get_random_port": "utils.experiments",
    "get_state_dict_prefix": "utils.experiments",
    "get_training_code": "utils.experiments",
    "save_config": "utils.experiments",
    "stage_environment": "utils.experiments",
    "to_markdown_table": "utils.experiments",
    "grad": "utils.jax",
    "jit": "utils.jax",
    "scan": "utils.jax",
    "vmap": "utils.jax",
    "save_jaxpr_dot": "utils.jaxpr",
    "ColoredFormatter": "utils.logging",
    "configure_logging": "utils.logging",
    "one_hot": "utils.numpy",
    "partial_flatten": "utils.numpy",
    "worker_chunk": "utils.numpy",
    "profile": "utils.profile",
    "compute_nan_ratio": "utils.pytree",
    "diff_pytree": "utils.pytree",
    "flatten_array": "utils.pytree",
    "flatten_pytree": "utils.pytree",
    "get_pytree_mapping": "utils.pytree",
    "get_pytree_param_count": "utils.pytree",
    "pytree_has_nans": "utils.pytree",
    "reshuffle_pytree": "utils.pytree",
    "reshuffle_pytree_along_dims": "utils.pytree",
    "reshuffle_pytree_independently": "utils.pytree",
    "slice_array": "utils.pytree",
    "slice_pytree": "utils.pytree",
    "tuple_insert": "utils.pytree",
    "update_pytree": "utils.pytree",
    "TextBlock": "utils.text",
    "camelcase_to_snakecase": "utils.text",
    "colored": "utils.text",
    "format_datetime": "utils.text",
    "format_timedelta": "utils.text",
    "highlight_exception_message": "utils.text",
    "is_interactive_session": "utils.text",
    "outlined": "utils.text",
    "render_text_blocks": "utils.text",
    "show_error": "utils.text",
    "show_info": "utils.text",
    "show_warning": "utils.text",
    "snakecase_to_camelcase": "utils.text",
    "uncolored": "utils.text",
    "wrapped": "utils.text",
    "FrozenDict": "utils.types.frozen_dict",
    "freeze_dict": "utils.types.frozen_dict",
    "HashableArray": "utils.types.hashable_array",
    "hashable_array": "utils.types.hashable_array",
}

# Need to manually set some values which can't be auto-generated.
NAME_MAP.update(
    {
        "Batch": "task.mixins.train",
        "CollateMode": "utils.data.collate",
        "EmbeddingKind": "nn.embeddings",
        "LOG_ERROR_SUMMARY": "utils.logging",
        "LOG_PING": "utils.logging",
        "LOG_STATUS": "utils.logging",
        "NormType": "nn.metrics",
        "Output": "task.mixins.output",
        "RawConfigType": "task.base",
        "ActivationFunction": "nn.equinox",
        "DTYPE": "nn.equinox",
    },
)

# In NAME_MAP
NAME_MAP["TransformerStack"] = "nn.attention"


def __getattr__(name: str) -> object:
    if name not in NAME_MAP:
        raise AttributeError(f"{__name__} has no attribute {name}")

    module_name = f"xax.{NAME_MAP[name]}"
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)


if IMPORT_ALL or TYPE_CHECKING:
    from xax.core.conf import (
        UserConfig,
        field,
        get_data_dir,
        get_pretrained_models_dir,
        get_run_dir,
        load_user_config,
    )
    from xax.core.state import State
    from xax.nn.attention import (
        AttentionCache,
        AttentionCacheDict,
        CrossAttentionBlock,
        SelfAttentionBlock,
        Transformer,
        TransformerBlock,
        TransformerCache,
        TransformerStack,
    )
    from xax.nn.distributions import (
        Categorical,
        Distribution,
        MixtureOfGaussians,
        Normal,
    )
    from xax.nn.embeddings import (
        EmbeddingKind,
        FourierEmbeddings,
        IdentityPositionalEmbeddings,
        LearnedPositionalEmbeddings,
        RotaryEmbeddings,
        SinusoidalEmbeddings,
        apply_rotary_embeddings,
        cast_embedding_kind,
        fourier_embeddings,
        get_positional_embeddings,
        get_rotary_embeddings,
        rotary_embeddings,
    )
    from xax.nn.geom import (
        cubic_bezier_interpolation,
        euler_to_quat,
        get_projected_gravity_vector_from_quat,
        normalize,
        quat_mul,
        quat_to_euler,
        quat_to_rotmat,
        quat_to_yaw,
        rotate_vector_by_quat,
        rotation6d_to_rotation_matrix,
        rotation_matrix_to_quat,
        rotation_matrix_to_rotation6d,
    )
    from xax.nn.losses import cross_entropy
    from xax.nn.metrics import NormType, cast_norm_type, dynamic_time_warping, get_norm
    from xax.nn.parallel import is_master
    from xax.nn.ssm import SSM, BaseSSMBlock, DiagSSMBlock, SSMBlock
    from xax.task.base import RawConfigType
    from xax.task.launchers.base import BaseLauncher
    from xax.task.launchers.cli import CliLauncher
    from xax.task.launchers.multi_device import MultiDeviceLauncher
    from xax.task.launchers.single_device import SingleDeviceLauncher
    from xax.task.logger import (
        LogDistribution,
        LogError,
        LogErrorSummary,
        Logger,
        LoggerImpl,
        LogGraph,
        LogHistogram,
        LogImage,
        LogLine,
        LogMesh,
        LogPing,
        LogScalar,
        LogStatus,
        LogVideo,
    )
    from xax.task.loggers.callback import CallbackLogger
    from xax.task.loggers.json import JsonLogger
    from xax.task.loggers.state import StateLogger
    from xax.task.loggers.stdout import StdoutLogger
    from xax.task.loggers.tensorboard import TensorboardLogger
    from xax.task.mixins.checkpointing import load_ckpt
    from xax.task.mixins.cpu_stats import CPUStatsOptions
    from xax.task.mixins.data_loader import DataloadersConfig, DataloadersMixin
    from xax.task.mixins.gpu_stats import GPUStatsOptions
    from xax.task.mixins.parallel import ParallelConfig, ParallelMixin
    from xax.task.mixins.step_wrapper import StepContext
    from xax.task.mixins.train import Batch, InitParams, Output
    from xax.task.script import Script, ScriptConfig
    from xax.task.task import Config, SupervisedConfig, SupervisedTask, Task
    from xax.utils.data.collate import CollateMode, collate, collate_non_null
    from xax.utils.debugging import (
        breakpoint_if_nonfinite,
        get_named_leaves,
        log_if_nonfinite,
    )
    from xax.utils.experiments import (
        BaseFileDownloader,
        ContextTimer,
        CumulativeTimer,
        DataDownloader,
        IntervalTicker,
        IterationTimer,
        MinGradScaleError,
        ModelDownloader,
        NaNError,
        StateTimer,
        TrainingFinishedError,
        check_md5,
        check_sha256,
        cpu_count,
        date_str,
        diff_configs,
        get_git_state,
        get_random_port,
        get_state_dict_prefix,
        get_training_code,
        save_config,
        stage_environment,
        to_markdown_table,
    )
    from xax.utils.jax import grad, jit, scan, vmap
    from xax.utils.jaxpr import save_jaxpr_dot
    from xax.utils.logging import (
        LOG_ERROR_SUMMARY,
        LOG_PING,
        LOG_STATUS,
        ColoredFormatter,
        configure_logging,
    )
    from xax.utils.numpy import one_hot, partial_flatten, worker_chunk
    from xax.utils.profile import profile
    from xax.utils.pytree import (
        compute_nan_ratio,
        diff_pytree,
        flatten_array,
        flatten_pytree,
        get_pytree_mapping,
        get_pytree_param_count,
        pytree_has_nans,
        reshuffle_pytree,
        reshuffle_pytree_along_dims,
        reshuffle_pytree_independently,
        slice_array,
        slice_pytree,
        tuple_insert,
        update_pytree,
    )
    from xax.utils.text import (
        TextBlock,
        camelcase_to_snakecase,
        colored,
        format_datetime,
        format_timedelta,
        highlight_exception_message,
        is_interactive_session,
        outlined,
        render_text_blocks,
        show_error,
        show_info,
        show_warning,
        snakecase_to_camelcase,
        uncolored,
        wrapped,
    )
    from xax.utils.types.frozen_dict import FrozenDict, freeze_dict
    from xax.utils.types.hashable_array import HashableArray, hashable_array

del TYPE_CHECKING, IMPORT_ALL
