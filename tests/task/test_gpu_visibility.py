"""Tests for queue-aware GPU visibility masking."""

import os
from subprocess import CompletedProcess
from typing import Any, cast

import pytest

from xax.task.launchers.multi_device import MultiDeviceLauncher
from xax.task.launchers.single_device import SingleDeviceLauncher
from xax.utils.launcher.gpu_visibility import (
    QUEUE_GPUS_ENV_VAR,
    QUEUE_JOB_FLAG_ENV_VAR,
    QUEUE_NUM_GPUS_ENV_VAR,
    apply_queue_gpu_visibility,
)
from xax.utils.launcher.queue_state import ObserverInfo


def _observer_info() -> ObserverInfo:
    return ObserverInfo(
        pid=123,
        hostname="test-host",
        started_at=0.0,
        updated_at=0.0,
        status="idle",
    )


def test_apply_queue_gpu_visibility_masks_reserved_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv(QUEUE_JOB_FLAG_ENV_VAR, raising=False)

    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.is_observer_active", lambda: True)
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.read_observer_info", _observer_info)
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_visibility._read_process_environ",
        lambda _pid: {QUEUE_GPUS_ENV_VAR: "0,1"},
    )
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility._discover_gpu_indices", lambda: [0, 1, 2, 3])

    visible = apply_queue_gpu_visibility()

    assert visible == [2, 3]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "2,3"


def test_apply_queue_gpu_visibility_intersects_existing_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3")
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv(QUEUE_JOB_FLAG_ENV_VAR, raising=False)

    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.is_observer_active", lambda: True)
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.read_observer_info", _observer_info)
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_visibility._read_process_environ",
        lambda _pid: {QUEUE_GPUS_ENV_VAR: "0,1"},
    )

    visible = apply_queue_gpu_visibility()

    assert visible == [2, 3]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_apply_queue_gpu_visibility_skips_inside_queue_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(QUEUE_JOB_FLAG_ENV_VAR, "1")

    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.is_observer_active", lambda: True)
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.read_observer_info", _observer_info)
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_visibility._read_process_environ",
        lambda _pid: {QUEUE_GPUS_ENV_VAR: "0,1"},
    )
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility._discover_gpu_indices", lambda: [0, 1, 2, 3])

    visible = apply_queue_gpu_visibility()

    assert visible is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_queue_gpu_visibility_uses_queue_num_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv(QUEUE_JOB_FLAG_ENV_VAR, raising=False)

    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.is_observer_active", lambda: True)
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.read_observer_info", _observer_info)
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_visibility._read_process_environ",
        lambda _pid: {QUEUE_NUM_GPUS_ENV_VAR: "2"},
    )
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility._discover_gpu_indices", lambda: [0, 1, 2, 3])

    visible = apply_queue_gpu_visibility()

    assert visible == [2, 3]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_apply_queue_gpu_visibility_skips_unparseable_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    monkeypatch.delenv(QUEUE_JOB_FLAG_ENV_VAR, raising=False)

    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.is_observer_active", lambda: True)
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.read_observer_info", _observer_info)
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_visibility._read_process_environ",
        lambda _pid: {QUEUE_GPUS_ENV_VAR: "0,1"},
    )

    visible = apply_queue_gpu_visibility()

    assert visible is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-abc"


def test_apply_queue_gpu_visibility_queries_gpus_when_cuda_visible_devices_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv(QUEUE_JOB_FLAG_ENV_VAR, raising=False)

    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.is_observer_active", lambda: True)
    monkeypatch.setattr("xax.utils.launcher.gpu_visibility.read_observer_info", _observer_info)
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_visibility._read_process_environ",
        lambda _pid: {QUEUE_GPUS_ENV_VAR: "1"},
    )
    monkeypatch.setattr("xax.utils.launcher.gpu_utils.shutil.which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "xax.utils.launcher.gpu_utils.subprocess.run",
        lambda *_args, **_kwargs: CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0\n1\n",
            stderr="",
        ),
    )

    visible = apply_queue_gpu_visibility()

    assert visible == [0]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "0"


class _DummyTask:
    pass


def test_single_device_launcher_raises_when_no_non_queue_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("xax.task.launchers.single_device.configure_logging", lambda: None)
    monkeypatch.setattr("xax.task.launchers.single_device.apply_queue_gpu_visibility", lambda _logger: [])

    with pytest.raises(RuntimeError, match="No GPUs remain for local single-device launch"):
        SingleDeviceLauncher().launch(cast(Any, _DummyTask))


def test_multi_device_launcher_raises_when_no_non_queue_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("xax.task.launchers.multi_device.configure_logging", lambda: None)
    monkeypatch.setattr("xax.task.launchers.multi_device.apply_queue_gpu_visibility", lambda _logger: [])

    with pytest.raises(RuntimeError, match="No GPUs remain for local multi-device launch"):
        MultiDeviceLauncher().launch(cast(Any, _DummyTask))


def test_single_device_launcher_skips_gpu_masking_on_help(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"run": False}

    monkeypatch.setattr(
        "xax.task.launchers.single_device.apply_queue_gpu_visibility",
        lambda _logger: (_ for _ in ()).throw(AssertionError("unexpected gpu masking call")),
    )
    monkeypatch.setattr(
        "xax.task.launchers.single_device.configure_single_device",
        lambda _logger: (_ for _ in ()).throw(AssertionError("unexpected device configuration")),
    )
    monkeypatch.setattr(
        "xax.task.launchers.single_device.run_runnable_task",
        lambda *_args, **_kwargs: called.__setitem__("run", True),
    )

    SingleDeviceLauncher().launch(cast(Any, _DummyTask), use_cli=["--help"])
    assert called["run"]


def test_multi_device_launcher_skips_gpu_masking_on_help(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"run": False}

    monkeypatch.setattr(
        "xax.task.launchers.multi_device.apply_queue_gpu_visibility",
        lambda _logger: (_ for _ in ()).throw(AssertionError("unexpected gpu masking call")),
    )
    monkeypatch.setattr(
        "xax.task.launchers.multi_device.run_runnable_task",
        lambda *_args, **_kwargs: called.__setitem__("run", True),
    )

    MultiDeviceLauncher().launch(cast(Any, _DummyTask), use_cli=["--help"])
    assert called["run"]
