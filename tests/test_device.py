"""Tests for inference device resolution (CPU vs CUDA)."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from piper_tts_server.config import DeviceChoice, Settings
from piper_tts_server.device import DeviceResolver


def _settings(device: str) -> Settings:
    return Settings(device=DeviceChoice(device), voice_cache_dir="./voices")


@pytest.mark.parametrize(
    ("requested", "providers", "expected"),
    [
        # Explicit CPU always wins.
        ("cpu", ["CPUExecutionProvider"], "cpu"),
        ("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"], "cpu"),
        # Explicit CUDA requires the provider, otherwise falls back to CPU.
        ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"),
        ("cuda", ["CPUExecutionProvider"], "cpu"),
        # Auto picks CUDA when available, else CPU.
        ("auto", ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"], "cuda"),
        ("auto", ["CPUExecutionProvider"], "cpu"),
    ],
)
def test_resolve(requested: str, providers: list[str], expected: str) -> None:
    fake_onnxruntime = types.ModuleType("onnxruntime")

    def _providers() -> list[str]:
        return providers

    fake_onnxruntime.get_available_providers = _providers  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), \
         patch("piper_tts_server.device._query_compute_capability", return_value=(8, 6)):
        resolver = DeviceResolver(_settings(requested))
        assert resolver.resolve() == expected


def test_use_cuda_property_follows_resolution() -> None:
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), \
         patch("piper_tts_server.device._query_compute_capability", return_value=(8, 6)):
        assert DeviceResolver(_settings("auto")).use_cuda is True
        assert DeviceResolver(_settings("cpu")).use_cuda is False


def test_resolution_is_cached() -> None:
    calls = 0

    def _providers() -> list[str]:
        nonlocal calls
        calls += 1
        return ["CPUExecutionProvider"]

    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = _providers  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), \
         patch("piper_tts_server.device._query_compute_capability", return_value=(8, 6)):
        resolver = DeviceResolver(_settings("auto"))
        assert resolver.resolve() == "cpu"
        assert resolver.resolve() == "cpu"
        assert calls == 1


def test_cuda_unavailable_logs_fallback_and_returns_cpu(caplog: pytest.LogCaptureFixture) -> None:
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), \
         patch("piper_tts_server.device._query_compute_capability", return_value=(8, 6)), \
         caplog.at_level("WARNING"):
        resolver = DeviceResolver(_settings("cuda"))
        assert resolver.resolve() == "cpu"
        assert any("falling back to CPU" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# GPU compute capability tests (Pascal fallback)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("requested", "compute_cap", "expected"),
    [
        # Pascal GTX 1080 (CC 6.1) — below minimum, falls back to CPU.
        ("auto", (6, 1), "cpu"),
        ("cuda", (6, 1), "cpu"),
        # Turing RTX 2080 Ti (CC 7.5) — compatible, uses CUDA.
        ("auto", (7, 5), "cuda"),
        ("cuda", (7, 5), "cuda"),
        # Ampere A100 (CC 8.0) — compatible.
        ("auto", (8, 0), "cuda"),
        # Unknown GPU (nvidia-smi unavailable) — assume compatible.
        ("auto", None, "cuda"),
    ],
)
def test_resolve_gpu_compute_capability(
    requested: str, compute_cap: tuple[int, int] | None, expected: str
) -> None:
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), \
         patch("piper_tts_server.device._query_compute_capability", return_value=compute_cap):
        resolver = DeviceResolver(_settings(requested))
        assert resolver.resolve() == expected


def test_pascal_auto_logs_info_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Auto mode on Pascal GPU should log an info message about CPU fallback."""
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), \
         patch("piper_tts_server.device._query_compute_capability", return_value=(6, 1)), \
         caplog.at_level("INFO"):
        resolver = DeviceResolver(_settings("auto"))
        assert resolver.resolve() == "cpu"
        assert any(
            "compute capability is below" in record.message for record in caplog.records
        )


def test_pascal_cuda_logs_warning_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Explicit cuda on Pascal GPU should log a warning about the fallback."""
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), \
         patch("piper_tts_server.device._query_compute_capability", return_value=(6, 1)), \
         caplog.at_level("WARNING"):
        resolver = DeviceResolver(_settings("cuda"))
        assert resolver.resolve() == "cpu"
        assert any(
            "does not meet the minimum compute capability" in record.message for record in caplog.records
        )


def test_query_compute_capability_parses_nvidia_smi_output() -> None:
    """_query_compute_capability should parse nvidia-smi CSV output correctly."""
    from piper_tts_server.device import _query_compute_capability

    fake_result = types.SimpleNamespace(
        returncode=0,
        stdout="8.6\n7.5\n",
        stderr="",
    )

    with patch("piper_tts_server.device.shutil.which", return_value="/usr/bin/nvidia-smi"), \
         patch("piper_tts_server.device.subprocess.run", return_value=fake_result):
        assert _query_compute_capability() == (7, 5)


def test_query_compute_capability_returns_none_when_no_nvidia_smi() -> None:
    """_query_compute_capability should return None when nvidia-smi is missing."""
    from piper_tts_server.device import _query_compute_capability

    with patch("piper_tts_server.device.shutil.which", return_value=None):
        assert _query_compute_capability() is None


def test_query_compute_capability_returns_none_on_error() -> None:
    """_query_compute_capability should return None on non-zero exit code."""
    from piper_tts_server.device import _query_compute_capability

    fake_result = types.SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="NVIDIA-SMI has failed",
    )

    with patch("piper_tts_server.device.shutil.which", return_value="/usr/bin/nvidia-smi"), \
         patch("piper_tts_server.device.subprocess.run", return_value=fake_result):
        assert _query_compute_capability() is None


def test_query_compute_capability_returns_none_on_empty_output() -> None:
    """_query_compute_capability should return None when no GPU lines are found."""
    from piper_tts_server.device import _query_compute_capability

    fake_result = types.SimpleNamespace(
        returncode=0,
        stdout="",
        stderr="",
    )

    with patch("piper_tts_server.device.shutil.which", return_value="/usr/bin/nvidia-smi"), \
         patch("piper_tts_server.device.subprocess.run", return_value=fake_result):
        assert _query_compute_capability() is None
