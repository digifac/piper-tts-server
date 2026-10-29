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

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}):
        resolver = DeviceResolver(_settings(requested))
        assert resolver.resolve() == expected


def test_use_cuda_property_follows_resolution() -> None:
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}):
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

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}):
        resolver = DeviceResolver(_settings("auto"))
        assert resolver.resolve() == "cpu"
        assert resolver.resolve() == "cpu"
        assert calls == 1


def test_cuda_unavailable_logs_fallback_and_returns_cpu(caplog: pytest.LogCaptureFixture) -> None:
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}), caplog.at_level("WARNING"):
        resolver = DeviceResolver(_settings("cuda"))
        assert resolver.resolve() == "cpu"
        assert any("falling back to CPU" in record.message for record in caplog.records)
