"""Inference device resolution (CPU vs CUDA).

Resolution order: explicit ``TTS_DEVICE`` override wins; with ``auto``, a CUDA
ONNX Runtime provider is used when available, otherwise CPU. The resolved
device is logged once and exposed for metrics.
"""

from __future__ import annotations

import logging

from .config import DeviceChoice, Settings

logger = logging.getLogger(__name__)


class DeviceResolver:
    """Resolves the ONNX Runtime execution device from settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._resolved: str | None = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import onnxruntime

            return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        except ImportError as exc:  # pragma: no cover - defensive
            logger.warning("Could not query ONNX Runtime providers: %s", exc)
            return False

    def resolve(self) -> str:
        """Return the resolved device name (``"cpu"`` or ``"cuda"``)."""
        if self._resolved is None:
            choice = self._settings.device
            if choice == DeviceChoice.CPU:
                device = "cpu"
            elif choice == DeviceChoice.CUDA:
                if not self._cuda_available():
                    logger.warning(
                        "TTS_DEVICE=cuda requested but no CUDA ONNX Runtime provider is "
                        "available; falling back to CPU. Install onnxruntime-gpu for GPU."
                    )
                    device = "cpu"
                else:
                    device = "cuda"
            else:  # auto
                device = "cuda" if self._cuda_available() else "cpu"

            self._resolved = device
            logger.info("Resolved inference device: %s (requested=%s)", device, choice.value)
        return self._resolved

    @property
    def use_cuda(self) -> bool:
        """Whether loaded PiperVoice models should be created with ``use_cuda=True``."""
        return self.resolve() == "cuda"
