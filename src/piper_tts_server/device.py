"""Inference device resolution (CPU vs CUDA).

Resolution order: explicit ``TTS_DEVICE`` override wins; with ``auto``, a CUDA
ONNX Runtime provider is used when available, otherwise CPU. The resolved
device is logged once and exposed for metrics.

GPU compatibility: ONNX Runtime's CUDAExecutionProvider requires compute
capability >= 7.0 (Turing or newer). Pascal GPUs (GTX 10xx) are not supported
by recent CUDA/ONNX Runtime builds; the resolver detects this via ``nvidia-smi``
and falls back to CPU automatically.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

from .config import DeviceChoice, Settings

logger = logging.getLogger(__name__)

# Minimum compute capability required by ONNX Runtime CUDAExecutionProvider.
_MIN_COMPUTE_CAPABILITY: tuple[int, int] = (7, 0)


def _query_compute_capability() -> tuple[int, int] | None:
    """Return the minimum compute capability across all visible NVIDIA GPUs.

    Uses ``nvidia-smi --query-gpu=compute_cap`` which returns values like
    ``"6.1"`` or ``"8.6"``. Returns ``None`` if nvidia-smi is unavailable,
    no GPU is detected, or the query fails for any reason (e.g. running in a
    container without the NVIDIA driver). Callers should treat ``None`` as
    "unknown — assume compatible".
    """
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=False,  # fixed args, no user input; non-zero handled below
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - defensive
        logger.debug("nvidia-smi query failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug(
            "nvidia-smi returned non-zero exit code %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return None

    caps: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*(\d+)\.(\d+)", line)
        if match:
            caps.append((int(match.group(1)), int(match.group(2))))

    return min(caps) if caps else None


def _gpu_compatible() -> bool:
    """Check whether the host GPU meets the minimum compute capability.

    Returns ``True`` when the check cannot be performed (no nvidia-smi, no
    GPU visible, etc.) so that we don't block CUDA on hosts where detection
    is simply unavailable.
    """
    cap = _query_compute_capability()
    if cap is None:
        return True  # unknown — assume compatible

    if cap < _MIN_COMPUTE_CAPABILITY:
        logger.warning(
            "GPU compute capability %d.%d detected, but ONNX Runtime CUDA requires >= %d.%d. "
            "CUDA acceleration will not be used.",
            cap[0],
            cap[1],
            *_MIN_COMPUTE_CAPABILITY,
        )
        return False

    logger.debug("GPU compute capability %d.%d — compatible with CUDA.", *cap)
    return True


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
                elif not _gpu_compatible():
                    logger.warning(
                        "TTS_DEVICE=cuda requested but the host GPU does not meet the minimum "
                        "compute capability (>= %d.%d); falling back to CPU.",
                        *_MIN_COMPUTE_CAPABILITY,
                    )
                    device = "cpu"
                else:
                    device = "cuda"
            else:  # auto
                cuda_ok = self._cuda_available()
                if cuda_ok and _gpu_compatible():
                    device = "cuda"
                elif cuda_ok:
                    logger.info(
                        "CUDA provider present but GPU compute capability is below %d.%d; "
                        "using CPU.",
                        *_MIN_COMPUTE_CAPABILITY,
                    )
                    device = "cpu"
                else:
                    device = "cpu"

            self._resolved = device
            logger.info("Resolved inference device: %s (requested=%s)", device, choice.value)
        return self._resolved

    @property
    def use_cuda(self) -> bool:
        """Whether loaded PiperVoice models should be created with ``use_cuda=True``."""
        return self.resolve() == "cuda"
