"""Voice discovery, on-disk caching and in-memory LRU of loaded models."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from piper import PiperVoice

from .config import Settings
from .device import DeviceResolver

logger = logging.getLogger(__name__)

# HuggingFace repo hosting the official Piper voices.
VOICES_REPO_ID = "rhasspy/piper-voices"


@dataclass(frozen=True)
class VoiceInfo:
    """Metadata about an available voice."""

    name: str  # e.g. "en_US-lessac-medium"
    language: str  # e.g. "en_US"
    quality: str  # e.g. "medium"


class VoiceManager:
    """Downloads, caches and loads Piper voices.

    - On-disk cache in ``settings.voice_cache_dir`` (persists across restarts).
    - In-memory LRU of loaded :class:`PiperVoice` models (size
      ``settings.max_loaded_voices``), thread-safe via a lock.
    """

    def __init__(self, settings: Settings, device_resolver: DeviceResolver) -> None:
        self._settings = settings
        self._device = device_resolver
        self._cache_dir = Path(settings.voice_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # LRU of loaded models.
        self._lock = threading.Lock()
        self._models: dict[str, PiperVoice] = {}  # insertion-ordered (LRU order)

        # Voice listing cache with TTL.
        self._list_lock = threading.Lock()
        self._voices_cache: list[VoiceInfo] | None = None
        self._voices_fetched_at: float = 0.0

    # ------------------------------------------------------------------ voices

    def list_voices(self, refresh: bool = False) -> list[VoiceInfo]:
        """Return available voice names from HuggingFace (cached with TTL)."""
        now = time.monotonic()
        if not refresh and self._voices_cache is not None:
            age = now - self._voices_fetched_at
            if age < self._settings.voices_list_ttl_seconds:
                return list(self._voices_cache)

        voices = self._fetch_voice_listing()
        with self._list_lock:
            self._voices_cache = voices
            self._voices_fetched_at = now
        return list(voices)

    def _fetch_voice_listing(self) -> list[VoiceInfo]:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            entries = api.list_repo_tree(VOICES_REPO_ID, repo_type="model", recursive=True)
        except (OSError, ValueError) as exc:
            logger.warning("HuggingFace voice discovery failed (%s); returning empty list.", exc)
            return []

        voices: dict[str, VoiceInfo] = {}
        for entry in entries:
            if not str(entry.path).endswith(".onnx"):
                continue
            # Piper voice names follow the pattern ``xx_YY-voice-quality`` where
            # the language code itself contains a dash (e.g. "ar_JO-kareem-low").
            name = Path(str(entry.path)).stem  # e.g. "en_US-lessac-medium"
            prefix, _, quality = name.rpartition("-")  # ("en_US-lessac", "", "medium")
            lang_code, _, _speaker = prefix.partition("-")  # ("en_US", "-", "lessac")
            voices[name] = VoiceInfo(name=name, language=lang_code, quality=quality)

        logger.info("Discovered %d voices on HuggingFace.", len(voices))
        return sorted(voices.values(), key=lambda v: v.name)

    # ------------------------------------------------------------------- model

    def get_voice(self, name: str) -> PiperVoice:
        """Return a loaded :class:`PiperVoice` for ``name`` (downloading if needed)."""
        with self._lock:
            voice = self._models.get(name)
            if voice is not None:
                # Move to end of the dict so it becomes most-recently-used.
                del self._models[name]
                self._models[name] = voice
                return voice

        path = self._ensure_on_disk(name)
        use_cuda = self._device.use_cuda
        logger.info("Loading voice %r (device=%s).", name, "cuda" if use_cuda else "cpu")
        loaded = PiperVoice.load(str(path), use_cuda=use_cuda)

        with self._lock:
            self._models[name] = loaded
            while len(self._models) > self._settings.max_loaded_voices:
                evicted_name, _ = self._models.popitem(last=False)  # LRU eviction (FIFO of least-recently-used)
                logger.info("Evicted voice %r from in-memory cache.", evicted_name)
        return loaded

    def _ensure_on_disk(self, name: str) -> Path:
        """Return the path to ``{name}.onnx``, downloading it if missing."""
        onnx_path = self._cache_dir / f"{name}.onnx"
        json_path = self._cache_dir / f"{name}.json"
        if onnx_path.exists() and json_path.exists():
            return onnx_path

        logger.info("Downloading voice %r to %s.", name, self._cache_dir)
        # Use the current interpreter so piper is importable (works in venvs and Docker).
        cmd = [sys.executable, "-m", "piper.download_voices", name, "--data-dir", str(self._cache_dir)]
        subprocess.run(cmd, check=True)

        if not onnx_path.exists():
            raise FileNotFoundError(f"Voice download for {name!r} did not produce {onnx_path}")
        return onnx_path
