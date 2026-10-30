"""Application configuration via environment variables (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", extra="ignore")

    # Directory where voice models (.onnx + .json) are cached on disk.
    voice_cache_dir: Path = Path("./voices")

    # In-memory LRU cache size for loaded PiperVoice models.
    max_loaded_voices: int = 4

    # HTTP server bind address / port.
    host: str = "0.0.0.0"
    port: int = 8000

    # TTL (seconds) for the cached voice listing fetched from HuggingFace.
    voices_list_ttl_seconds: float = 3600.0


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide, cached ``Settings`` instance."""
    return Settings()
