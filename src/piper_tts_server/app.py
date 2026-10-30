"""FastAPI application factory and OpenAI-compatible routes."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .device import DeviceResolver
from .synthesis import SynthesisService
from .voices import VoiceManager

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- metrics

REQUESTS_TOTAL = Counter(
    "piper_tts_requests_total",
    "Total HTTP requests by endpoint and status.",
    ["endpoint", "status"],
)
SYNTHESIS_DURATION = Histogram(
    "piper_tts_synthesis_duration_seconds",
    "End-to-end synthesis duration per request.",
    ["voice"],
)


class SpeechRequest(BaseModel):
    """OpenAI-compatible ``/v1/audio/speech`` request body."""

    model: str | None = Field(default=None, description="Ignored when 'voice' is provided.")
    input: str = Field(..., min_length=1, description="Text to synthesize.")
    # The Piper voice id (e.g. "fr_FR-tom-medium"). When omitted, `model` is used.
    voice: str | None = Field(
        default=None,
        description="Piper voice id, e.g. 'en_US-lessac-medium'. Takes precedence over 'model'.",
    )
    response_format: str = Field(default="wav")


class AppContext:
    """Holds shared singletons for the app."""

    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        self.device_resolver = DeviceResolver(self.settings)
        self.voice_manager = VoiceManager(self.settings, self.device_resolver)
        self.synthesis = SynthesisService()


def create_app(context: AppContext | None = None) -> FastAPI:
    """Build the FastAPI application."""
    ctx = context or AppContext()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        device = ctx.device_resolver.resolve() if ctx.device_resolver else "cpu"
        logger.info(
            "piper-tts-server starting (device=%s, voice_cache_dir=%s, max_loaded_voices=%d)",
            device,
            ctx.settings.voice_cache_dir,
            ctx.settings.max_loaded_voices,
        )
        yield

    app = FastAPI(
        title="piper-tts-server",
        version="0.1.0",
        description="OpenAI-compatible Piper text-to-speech server.",
        lifespan=lifespan,
    )
    app.state.ctx = ctx

    # ------------------------------------------------------------------ healthz

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------- voices

    @app.get("/v1/audio/voices")
    async def list_voices(
        refresh: bool = Query(default=False),
        format: Literal["openai", "simple"] = Query(default="simple"),
    ) -> JSONResponse:
        """List available Piper voices (discovered from HuggingFace, TTL-cached)."""
        started = time.perf_counter()
        try:
            voices = await asyncio.to_thread(ctx.voice_manager.list_voices, refresh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Voice listing failed")
            REQUESTS_TOTAL.labels("/v1/audio/voices", "500").inc()
            raise HTTPException(status_code=500, detail=f"Voice listing failed: {exc}") from exc

        # OpenAI-compatible Voice objects (id, object, created_at) plus Piper extras.
        payload = [
            {
                "id": v.name,
                "name": v.name,
                "object": "audio.voice",
                "created_at": 0,
                "language": v.language,
                "quality": v.quality,
            }
            for v in voices
        ]

        REQUESTS_TOTAL.labels("/v1/audio/voices", "200").inc()
        logger.debug("Listed %d voices in %.3fs.", len(payload), time.perf_counter() - started)

        if format == "openai":
            return JSONResponse({"object": "list", "data": payload, "has_more": False})

        # Default: OpenAI list wrapper + top-level `voices` array of names.
        # llama-swap's speech playground reads the `voices` key (see getVoiceIds);
        # OpenAI-compatible clients read `data`.
        return JSONResponse(
            {
                "object": "list",
                "data": payload,
                "has_more": False,
                "voices": [v["id"] for v in payload],
            }
        )

    # ------------------------------------------------------------------- speech

    @app.post("/v1/audio/speech")
    async def create_speech(request: Request, body: SpeechRequest):
        """Synthesize speech and stream it back as WAV (OpenAI-compatible)."""
        if body.response_format != "wav":
            raise HTTPException(
                status_code=400, detail=f"Unsupported response_format {body.response_format!r}; only 'wav' is supported."
            )

        # `voice` carries the Piper voice id; when omitted, fall back to `model`.
        voice_name = body.voice or body.model
        if not voice_name:
            raise HTTPException(status_code=400, detail="Either 'voice' or 'model' must be provided.")

        # Load (or download) the model off the event loop.
        try:
            voice = await asyncio.to_thread(ctx.voice_manager.get_voice, voice_name)
        except FileNotFoundError as exc:
            REQUESTS_TOTAL.labels("/v1/audio/speech", "404").inc()
            raise HTTPException(status_code=404, detail=f"Unknown voice: {voice_name}") from exc
        except Exception as exc:
            logger.exception("Failed to load voice %r", voice_name)
            REQUESTS_TOTAL.labels("/v1/audio/speech", "500").inc()
            raise HTTPException(status_code=500, detail=f"Voice loading failed: {exc}") from exc

        # Run the blocking Piper synthesis in a worker thread. The full WAV is
        # buffered there so the response always carries correct RIFF/data sizes;
        # errors surface before any response bytes are sent.
        try:
            wav_bytes = await asyncio.to_thread(ctx.synthesis.synthesize_wav_bytes, voice, body.input)
        except Exception as exc:  # noqa: BLE001 - synthesis failures map to a clean HTTP error
            logger.exception("Synthesis failed for voice %r", voice_name)
            REQUESTS_TOTAL.labels("/v1/audio/speech", "500").inc()
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

        REQUESTS_TOTAL.labels("/v1/audio/speech", "200").inc()
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Voice": voice_name},
        )

    # ------------------------------------------------------------------ metrics

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus scrape endpoint (default registry)."""
        from prometheus_client import REGISTRY as DEFAULT_REGISTRY

        return Response(generate_latest(DEFAULT_REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


# Convenience for ``uvicorn piper_tts_server.app:app``.
app = create_app()
