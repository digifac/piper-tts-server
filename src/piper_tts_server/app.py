"""FastAPI application factory and OpenAI-compatible routes."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
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

    model: str = Field(..., description="Voice name, e.g. 'en_US-lessac-medium'.")
    input: str = Field(..., min_length=1, description="Text to synthesize.")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] | None = Field(
        default=None,
        description="OpenAI voice field; ignored in favor of 'model' (voice name).",
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
    async def list_voices(refresh: bool = Query(default=False)) -> JSONResponse:
        """List available Piper voices (discovered from HuggingFace, TTL-cached)."""
        started = time.perf_counter()
        try:
            voices = await asyncio.to_thread(ctx.voice_manager.list_voices, refresh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Voice listing failed")
            REQUESTS_TOTAL.labels("/v1/audio/voices", "500").inc()
            raise HTTPException(status_code=500, detail=f"Voice listing failed: {exc}") from exc

        payload = [
            {"name": v.name, "language": v.language, "quality": v.quality} for v in voices
        ]
        REQUESTS_TOTAL.labels("/v1/audio/voices", "200").inc()
        logger.debug("Listed %d voices in %.3fs.", len(payload), time.perf_counter() - started)
        return JSONResponse({"data": payload})

    # ------------------------------------------------------------------- speech

    @app.post("/v1/audio/speech")
    async def create_speech(request: Request, body: SpeechRequest):
        """Synthesize speech and stream it back as WAV (OpenAI-compatible)."""
        if body.response_format != "wav":
            raise HTTPException(
                status_code=400, detail=f"Unsupported response_format {body.response_format!r}; only 'wav' is supported."
            )

        voice_name = body.model

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

        # Run the blocking Piper synthesis in a worker thread and bridge chunks.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | str | None] = asyncio.Queue(maxsize=8)

        def producer() -> None:
            try:
                for frame in ctx.synthesis.synthesize_stream(voice, body.input):
                    # Put from a worker thread must be thread-safe.
                    loop.call_soon_threadsafe(queue.put_nowait, frame)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Synthesis failed for voice %r", voice_name)
                loop.call_soon_threadsafe(queue.put_nowait, f"__error__:{exc}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer_task = loop.run_in_executor(None, producer)

        # Drain the queue until the first frame (the WAV header) or an error.
        # Errors raised here happen before any response bytes are sent, so a
        # clean HTTP status code can still be returned to the client.
        first_frame: bytes | None = None
        while True:
            item = await queue.get()
            if item is None:  # empty stream (no audio produced)
                REQUESTS_TOTAL.labels("/v1/audio/speech", "500").inc()
                raise HTTPException(status_code=500, detail="Synthesis produced no audio.") from None
            if isinstance(item, str) and item.startswith("__error__:"):
                REQUESTS_TOTAL.labels("/v1/audio/speech", "500").inc()
                raise HTTPException(
                    status_code=500, detail=item[len("__error__:"):]
                ) from None
            first_frame = item  # WAV header; remaining frames follow in the queue
            break

        async def audio_stream() -> AsyncIterator[bytes]:
            assert first_frame is not None
            yield first_frame
            while True:
                next_item = await queue.get()
                if next_item is None:
                    break
                if isinstance(next_item, str) and next_item.startswith("__error__:"):
                    # Response already started; log only. The client receives a
                    # truncated WAV (zero data size), which players reject cleanly.
                    logger.error(
                        "Synthesis failed mid-stream for voice %r: %s",
                        voice_name,
                        next_item[len("__error__:") :],
                    )
                    break
                yield next_item  # type: ignore[misc] - bytes at this point

            # Ensure the executor task has completed before returning.
            try:
                await producer_task
            except Exception as exc:  # noqa: BLE001 - intentional broad catch; errors already surfaced via queue
                logger.debug("Producer task finished with error: %s", exc)

        REQUESTS_TOTAL.labels("/v1/audio/speech", "200").inc()
        return StreamingResponse(
            audio_stream(),
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
