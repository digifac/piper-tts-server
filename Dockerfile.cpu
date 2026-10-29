# syntax=docker/dockerfile:1
# CPU image for piper-tts-server.
# Inference runs on ONNX Runtime (CPU) via the piper-tts library, which pulls in
# the plain `onnxruntime` package automatically.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Voice models are cached here; mount a volume at this path in production.
    TTS_VOICE_CACHE_DIR=/data/voices

# espeak-ng is required by Piper for phonemization. libgomp is needed by some
# ONNX Runtime CPU kernels (OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libespeak-ng1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the project (and its runtime deps) first to leverage layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Run as a non-root user; give it ownership of the voice cache dir.
RUN useradd --create-home --uid 1000 tts \
    && mkdir -p /data/voices \
    && chown -R tts:tts /data /app
USER tts

EXPOSE 8000

# Liveness/readiness probe hitting the app's health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["python", "-m", "piper_tts_server"]
