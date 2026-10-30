# piper-tts-server

[![CI](https://github.com/digifac/piper-tts-server/actions/workflows/ci.yml/badge.svg)](https://github.com/digifac/piper-tts-server/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/digifac/piper-tts-server?label=release&color=brightgreen)](https://github.com/digifac/piper-tts-server/releases/latest)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](./LICENSE)

> **⚠️ Beta** — This project is in beta. The API and behavior may change without notice, and some features may be unstable or incomplete. Use at your own risk.

A low-latency text-to-speech server based on [Piper](https://github.com/OHF-Voice/piper1-gpl), exposing an **OpenAI-compatible HTTP API**. Inference runs on ONNX Runtime via the [`piper-tts`](https://pypi.org/project/piper-tts/) library.

## Features

- OpenAI-compatible HTTP API for speech synthesis and voice listing — works with the official OpenAI SDKs as well as plain `curl`.
- **CPU inference** via ONNX Runtime, automatically leveraging AVX/AVX2 when available on the host.
- Low time-to-first-byte: `/v1/audio/speech` streams WAV frames back as synthesis progresses (`StreamingResponse`).
- Automatic voice model downloads with on-disk caching, plus an in-memory LRU of loaded models so the most-recently-used voices stay hot.
- HuggingFace-based discovery of available voices, exposed through the `/v1/audio/voices` endpoint (TTL-cached).
- Health check (`/healthz`) and Prometheus-compatible metrics (`/metrics`).

Inference runs entirely on the **CPU** via ONNX Runtime, automatically leveraging AVX/AVX2 when available on the host. CPU mode works on all hardware and is perfectly usable for moderate workloads.

## Installation

### From source (local development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # CPU inference + dev/test tooling
```

Run the server:

```bash
python -m piper_tts_server     # listens on 0.0.0.0:8000 by default
```

### Docker

Versioned images are published to [GHCR](https://ghcr.io/digifac/piper-tts-server) on every `v0.*` tag (see [Releases](https://github.com/digifac/piper-tts-server/releases)). Pull the latest release:

```bash
docker pull ghcr.io/digifac/piper-tts-server:latest          # CPU inference
```

Or pin a specific version, e.g. `0.1.0`:

```bash
docker run --rm -p 8000:8000 ghcr.io/digifac/piper-tts-server:0.1.0
```

> **Voice cache:** downloaded voice models are cached in `TTS_VOICE_CACHE_DIR` (default `/data/voices`). Mount a volume there so voices persist across container restarts and don't need to be re-downloaded. A complete example for running the server as a long-lived daemon:
>
> ```bash
> docker run -d \
>   --name piper-tts-server \
>   --restart unless-stopped \
>   -p 8000:8000 \
>   -v piper-voices:/data/voices \
>   ghcr.io/digifac/piper-tts-server:latest
> ```

> Images are built for `linux/amd64` only at the moment.

You can also build locally from source (useful when hacking on the Dockerfile):

```bash
docker build -f Dockerfile.cpu -t piper-tts-server:cpu .
```

Or use Compose (the `cpu` profile references a locally-built image):

```bash
docker compose up cpu                 # CPU inference
```

### Kubernetes

Manifests live in [`k8s/`](./k8s). The voice cache is backed by a PersistentVolumeClaim so downloaded voices persist across pod restarts:

```bash
kubectl apply -f k8s/pvc-voices.yaml
kubectl apply -f k8s/deployment-cpu.yaml -f k8s/service.yaml
```

> **PVC note:** `k8s/pvc-voices.yaml` requests 500Mi and sets no explicit `storageClassName`, so it uses your cluster's default StorageClass (commonly `local-path`). Set an explicit `storageClassName:` if you need a specific provisioner, and raise the request up to ~2Gi if you cache many voices.

## Usage

### List voices

```bash
curl http://localhost:8000/v1/audio/voices
# -> {"object": "list", "has_more": false, "data": [
#      {"id": "en_US-lessac-medium", "name": "en_US-lessac-medium", "object": "audio.voice",
#       "created_at": 0, "language": "en_US", "quality": "medium"}, ...]}
```

The response follows the OpenAI `Voice` schema (`id`, `object: "audio.voice"`, `created_at`) and adds Piper-specific extras (`language` as a bare `xx_YY` code, `quality`).

### Synthesize speech (streaming WAV)

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"en_US-lessac-medium","input":"Hello, this is a test."}' \
  --output out.wav
```

The `model` field selects the Piper voice. The optional `voice` field (OpenAI-compatible) is accepted but `model` takes precedence. `response_format` defaults to `wav`.

### OpenAI SDK

Because the API mirrors OpenAI's shape, you can point the official SDK at this server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with client.audio.speech.with_streaming_response.create(
    model="en_US-lessac-medium",
    voice="en_US-lessac-medium",  # mapped to the Piper voice via `model`
    input="Hello, this is a test.",
) as response:
    with open("out.wav", "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
```

## Configuration

Configuration is read from environment variables (prefix `TTS_`) or a `.env` file. See [`.env.example`](./.env.example).

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_DEVICE`        | `cpu`   | Inference device: `auto`, `cpu`, or `cuda`. Only CPU is used by the distributed image; CUDA requires a local install with a GPU runtime. |
| `TTS_VOICE_CACHE_DIR` | `./voices` | Directory where voice models are downloaded and cached. Mount a volume/PVC here in production. |
| `TTS_MAX_LOADED_VOICES` | `4`   | Size of the in-memory LRU cache of loaded voice models. |
| `TTS_HOST`          | `0.0.0.0` | Bind host for the HTTP server. |
| `TTS_PORT`          | `8000`  | Bind port for the HTTP server. |
| `TTS_VOICES_LIST_TTL_SECONDS` | `3600` | How long to cache the `/v1/audio/voices` listing before re-querying HuggingFace. |

## API reference

| Method & path            | Description |
|--------------------------|-------------|
| `POST /v1/audio/speech`  | Synthesize speech; streams a WAV response (`audio/wav`). Body: `{ model, input, voice?, response_format? }`. |
| `GET  /v1/audio/voices`  | List available voices (HuggingFace discovery). Optional query param `refresh=1` to bypass the cache. |
| `GET  /healthz`          | Liveness/readiness probe; returns `{"status":"ok"}` when up. |
| `GET  /metrics`          | Prometheus text exposition of request counts and synthesis latency. |

## Development

```bash
pip install -e ".[dev]"
ruff check .     # lint
pytest -q
```

## License

This project is licensed under the [GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0-or-later), to stay consistent with [Piper](https://github.com/OHF-Voice/piper1-gpl). See [`LICENSE`](./LICENSE) for the full text.

