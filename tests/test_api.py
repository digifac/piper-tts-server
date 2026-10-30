"""API tests using a mocked Piper voice (no model downloads, no real inference)."""

from __future__ import annotations

import struct
import wave

from fastapi.testclient import TestClient

from piper_tts_server.app import AppContext, create_app


class FakeChunk:
    """Mimics the chunk objects yielded by ``PiperVoice.synthesize``."""

    sample_rate = 22050
    sample_width = 2
    sample_channels = 1

    def __init__(self, pcm: bytes) -> None:
        self.audio_int16_bytes = pcm


class FakeVoice:
    """Stand-in for ``PiperVoice`` that produces deterministic sine-ish PCM."""

    def synthesize(self, text: str):
        # Two chunks of 2205 Hz int16 mono (a few hundred samples each).
        n = 4410 // 2
        pcm_a = struct.pack(f"<{n}d", *([1000] * n))
        pcm_b = struct.pack(f"<{n}d", *([-500] * n))
        yield FakeChunk(pcm_a)
        yield FakeChunk(pcm_b)

    def synthesize_wav(self, text: str, wav_file):
        with wave.open(wav_file, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * 4410)


def _make_client() -> tuple[TestClient, AppContext]:
    ctx = AppContext.__new__(AppContext)  # bypass __init__ (no settings/device/manager construction)
    from piper_tts_server.config import Settings

    ctx.settings = Settings(voice_cache_dir="./voices")
    ctx.device_resolver = None  # type: ignore[assignment]
    ctx.voice_manager = _FakeVoiceManager()
    from piper_tts_server.synthesis import SynthesisService

    ctx.synthesis = SynthesisService()
    return TestClient(create_app(ctx)), ctx


class _FakeVoiceManager:
    def list_voices(self, refresh: bool = False):
        from piper_tts_server.voices import VoiceInfo

        return [
            VoiceInfo(name="en_US-lessac-medium", language="en_US", quality="medium"),
            VoiceInfo(name="ar_JO-kareem-low", language="ar_JO", quality="low"),
        ]

    def get_voice(self, name: str):
        if name not in {"en_US-lessac-medium", "fr_FR-upmc-medium"}:
            raise FileNotFoundError(f"no such voice: {name}")
        return FakeVoice()


def test_healthz() -> None:
    client, _ = _make_client()
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_voices_default_simple_shape() -> None:
    client, _ = _make_client()
    resp = client.get("/v1/audio/voices")
    assert resp.status_code == 200
    # Default is a plain array of voice ids (the shape llama-swap's playground parses).
    body = resp.json()
    assert isinstance(body, list)
    assert "en_US-lessac-medium" in body
    assert "ar_JO-kareem-low" in body


def test_list_voices_openai_shape() -> None:
    client, _ = _make_client()
    resp = client.get("/v1/audio/voices", params={"format": "openai"})
    assert resp.status_code == 200
    body = resp.json()
    # OpenAI list wrapper.
    assert set(body) == {"object", "data", "has_more"}
    assert body["object"] == "list"
    assert body["has_more"] is False

    names = {v["name"] for v in body["data"]}
    assert "en_US-lessac-medium" in names

    entry = next(v for v in body["data"] if v["name"] == "ar_JO-kareem-low")
    # OpenAI Voice fields.
    assert entry["id"] == "ar_JO-kareem-low"
    assert entry["object"] == "audio.voice"
    assert entry["created_at"] == 0
    # Piper extras; language must be the bare xx_YY code (no speaker name).
    assert entry["language"] == "ar_JO"
    assert entry["quality"] == "low"


def test_speech_streams_valid_wav() -> None:
    client, _ = _make_client()
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "en_US-lessac-medium", "input": "Hello world."},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"

    data = resp.content
    # Must start with a RIFF/WAVE header.
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    # The PCM payload (after the 44-byte canonical header) must be non-empty.
    pcm = data[44:]
    assert len(pcm) > 0
    assert len(pcm) % 2 == 0


def test_speech_unknown_voice_returns_404() -> None:
    client, _ = _make_client()
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "xx_XX-doesnotexist-medium", "input": "hi"},
    )
    assert resp.status_code == 404


def test_speech_bad_response_format_returns_400() -> None:
    client, _ = _make_client()
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "en_US-lessac-medium", "input": "hi", "response_format": "mp3"},
    )
    assert resp.status_code == 400


def test_speech_empty_input_rejected() -> None:
    client, _ = _make_client()
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "en_US-lessac-medium", "input": ""},
    )
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_counters() -> None:
    client, _ = _make_client()
    client.get("/healthz")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "piper_tts_requests_total" in text
