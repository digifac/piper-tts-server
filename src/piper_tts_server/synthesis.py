"""Streaming WAV synthesis built on top of Piper's chunked ``synthesize`` API."""

from __future__ import annotations

import logging
import struct
import wave
from collections.abc import Generator, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioFormat:
    """WAV format parameters taken from the first Piper chunk."""

    sample_rate: int
    sample_width: int  # bytes per sample (2 for int16)
    channels: int


def _wav_header(fmt: AudioFormat, data_size: int | None = None) -> bytes:
    """Build a RIFF/WAVE header. ``data_size=None`` yields a zero placeholder."""
    if data_size is None:
        data_size = 0
    byte_rate = fmt.sample_rate * fmt.channels * fmt.sample_width
    block_align = fmt.channels * fmt.sample_width
    return (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, fmt.channels, fmt.sample_rate, byte_rate, block_align, fmt.sample_width * 8)
        + b"data" + struct.pack("<I", data_size)
    )


class SynthesisService:
    """Wraps Piper synthesis into streamable WAV output.

    Streaming is the primary path: :meth:`synthesize_stream` yields a WAV header
    followed by raw PCM frames as Piper produces them, giving low time-to-first-byte.
    Blocking Piper work must be run off the event loop (see ``app.py``).
    """

    def synthesize_stream(self, voice, text: str) -> Generator[bytes, None, None]:
        """Yield WAV bytes for ``text`` using ``voice``, streaming chunk by chunk.

        The first yielded item is always the WAV header. Any synthesis error is
        raised *before* the header is produced (Piper validates phonemization and
        runs the encoder before emitting audio), so callers can fail with a clean
        HTTP status code before any response bytes are sent.
        """
        header_sent = False

        for chunk in voice.synthesize(text):
            if not header_sent:
                fmt = AudioFormat(
                    sample_rate=chunk.sample_rate,
                    sample_width=chunk.sample_width,
                    channels=chunk.sample_channels,
                )
                yield _wav_header(fmt)  # data size unknown up front; clients tolerate this for streams
                logger.debug("First audio chunk produced (%d bytes).", len(chunk.audio_int16_bytes))
                header_sent = True

            pcm: bytes = chunk.audio_int16_bytes
            if not pcm:
                continue
            yield pcm

        # Note: the RIFF/data sizes in the header were written as 0 (streaming mode).
        # Most players and the OpenAI SDK accept this; a fully valid file can be built
        # with synthesize_wav_bytes() when total size is known.

    def synthesize_wav_bytes(self, voice, text: str) -> bytes:
        """Return the full WAV file as a single ``bytes`` object (non-streaming).

        Builds the file from Piper's chunks so the RIFF/data sizes are always
        correct. Raises :class:`RuntimeError` when no audio is produced or when
        the resulting duration is suspiciously short (< 0.1 s), which typically
        indicates a GPU memory issue or an empty phonemization result.
        """
        pcm_parts: list[bytes] = []
        fmt: AudioFormat | None = None

        for chunk in voice.synthesize(text):
            if fmt is None:
                fmt = AudioFormat(
                    sample_rate=chunk.sample_rate,
                    sample_width=chunk.sample_width,
                    channels=chunk.sample_channels,
                )
            pcm: bytes = chunk.audio_int16_bytes
            if pcm:
                pcm_parts.append(pcm)

        if fmt is None or not pcm_parts:
            raise RuntimeError("Synthesis produced no audio.")

        data = b"".join(pcm_parts)
        
        # Diagnostic logging
        duration_seconds = len(data) / (fmt.sample_rate * fmt.channels * fmt.sample_width)
        logger.info(
            "Synthesized %d chunks, %d bytes PCM (%.2fs at %d Hz)",
            len(pcm_parts),
            len(data),
            duration_seconds,
            fmt.sample_rate,
        )

        # Sanity check: reject suspiciously short audio (likely GPU OOM or empty phonemization)
        if duration_seconds < 0.1:
            raise RuntimeError(
                f"Synthesis produced abnormally short audio ({duration_seconds:.3f}s). "
                "This may indicate a GPU memory issue or an empty input."
            )

        return _wav_header(fmt, len(data)) + data


def iter_chunks(voice, text: str) -> Iterator[object]:  # pragma: no cover - trivial passthrough
    """Passthrough helper kept for testability of chunk iteration."""
    yield from voice.synthesize(text)
