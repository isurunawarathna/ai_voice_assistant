"""Audio decoding, validation and encoding.

This is the only module that knows about audio containers. Both the inbound
microphone frames and the pre-recorded prompts go through it, so every byte the
service touches is validated the same way.

All functions here are CPU-bound and synchronous; callers run them in a worker
thread (see :mod:`app.audio.pipeline`).
"""

from __future__ import annotations

import hashlib
import io
import logging
import wave
from enum import StrEnum

import numpy as np
from scipy.signal import resample_poly

from app.core.config import AudioSettings
from app.core.errors import (
    AudioTooLargeError,
    AudioTooShortError,
    AudioValidationError,
)
from app.domain.models import AudioClip, DecodedAudio

logger = logging.getLogger(__name__)

_INT16_SCALE = 32768.0


class AudioContainer(StrEnum):
    WAV = "wav"
    WEBM = "webm"
    OGG = "ogg"
    MP4 = "mp4"
    MP3 = "mp3"
    UNKNOWN = "unknown"


def detect_container(data: bytes) -> AudioContainer:
    """Identify the container from its magic bytes."""

    if len(data) < 12:
        return AudioContainer.UNKNOWN
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return AudioContainer.WAV
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return AudioContainer.WEBM
    if data[:4] == b"OggS":
        return AudioContainer.OGG
    if data[4:8] == b"ftyp":
        return AudioContainer.MP4
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return AudioContainer.MP3
    return AudioContainer.UNKNOWN


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples

    gcd = np.gcd(source_rate, target_rate)
    resampled = resample_poly(samples, target_rate // gcd, source_rate // gcd)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 PCM as a 16-bit WAV container."""

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * (_INT16_SCALE - 1)).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def decode(data: bytes, settings: AudioSettings) -> DecodedAudio:
    """Validate and decode an audio payload to mono float32 at the target rate.

    Raises:
        AudioTooLargeError / AudioTooShortError / AudioValidationError
    """

    if not data:
        raise AudioValidationError("Audio payload is empty")

    if len(data) > settings.max_bytes:
        raise AudioTooLargeError(
            f"Audio payload is {len(data)} bytes, limit is {settings.max_bytes}"
        )

    container = detect_container(data)

    if container is AudioContainer.WAV:
        samples, source_rate, channels = _decode_wav(data)
    elif settings.allow_compressed_input and container is not AudioContainer.UNKNOWN:
        samples, source_rate, channels = _decode_with_pyav(data, container)
    else:
        raise AudioValidationError(
            f"Unsupported audio container: {container.value}",
            client_message="Unsupported audio format. Send 16-bit PCM WAV.",
        )

    if samples.size == 0:
        raise AudioTooShortError("Decoded audio contains no samples")

    duration = samples.size / source_rate

    if duration > settings.max_seconds:
        raise AudioTooLargeError(
            f"Audio is {duration:.1f}s, limit is {settings.max_seconds:.1f}s",
            client_message="The recording is too long.",
        )

    if duration < settings.min_seconds:
        raise AudioTooShortError(
            f"Audio is {duration:.3f}s, minimum is {settings.min_seconds:.3f}s"
        )

    resampled = False
    if source_rate != settings.target_sample_rate:
        if not settings.allow_resampling:
            raise AudioValidationError(
                f"Expected {settings.target_sample_rate} Hz, got {source_rate} Hz",
                client_message=(
                    f"Audio must be {settings.target_sample_rate} Hz 16-bit PCM."
                ),
            )
        samples = resample(samples, source_rate, settings.target_sample_rate)
        resampled = True

    return DecodedAudio(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=settings.target_sample_rate,
        duration_seconds=duration,
        source_sample_rate=source_rate,
        channels=channels,
        resampled=resampled,
    )


def load_clip(data: bytes, settings: AudioSettings) -> AudioClip:
    """Validate an outbound WAV payload and describe it for the wire."""

    decoded = decode(data, settings)
    return AudioClip(
        data=data,
        content_type="audio/wav",
        sample_rate=decoded.source_sample_rate,
        duration_seconds=decoded.duration_seconds,
        checksum=checksum(data),
    )


def _decode_wav(data: bytes) -> tuple[np.ndarray, int, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise AudioValidationError("Compressed WAV is not supported")

            sample_width = handle.getsampwidth()
            if sample_width != 2:
                raise AudioValidationError(
                    f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit"
                )

            channels = handle.getnchannels()
            if channels not in (1, 2):
                raise AudioValidationError("Only mono or stereo WAV is supported")

            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except AudioValidationError:
        raise
    except wave.Error as exc:
        raise AudioValidationError(f"Malformed WAV container: {exc}") from exc

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / _INT16_SCALE

    if channels == 2:
        usable = samples.size - (samples.size % 2)
        samples = samples[:usable].reshape(-1, 2).mean(axis=1)

    return samples, sample_rate, channels


def _decode_with_pyav(
    data: bytes, container: AudioContainer
) -> tuple[np.ndarray, int, int]:
    try:
        import av  # imported lazily: only compressed input needs it
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AudioValidationError(
            "Compressed audio support requires PyAV",
            client_message="Unsupported audio format. Send 16-bit PCM WAV.",
        ) from exc

    try:
        with av.open(io.BytesIO(data)) as handle:
            stream = next((s for s in handle.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioValidationError("No audio stream found")

            resampler = av.AudioResampler(format="flt", layout="mono")
            sample_rate = stream.rate or 0
            blocks: list[np.ndarray] = []

            for frame in handle.decode(stream):
                for converted in resampler.resample(frame):
                    sample_rate = converted.sample_rate or sample_rate
                    blocks.append(converted.to_ndarray().reshape(-1))
    except AudioValidationError:
        raise
    except Exception as exc:  # PyAV raises a wide range of errors
        logger.debug("PyAV decode failed for %s container", container.value)
        raise AudioValidationError(f"Could not decode {container.value}: {exc}") from exc

    if not blocks or not sample_rate:
        raise AudioTooShortError("Decoded audio contains no samples")

    return np.concatenate(blocks).astype(np.float32), sample_rate, 1
