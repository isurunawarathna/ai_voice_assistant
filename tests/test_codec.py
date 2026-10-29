from __future__ import annotations

import pytest

from app.audio import codec
from app.core.config import AudioSettings
from app.core.errors import AudioTooLargeError, AudioTooShortError, AudioValidationError
from tests.conftest import make_wav


@pytest.fixture
def audio_settings() -> AudioSettings:
    return AudioSettings()


def test_decodes_mono_wav(audio_settings: AudioSettings) -> None:
    decoded = codec.decode(make_wav(1.0), audio_settings)

    assert decoded.sample_rate == 16_000
    assert decoded.channels == 1
    assert decoded.resampled is False
    assert decoded.samples.dtype.name == "float32"
    assert decoded.duration_seconds == pytest.approx(1.0, abs=0.01)
    assert abs(decoded.samples).max() <= 1.0


def test_downmixes_stereo(audio_settings: AudioSettings) -> None:
    decoded = codec.decode(make_wav(0.5, channels=2), audio_settings)

    assert decoded.channels == 2
    assert decoded.duration_seconds == pytest.approx(0.5, abs=0.01)


def test_resamples_to_target_rate(audio_settings: AudioSettings) -> None:
    decoded = codec.decode(make_wav(1.0, sample_rate=44_100), audio_settings)

    assert decoded.resampled is True
    assert decoded.sample_rate == 16_000
    assert decoded.source_sample_rate == 44_100
    assert decoded.samples.size == pytest.approx(16_000, rel=0.01)


def test_rejects_resampling_when_disabled() -> None:
    settings = AudioSettings(allow_resampling=False)

    with pytest.raises(AudioValidationError):
        codec.decode(make_wav(0.5, sample_rate=8_000), settings)


def test_rejects_empty_payload(audio_settings: AudioSettings) -> None:
    with pytest.raises(AudioValidationError):
        codec.decode(b"", audio_settings)


def test_rejects_unknown_container(audio_settings: AudioSettings) -> None:
    with pytest.raises(AudioValidationError):
        codec.decode(b"not audio at all, just bytes", audio_settings)


def test_rejects_oversized_payload() -> None:
    settings = AudioSettings(max_bytes=1_000)

    with pytest.raises(AudioTooLargeError):
        codec.decode(make_wav(1.0), settings)


def test_rejects_long_audio() -> None:
    settings = AudioSettings(max_seconds=0.5)

    with pytest.raises(AudioTooLargeError):
        codec.decode(make_wav(2.0), settings)


def test_rejects_short_audio(audio_settings: AudioSettings) -> None:
    with pytest.raises(AudioTooShortError):
        codec.decode(make_wav(0.05), audio_settings)


def test_detects_containers() -> None:
    assert codec.detect_container(make_wav(0.3)) is codec.AudioContainer.WAV
    assert codec.detect_container(b"OggS" + b"\x00" * 16) is codec.AudioContainer.OGG
    assert (
        codec.detect_container(b"\x1a\x45\xdf\xa3" + b"\x00" * 16)
        is codec.AudioContainer.WEBM
    )
    assert codec.detect_container(b"short") is codec.AudioContainer.UNKNOWN


def test_round_trips_through_encode_wav(audio_settings: AudioSettings) -> None:
    decoded = codec.decode(make_wav(0.5), audio_settings)
    reencoded = codec.encode_wav(decoded.samples, decoded.sample_rate)

    again = codec.decode(reencoded, audio_settings)
    assert again.duration_seconds == pytest.approx(decoded.duration_seconds, abs=0.01)


def test_load_clip_describes_payload(audio_settings: AudioSettings) -> None:
    payload = make_wav(0.75)
    clip = codec.load_clip(payload, audio_settings)

    assert clip.size_bytes == len(payload)
    assert clip.content_type == "audio/wav"
    assert clip.duration_seconds == pytest.approx(0.75, abs=0.01)
    assert len(clip.checksum) == 32
