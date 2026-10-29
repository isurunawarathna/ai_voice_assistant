"""Speech-to-text port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.enums import Language
from app.domain.models import DecodedAudio, Transcription


class SpeechToTextEngine(ABC):
    """Transcribes decoded audio. Implementations must be concurrency-safe."""

    name: str = "stt"

    @property
    def is_ready(self) -> bool:
        return True

    @abstractmethod
    async def transcribe(
        self,
        audio: DecodedAudio,
        *,
        language: Language | None = None,
        prompt: str | None = None,
    ) -> Transcription:
        """Transcribe one utterance.

        Args:
            audio: mono float32 PCM at the engine's expected sample rate.
            language: forces decoding in this language; ``None`` auto-detects.
            prompt: optional decoder bias (e.g. expected vocabulary).
        """

    async def warmup(self) -> None:
        """Load weights and run a throwaway inference. Called at startup."""

    async def aclose(self) -> None:
        """Release model resources."""
