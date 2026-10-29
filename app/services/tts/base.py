"""Text-to-speech port.

The onboarding flow uses pre-recorded prompts, so no TTS engine is required to
run the service. Once answers come from the hybrid RAG agent they have to be
spoken, which is what this port is for: implement it, register it in
``app.container``, and the orchestrator starts sending ``agent.audio`` frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.enums import Language
from app.domain.models import AudioClip


class TextToSpeechEngine(ABC):
    name: str = "tts"

    @property
    def is_available(self) -> bool:
        """Whether synthesised replies should be sent to the client."""

        return True

    @abstractmethod
    async def synthesize(self, text: str, language: Language) -> AudioClip:
        """Render ``text`` as a WAV clip in the caller's language."""

    async def warmup(self) -> None:
        """Pre-load voices/models."""

    async def aclose(self) -> None:
        """Release engine resources."""


class NullTextToSpeech(TextToSpeechEngine):
    """Default engine: text-only replies.

    ``is_available`` is ``False``, so the orchestrator skips synthesis entirely
    instead of emitting empty audio frames.
    """

    name = "null"

    @property
    def is_available(self) -> bool:
        return False

    async def synthesize(self, text: str, language: Language) -> AudioClip:
        raise NotImplementedError(
            "No TTS engine is configured. Set TTS__BACKEND and register an "
            "engine in app.container."
        )
