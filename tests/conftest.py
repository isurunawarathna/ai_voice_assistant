"""Shared fixtures and test doubles."""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import Iterable

import numpy as np
import pytest

from app.audio.prompts import AudioPromptLibrary
from app.conversation.orchestrator import ConversationOrchestrator
from app.conversation.transport import InboundFrame, VoiceTransport
from app.core.config import Settings
from app.domain.enums import Language, PromptKey
from app.domain.models import (
    AgentReply,
    AgentRequest,
    AudioClip,
    DecodedAudio,
    Transcription,
)
from app.domain.protocol import AudioHeader, OutboundEvent
from app.services.agent.base import ConversationAgent
from app.services.language import LanguageDetector
from app.services.stt.base import SpeechToTextEngine
from app.services.tts.base import NullTextToSpeech
from app.session.memory_store import InMemorySessionStore


def make_wav(
    duration_seconds: float = 1.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    frequency: float = 220.0,
) -> bytes:
    """Build a valid 16-bit PCM WAV payload."""

    count = int(duration_seconds * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    wave_data = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")

    if channels == 2:
        wave_data = np.repeat(wave_data, 2)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(wave_data.tobytes())
    return buffer.getvalue()


class FakeTransport(VoiceTransport):
    """Records outbound traffic and replays a scripted inbound sequence."""

    def __init__(self, frames: Iterable[InboundFrame] = ()) -> None:
        self.events: list[OutboundEvent] = []
        self.clips: list[tuple[AudioHeader, AudioClip]] = []
        self.closed_with: tuple[int, str] | None = None
        self._inbound: asyncio.Queue[InboundFrame] = asyncio.Queue()

        for frame in frames:
            self._inbound.put_nowait(frame)

    async def send_event(self, event: OutboundEvent) -> None:
        self.events.append(event)

    async def send_clip(self, header: AudioHeader, clip: AudioClip) -> None:
        # Mirror the wire: the header is an ordinary event followed by bytes.
        self.events.append(header)
        self.clips.append((header, clip))

    async def receive(self) -> InboundFrame:
        if self._inbound.empty():
            return InboundFrame.disconnect_frame()
        return await self._inbound.get()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed_with is None:
            self.closed_with = (code, reason)

    def push(self, frame: InboundFrame) -> None:
        self._inbound.put_nowait(frame)

    # -- assertions helpers -------------------------------------------- #
    def event_types(self) -> list[str]:
        return [event.type for event in self.events]

    def first(self, event_type: str) -> OutboundEvent:
        for event in self.events:
            if event.type == event_type:
                return event
        raise AssertionError(f"{event_type} was never sent: {self.event_types()}")

    def has(self, event_type: str) -> bool:
        return any(event.type == event_type for event in self.events)


class ScriptedSTT(SpeechToTextEngine):
    """Returns queued transcriptions, one per call."""

    name = "scripted"

    def __init__(self, transcriptions: Iterable[Transcription | str] = ()) -> None:
        self.queue: list[Transcription] = [
            item if isinstance(item, Transcription) else Transcription(text=item)
            for item in transcriptions
        ]
        self.calls: list[tuple[float, Language | None]] = []

    async def transcribe(
        self,
        audio: DecodedAudio,
        *,
        language: Language | None = None,
        prompt: str | None = None,
    ) -> Transcription:
        self.calls.append((audio.duration_seconds, language))
        if not self.queue:
            return Transcription(text="")
        return self.queue.pop(0)


class StubAgent(ConversationAgent):
    name = "stub"

    def __init__(self, reply: str = "Registration closes on 30 June.") -> None:
        self.reply = reply
        self.requests: list[AgentRequest] = []

    async def answer(self, request: AgentRequest) -> AgentReply:
        self.requests.append(request)
        return AgentReply(text=self.reply, language=request.language)


class StubPrompts(AudioPromptLibrary):
    """Prompt library backed by generated audio instead of the asset files."""

    def __init__(self, settings) -> None:
        super().__init__(settings)
        clip = AudioClip(data=make_wav(0.5), duration_seconds=0.5, checksum="stub")
        self._clips = {
            (PromptKey.WELCOME, None): clip,
            (PromptKey.LANGUAGE_RETRY, None): clip,
            (PromptKey.LANGUAGE_CONFIRM, Language.SINHALA): clip,
            (PromptKey.LANGUAGE_CONFIRM, Language.ENGLISH): clip,
        }
        self._loaded = True

    async def load(self) -> None:  # already populated
        return None


@pytest.fixture
def settings() -> Settings:
    overrides = Settings()
    overrides.conversation.idle_timeout_seconds = 5.0
    overrides.conversation.language_selection_timeout_seconds = 5.0
    overrides.conversation.agent_timeout_seconds = 5.0
    overrides.session.sweep_interval_seconds = 0
    return overrides


@pytest.fixture
def sessions(settings: Settings) -> InMemorySessionStore:
    return InMemorySessionStore(settings.session)


@pytest.fixture
def stt() -> ScriptedSTT:
    return ScriptedSTT()


@pytest.fixture
def agent() -> StubAgent:
    return StubAgent()


@pytest.fixture
def orchestrator(
    settings: Settings,
    sessions: InMemorySessionStore,
    stt: ScriptedSTT,
    agent: StubAgent,
) -> ConversationOrchestrator:
    return ConversationOrchestrator(
        settings=settings,
        sessions=sessions,
        prompts=StubPrompts(settings.audio),
        stt=stt,
        agent=agent,
        tts=NullTextToSpeech(),
        language_detector=LanguageDetector(),
    )
