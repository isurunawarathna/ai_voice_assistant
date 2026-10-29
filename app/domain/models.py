"""Domain models.

``VoiceSession`` is a pydantic model because it is persisted (Redis) as JSON.
The audio/transcription value objects are dataclasses: they live for the
duration of one turn and never leave the process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ConversationState, Language, Speaker

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Turn(BaseModel):
    """One utterance in the conversation history handed to the agent."""

    model_config = ConfigDict(frozen=True)

    speaker: Speaker
    text: str
    created_at: datetime = Field(default_factory=_utcnow)


class VoiceSession(BaseModel):
    """Everything the orchestrator needs to know about one caller."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    state: ConversationState = ConversationState.CONNECTING
    language: Language | None = None
    language_retries: int = 0
    # How the language was chosen: "voice", "text" or "ui". Handed to the agent.
    language_method: str | None = None
    # Onboarding bookkeeping. Persisted so a Redis-backed session keeps the
    # same abuse budget whichever worker handles the next frame.
    selection_started_at: datetime | None = None
    invalid_inputs: int = 0
    resets: int = 0
    onboarding_outcome: str | None = None
    turns: list[Turn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_language_selected(self) -> bool:
        return self.language is not None

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def transition(self, state: ConversationState) -> None:
        self.state = state
        self.touch()

    def add_turn(self, speaker: Speaker, text: str, *, max_turns: int) -> None:
        self.turns.append(Turn(speaker=speaker, text=text))
        if max_turns > 0 and len(self.turns) > max_turns:
            del self.turns[: len(self.turns) - max_turns]
        self.touch()

    def history(self) -> tuple[Turn, ...]:
        return tuple(self.turns)


@dataclass(slots=True, frozen=True)
class AudioClip:
    """Encoded audio ready to be written to the wire."""

    data: bytes
    content_type: str = "audio/wav"
    sample_rate: int = 16_000
    duration_seconds: float = 0.0
    checksum: str = ""

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass(slots=True)
class DecodedAudio:
    """Mono float32 PCM in the range [-1, 1], ready for inference."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    source_sample_rate: int
    channels: int
    resampled: bool = False


@dataclass(slots=True, frozen=True)
class Transcription:
    text: str
    detected_language: str | None = None
    language_probability: float = 0.0
    duration_seconds: float = 0.0
    no_speech_probability: float = 0.0
    segments: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(slots=True, frozen=True)
class LanguageDecision:
    """Outcome of interpreting the caller's language choice."""

    language: Language | None
    confidence: float
    reason: str

    @property
    def resolved(self) -> bool:
        return self.language is not None


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """A single retrieval hit. Populated by the hybrid RAG implementation."""

    id: str
    text: str
    score: float = 0.0
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentRequest:
    """Everything the (future) hybrid RAG agent receives for one question."""

    session_id: str
    language: Language
    query: str
    history: tuple[Turn, ...] = ()
    trace_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentReply:
    """The agent's answer, optionally with the evidence it used."""

    text: str
    chunks: tuple[RetrievedChunk, ...] = ()
    language: Language | None = None
    finished: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentDelta:
    """One streamed fragment of an answer."""

    text: str
    finished: bool = False
