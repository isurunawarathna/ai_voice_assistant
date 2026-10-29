"""Wire protocol for the ``/voice`` WebSocket.

Framing rules
-------------
* Every server -> client **text** frame is one JSON object with a ``type``
  discriminator and a ``v`` protocol version.
* Every server -> client **binary** frame is audio, and is always preceded by
  a JSON frame whose type ends in ``.audio`` describing it (``content_type``,
  ``size_bytes``, ``duration_seconds``, ``checksum``). Clients must therefore
  buffer the header and apply it to the next binary frame.
* Client -> server **binary** frames are WAV recordings. Client -> server
  **text** frames are JSON control messages (see ``InboundMessage``).

Bumping ``PROTOCOL_VERSION`` is a breaking change for the Angular client.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.domain.enums import ConversationState, Language, PromptKey, Speaker

PROTOCOL_VERSION = 1


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


# --------------------------------------------------------------------------- #
# Server -> client
# --------------------------------------------------------------------------- #
class ServerEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    v: int = PROTOCOL_VERSION
    ts: datetime = Field(default_factory=_utcnow)


class SessionCreated(ServerEvent):
    type: Literal["session.created"] = "session.created"
    session_id: str
    state: ConversationState
    supported_languages: list[Language]
    protocol_version: int = PROTOCOL_VERSION


class StateChanged(ServerEvent):
    type: Literal["state.changed"] = "state.changed"
    state: ConversationState
    message: str | None = None


class AudioHeader(ServerEvent):
    """Describes the binary frame that follows immediately after."""

    type: Literal["prompt.audio", "agent.audio"] = "prompt.audio"
    prompt: PromptKey | None = None
    language: Language | None = None
    content_type: str = "audio/wav"
    size_bytes: int
    duration_seconds: float = 0.0
    sample_rate: int = 16_000
    checksum: str = ""


class TranscriptEvent(ServerEvent):
    type: Literal["transcript"] = "transcript"
    text: str
    speaker: Speaker = Speaker.USER
    final: bool = True
    language: Language | None = None
    detected_language: str | None = None
    language_probability: float = 0.0


class LanguageSelected(ServerEvent):
    type: Literal["language.selected"] = "language.selected"
    language: Language
    display_name: str
    state: ConversationState = ConversationState.ACTIVE
    # Additive in protocol v1: how the choice was made and how sure we are.
    method: Literal["voice", "text", "ui"] = "voice"
    confidence: float = 1.0
    reason: str | None = None


class LanguageRejected(ServerEvent):
    type: Literal["language.rejected"] = "language.rejected"
    attempt: int
    attempts_remaining: int
    reason: str
    message: str = "Please say Sinhala or English."


class LanguageFailed(ServerEvent):
    type: Literal["language.failed"] = "language.failed"
    attempts: int
    message: str = "Maximum attempts reached. Please reconnect."


class AgentThinking(ServerEvent):
    type: Literal["agent.thinking"] = "agent.thinking"


class AgentSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: str | None = None
    score: float = 0.0


class AgentMessage(ServerEvent):
    type: Literal["agent.message"] = "agent.message"
    text: str
    language: Language | None = None
    final: bool = True
    sources: list[AgentSource] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentDeltaEvent(ServerEvent):
    type: Literal["agent.delta"] = "agent.delta"
    text: str
    final: bool = False


class ErrorEvent(ServerEvent):
    type: Literal["error"] = "error"
    code: str
    message: str
    fatal: bool = False
    retryable: bool = True


class PongEvent(ServerEvent):
    type: Literal["pong"] = "pong"


class SessionClosed(ServerEvent):
    type: Literal["session.closed"] = "session.closed"
    reason: str


OutboundEvent = Annotated[
    SessionCreated
    | StateChanged
    | AudioHeader
    | TranscriptEvent
    | LanguageSelected
    | LanguageRejected
    | LanguageFailed
    | AgentThinking
    | AgentMessage
    | AgentDeltaEvent
    | ErrorEvent
    | PongEvent
    | SessionClosed,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Client -> server control messages
# --------------------------------------------------------------------------- #
class ClientMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PingMessage(ClientMessage):
    type: Literal["ping"]


class ResetMessage(ClientMessage):
    """Return to language selection without dropping the socket."""

    type: Literal["reset"]


class TextMessage(ClientMessage):
    """Text input instead of audio. Useful for the web client and for testing
    the RAG pipeline without a microphone."""

    type: Literal["text"]
    text: str = Field(min_length=1, max_length=2_000)


class SelectLanguageMessage(ClientMessage):
    """Explicit language choice, e.g. from a UI button rather than speech."""

    type: Literal["select_language"]
    language: Language


class EndMessage(ClientMessage):
    type: Literal["end"]


InboundMessage = Annotated[
    PingMessage | ResetMessage | TextMessage | SelectLanguageMessage | EndMessage,
    Field(discriminator="type"),
]

inbound_adapter: TypeAdapter[InboundMessage] = TypeAdapter(InboundMessage)
