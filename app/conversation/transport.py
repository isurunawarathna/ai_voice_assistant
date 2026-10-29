"""Transport port for the conversation orchestrator.

The orchestrator drives the conversation state machine without knowing it is
talking over a WebSocket. That keeps the flow testable with an in-memory fake
transport, and leaves room for a second channel (SIP/telephony, gRPC) later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from app.domain.models import AudioClip
from app.domain.protocol import AudioHeader, InboundMessage, OutboundEvent


class FrameKind(StrEnum):
    AUDIO = "audio"
    CONTROL = "control"
    INVALID = "invalid"
    DISCONNECT = "disconnect"


@dataclass(slots=True, frozen=True)
class InboundFrame:
    kind: FrameKind
    audio: bytes | None = None
    message: InboundMessage | None = None
    error: str | None = None

    @classmethod
    def audio_frame(cls, payload: bytes) -> InboundFrame:
        return cls(FrameKind.AUDIO, audio=payload)

    @classmethod
    def control_frame(cls, message: InboundMessage) -> InboundFrame:
        return cls(FrameKind.CONTROL, message=message)

    @classmethod
    def invalid_frame(cls, error: str) -> InboundFrame:
        return cls(FrameKind.INVALID, error=error)

    @classmethod
    def disconnect_frame(cls) -> InboundFrame:
        return cls(FrameKind.DISCONNECT)


class VoiceTransport(ABC):
    """Bidirectional channel carrying protocol events and audio frames."""

    @abstractmethod
    async def send_event(self, event: OutboundEvent) -> None:
        """Send one JSON protocol event."""

    @abstractmethod
    async def send_clip(self, header: AudioHeader, clip: AudioClip) -> None:
        """Send an audio header followed by its binary frame, atomically."""

    @abstractmethod
    async def receive(self) -> InboundFrame:
        """Await the next client frame. Never raises on client misbehaviour;
        returns an ``INVALID`` or ``DISCONNECT`` frame instead."""

    @abstractmethod
    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the channel if it is still open. Idempotent."""

    @property
    def peer(self) -> str:
        """Human-readable client identity for logs."""

        return "unknown"
