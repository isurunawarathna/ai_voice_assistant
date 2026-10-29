"""Session store port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import VoiceSession


class SessionStore(ABC):
    """Persistence for live conversations.

    A single process can keep sessions in memory, but any horizontally scaled
    deployment behind a load balancer needs the Redis store so a reconnecting
    client resumes its language choice and history.
    """

    name: str = "session-store"

    @abstractmethod
    async def create(self) -> VoiceSession:
        """Create and persist a new session.

        Raises:
            CapacityExceededError: when the active-session limit is reached.
        """

    @abstractmethod
    async def get(self, session_id: str) -> VoiceSession | None:
        """Return a session, or ``None`` when it is unknown or expired."""

    @abstractmethod
    async def save(self, session: VoiceSession) -> None:
        """Persist mutations and refresh the TTL."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Remove a session; must be idempotent."""

    @abstractmethod
    async def active_count(self) -> int:
        """Number of non-expired sessions."""

    async def start(self) -> None:
        """Open connections and background maintenance."""

    async def health(self) -> dict[str, object]:
        return {"name": self.name, "ready": True, "active": await self.active_count()}

    async def aclose(self) -> None:
        """Close connections and cancel background tasks."""
