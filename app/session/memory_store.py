"""In-process session store.

Suitable for a single uvicorn worker (development, or a pinned-instance
deployment). Expired sessions are swept by a background task so a flood of
abandoned WebSockets cannot leak memory.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.core.config import SessionSettings
from app.core.errors import CapacityExceededError
from app.domain.models import VoiceSession
from app.session.store import SessionStore

logger = logging.getLogger(__name__)


class InMemorySessionStore(SessionStore):
    name = "memory"

    def __init__(self, settings: SessionSettings) -> None:
        self._settings = settings
        self._sessions: dict[str, tuple[VoiceSession, float]] = {}
        self._lock = asyncio.Lock()
        self._sweeper: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._sweeper is None and self._settings.sweep_interval_seconds > 0:
            self._sweeper = asyncio.create_task(self._sweep_loop(), name="session-sweep")

    async def create(self) -> VoiceSession:
        session = VoiceSession()

        async with self._lock:
            self._purge_expired()

            if len(self._sessions) >= self._settings.max_active_sessions:
                raise CapacityExceededError(
                    f"Active session limit reached "
                    f"({self._settings.max_active_sessions})"
                )

            self._sessions[session.session_id] = (session, self._expiry())

        return session

    async def get(self, session_id: str) -> VoiceSession | None:
        async with self._lock:
            entry = self._sessions.get(session_id)

            if entry is None:
                return None

            session, expires_at = entry
            if expires_at <= time.monotonic():
                del self._sessions[session_id]
                return None

            return session

    async def save(self, session: VoiceSession) -> None:
        session.touch()
        async with self._lock:
            self._sessions[session.session_id] = (session, self._expiry())

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def active_count(self) -> int:
        async with self._lock:
            self._purge_expired()
            return len(self._sessions)

    async def aclose(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None

        async with self._lock:
            self._sessions.clear()

    def _expiry(self) -> float:
        return time.monotonic() + self._settings.ttl_seconds

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, (_, expires_at) in self._sessions.items()
            if expires_at <= now
        ]
        for session_id in expired:
            del self._sessions[session_id]

        if expired:
            logger.info("Swept expired sessions", extra={"count": len(expired)})

    async def _sweep_loop(self) -> None:
        interval = self._settings.sweep_interval_seconds
        while True:
            try:
                await asyncio.sleep(interval)
                async with self._lock:
                    self._purge_expired()
            except asyncio.CancelledError:
                raise
            except Exception:  # never let the sweeper die silently
                logger.exception("Session sweep failed")
