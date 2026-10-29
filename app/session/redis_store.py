"""Redis-backed session store for multi-worker / multi-instance deployments.

Sessions are stored as JSON under ``{prefix}{session_id}`` with a TTL, and
indexed in a sorted set scored by absolute expiry so the active-session count
(and the capacity limit) is accurate across every instance.
"""

from __future__ import annotations

import logging
import time

from app.core.config import SessionSettings
from app.core.errors import CapacityExceededError, ConfigurationError
from app.domain.models import VoiceSession
from app.session.store import SessionStore

logger = logging.getLogger(__name__)


class RedisSessionStore(SessionStore):
    name = "redis"

    def __init__(self, settings: SessionSettings) -> None:
        self._settings = settings
        self._client = None

    @property
    def _index_key(self) -> str:
        return f"{self._settings.redis_key_prefix}index"

    def _key(self, session_id: str) -> str:
        return f"{self._settings.redis_key_prefix}{session_id}"

    async def start(self) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ConfigurationError(
                "SESSION__BACKEND=redis requires the 'redis' package"
            ) from exc

        self._client = Redis.from_url(
            self._settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30,
        )

        await self._client.ping()
        logger.info("Redis session store connected")

    async def create(self) -> VoiceSession:
        client = self._require_client()
        await self._prune_index()

        active = await client.zcard(self._index_key)
        if active >= self._settings.max_active_sessions:
            raise CapacityExceededError(
                f"Active session limit reached ({self._settings.max_active_sessions})"
            )

        session = VoiceSession()
        await self.save(session)
        return session

    async def get(self, session_id: str) -> VoiceSession | None:
        client = self._require_client()
        raw = await client.get(self._key(session_id))

        if raw is None:
            await client.zrem(self._index_key, session_id)
            return None

        try:
            return VoiceSession.model_validate_json(raw)
        except ValueError:
            logger.warning(
                "Discarding unreadable session payload",
                extra={"stored_session_id": session_id},
            )
            await self.delete(session_id)
            return None

    async def save(self, session: VoiceSession) -> None:
        client = self._require_client()
        session.touch()
        ttl = self._settings.ttl_seconds

        pipeline = client.pipeline(transaction=True)
        pipeline.set(self._key(session.session_id), session.model_dump_json(), ex=ttl)
        pipeline.zadd(self._index_key, {session.session_id: time.time() + ttl})
        await pipeline.execute()

    async def delete(self, session_id: str) -> None:
        client = self._require_client()
        pipeline = client.pipeline(transaction=True)
        pipeline.delete(self._key(session_id))
        pipeline.zrem(self._index_key, session_id)
        await pipeline.execute()

    async def active_count(self) -> int:
        client = self._require_client()
        await self._prune_index()
        return int(await client.zcard(self._index_key))

    async def health(self) -> dict[str, object]:
        try:
            client = self._require_client()
            await client.ping()
            return {
                "name": self.name,
                "ready": True,
                "active": await self.active_count(),
            }
        except Exception as exc:
            return {"name": self.name, "ready": False, "error": str(exc)}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _prune_index(self) -> None:
        client = self._require_client()
        await client.zremrangebyscore(self._index_key, "-inf", time.time())

    def _require_client(self):
        if self._client is None:
            raise ConfigurationError("Redis session store is not started")
        return self._client
