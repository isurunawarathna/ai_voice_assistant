"""Composition root.

Everything is constructed here, once, at startup — and this is the only place
to touch when the hybrid RAG agent or a real TTS engine is plugged in.
"""

from __future__ import annotations

import logging
import time

from app.api.ws.guard import ConnectionGuard
from app.audio.prompts import AudioPromptLibrary
from app.conversation.orchestrator import ConversationOrchestrator
from app.core.config import SessionBackend, Settings, get_settings
from app.core.metrics import Metrics
from app.services.agent.base import ConversationAgent
from app.services.agent.loader import build_agent
from app.services.language import LanguageDetector
from app.services.stt.base import SpeechToTextEngine
from app.services.stt.faster_whisper_engine import FasterWhisperEngine
from app.services.tts.base import NullTextToSpeech, TextToSpeechEngine
from app.session.memory_store import InMemorySessionStore
from app.session.redis_store import RedisSessionStore
from app.session.store import SessionStore

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Owns every long-lived service and its lifecycle."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.started_at: float | None = None

        self.metrics = Metrics()
        self.guard = ConnectionGuard(self.settings)
        self.prompts = AudioPromptLibrary(self.settings.audio)
        self.language_detector = LanguageDetector()
        self.sessions: SessionStore = self._build_session_store()
        self.stt: SpeechToTextEngine = FasterWhisperEngine(self.settings.stt)
        self.tts: TextToSpeechEngine = NullTextToSpeech()
        self.agent: ConversationAgent | None = None
        self.orchestrator: ConversationOrchestrator | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def startup(self) -> None:
        started = time.perf_counter()

        # Order matters: fail on configuration (prompts, Redis, agent) before
        # spending time loading Whisper weights.
        await self.prompts.load()
        await self.sessions.start()

        self.agent = await build_agent(self.settings.agent)
        await self.agent.warmup()
        await self.tts.warmup()
        await self.stt.warmup()

        self.orchestrator = ConversationOrchestrator(
            settings=self.settings,
            sessions=self.sessions,
            prompts=self.prompts,
            stt=self.stt,
            agent=self.agent,
            tts=self.tts,
            language_detector=self.language_detector,
            metrics=self.metrics,
        )

        self.started_at = time.time()
        logger.info(
            "Service ready",
            extra={
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "environment": self.settings.environment.value,
                "session_backend": self.sessions.name,
                "stt": self.stt.name,
                "agent": self.agent.name,
                "tts": self.tts.name,
            },
        )

    async def shutdown(self) -> None:
        for component in (self.agent, self.tts, self.stt, self.sessions):
            if component is None:
                continue
            try:
                await component.aclose()
            except Exception:
                logger.exception("Error closing %s", type(component).__name__)

        self.orchestrator = None
        logger.info("Service stopped")

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def require_orchestrator(self) -> ConversationOrchestrator:
        if self.orchestrator is None:
            raise RuntimeError("Service container has not completed startup")
        return self.orchestrator

    @property
    def uptime_seconds(self) -> float:
        return 0.0 if self.started_at is None else time.time() - self.started_at

    async def readiness(self) -> dict[str, object]:
        components: dict[str, object] = {
            "prompts": {
                "ready": self.prompts.is_loaded,
                "count": len(self.prompts.describe()),
            },
            "stt": {"name": self.stt.name, "ready": self.stt.is_ready},
            "tts": {"name": self.tts.name, "ready": self.tts.is_available},
            "sessions": await self.sessions.health(),
            "agent": (
                await self.agent.health()
                if self.agent is not None
                else {"ready": False, "error": "not initialised"}
            ),
        }

        ready = (
            self.orchestrator is not None
            and self.prompts.is_loaded
            and self.stt.is_ready
            and bool(components["sessions"].get("ready"))  # type: ignore[union-attr]
            and bool(components["agent"].get("ready"))  # type: ignore[union-attr]
        )

        return {"ready": ready, "components": components}

    def _build_session_store(self) -> SessionStore:
        if self.settings.session.backend is SessionBackend.REDIS:
            return RedisSessionStore(self.settings.session)
        return InMemorySessionStore(self.settings.session)
