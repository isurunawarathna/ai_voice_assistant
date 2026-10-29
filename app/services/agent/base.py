"""Conversation agent port — the seam where hybrid RAG plugs in.

The orchestrator knows nothing about retrieval, embeddings, rerankers or LLMs.
It hands a :class:`~app.domain.models.AgentRequest` (query + language +
bounded history + trace id) to whatever implements this interface and sends the
resulting :class:`~app.domain.models.AgentReply` back to the student.

To wire in your hybrid RAG pipeline:

1. Implement ``ConversationAgent`` (keyword + vector retrieval, fusion,
   rerank, generation — all internal to your class).
2. Expose a zero-argument or settings-taking factory, e.g.
   ``def build_agent(settings: AgentSettings) -> ConversationAgent``.
3. Point configuration at it::

       AGENT__BACKEND=hybrid_rag
       AGENT__IMPORT_PATH=my_rag.pipeline:build_agent

Nothing else in the application has to change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from app.domain.models import AgentDelta, AgentReply, AgentRequest

logger = logging.getLogger(__name__)


class ConversationAgent(ABC):
    """Answers a student question, optionally with retrieved evidence."""

    name: str = "agent"

    @property
    def supports_streaming(self) -> bool:
        return False

    @abstractmethod
    async def answer(self, request: AgentRequest) -> AgentReply:
        """Produce a complete answer. Must not raise for ordinary failures —
        raise :class:`app.core.errors.AgentError` so the caller can report a
        safe message to the student."""

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentDelta]:
        """Yield partial answers. Defaults to a single chunk from ``answer``."""

        reply = await self.answer(request)
        yield AgentDelta(text=reply.text, finished=True)

    async def warmup(self) -> None:
        """Open connections, load indices, verify credentials."""

    async def health(self) -> dict[str, object]:
        """Readiness detail surfaced by ``/health/ready``."""

        return {"name": self.name, "ready": True}

    async def aclose(self) -> None:
        """Close clients and release index handles."""


class EchoAgent(ConversationAgent):
    """Placeholder agent used until the hybrid RAG pipeline is connected.

    It confirms the end-to-end path (audio -> STT -> agent -> client) without
    pretending to answer anything, and it is deliberately obvious in the
    transcript so it can never be mistaken for a real answer in production.
    """

    name = "echo"

    _TEMPLATES: ClassVar[dict[str, str]] = {
        "si-LK": "ඔබ පැවසුවේ: {text}",
        "en-US": "I heard: {text}",
    }

    async def answer(self, request: AgentRequest) -> AgentReply:
        template = self._TEMPLATES.get(
            request.language.value, "I heard: {text}"
        )
        return AgentReply(
            text=template.format(text=request.query),
            language=request.language,
            metadata={"agent": self.name, "placeholder": True},
        )

    async def health(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ready": True,
            "note": "placeholder agent; hybrid RAG not connected",
        }
