"""Resolve the configured conversation agent.

Keeps ``app.container`` free of import-path plumbing and gives a single, clear
error when a hybrid RAG implementation is misconfigured.
"""

from __future__ import annotations

import inspect
import logging
from importlib import import_module

from app.core.config import AgentBackend, AgentSettings
from app.core.errors import ConfigurationError
from app.services.agent.base import ConversationAgent, EchoAgent

logger = logging.getLogger(__name__)


async def build_agent(settings: AgentSettings) -> ConversationAgent:
    if settings.backend is AgentBackend.ECHO:
        logger.warning(
            "Using the placeholder echo agent; set AGENT__BACKEND=hybrid_rag "
            "and AGENT__IMPORT_PATH once the RAG pipeline is ready"
        )
        return EchoAgent()

    if not settings.import_path:
        raise ConfigurationError(
            "AGENT__BACKEND=hybrid_rag requires AGENT__IMPORT_PATH "
            "(e.g. 'my_rag.pipeline:build_agent')"
        )

    agent = await _load(settings.import_path, settings)

    if not isinstance(agent, ConversationAgent):
        raise ConfigurationError(
            f"{settings.import_path} produced {type(agent).__name__}, "
            "which does not implement ConversationAgent"
        )

    logger.info("Conversation agent loaded", extra={"agent": agent.name})
    return agent


async def _load(import_path: str, settings: AgentSettings) -> object:
    module_name, _, attribute = import_path.partition(":")

    if not module_name or not attribute:
        raise ConfigurationError(
            f"Invalid AGENT__IMPORT_PATH {import_path!r}; expected 'module:attribute'"
        )

    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ConfigurationError(f"Cannot import {module_name!r}: {exc}") from exc

    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise ConfigurationError(
            f"{module_name!r} has no attribute {attribute!r}"
        ) from exc

    result = target(settings) if _accepts_settings(target) else target()

    if inspect.isawaitable(result):
        result = await result

    return result


def _accepts_settings(target: object) -> bool:
    try:
        signature = inspect.signature(target)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return len(signature.parameters) > 0
