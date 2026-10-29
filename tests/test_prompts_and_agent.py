"""Asset and wiring checks that guard the onboarding flow."""

from __future__ import annotations

import pytest

from app.audio.prompts import PROMPT_MANIFEST, AudioPromptLibrary
from app.core.config import AgentBackend, AgentSettings, AudioSettings, Settings
from app.core.errors import ConfigurationError, PromptNotFoundError
from app.domain.enums import Language, PromptKey
from app.services.agent.base import EchoAgent
from app.services.agent.loader import build_agent

pytestmark = pytest.mark.asyncio


async def test_every_declared_prompt_ships_with_the_repo() -> None:
    """Fails the build if a prompt file is missing, renamed or corrupt."""

    library = AudioPromptLibrary(Settings().audio)
    await library.load()

    assert library.is_loaded
    assert len(library.describe()) == len(PROMPT_MANIFEST)

    for key, language in PROMPT_MANIFEST:
        clip = library.get(key, language)
        assert clip.size_bytes > 0
        assert clip.duration_seconds > 0
        assert clip.content_type == "audio/wav"


async def test_confirmation_prompts_exist_for_every_language() -> None:
    library = AudioPromptLibrary(Settings().audio)
    await library.load()

    for language in Language:
        assert library.has(PromptKey.LANGUAGE_CONFIRM, language)


async def test_prompt_lookup_falls_back_to_the_neutral_recording() -> None:
    library = AudioPromptLibrary(Settings().audio)
    await library.load()

    neutral = library.get(PromptKey.WELCOME)
    per_language = library.get(PromptKey.WELCOME, Language.SINHALA)

    assert per_language.checksum == neutral.checksum


async def test_strict_loading_fails_fast_on_a_missing_directory(tmp_path) -> None:
    library = AudioPromptLibrary(AudioSettings(assets_dir=tmp_path))

    with pytest.raises(PromptNotFoundError):
        await library.load()


async def test_lenient_loading_degrades_instead_of_crashing(tmp_path) -> None:
    library = AudioPromptLibrary(
        AudioSettings(assets_dir=tmp_path, strict_prompt_loading=False)
    )

    await library.load()

    assert library.is_loaded
    assert library.describe() == []


async def test_default_backend_is_the_echo_placeholder() -> None:
    agent = await build_agent(AgentSettings())

    assert isinstance(agent, EchoAgent)
    assert (await agent.health())["ready"] is True


async def test_hybrid_rag_backend_requires_an_import_path() -> None:
    settings = AgentSettings(backend=AgentBackend.HYBRID_RAG)

    with pytest.raises(ConfigurationError, match="AGENT__IMPORT_PATH"):
        await build_agent(settings)


async def test_hybrid_rag_import_path_is_validated() -> None:
    settings = AgentSettings(
        backend=AgentBackend.HYBRID_RAG, import_path="nope.module:factory"
    )

    with pytest.raises(ConfigurationError, match="Cannot import"):
        await build_agent(settings)


async def test_loader_rejects_a_non_agent_object() -> None:
    settings = AgentSettings(
        backend=AgentBackend.HYBRID_RAG, import_path="builtins:dict"
    )

    with pytest.raises(ConfigurationError, match="does not implement"):
        await build_agent(settings)


async def test_loader_accepts_a_custom_agent(monkeypatch) -> None:
    import sys
    import types

    module = types.ModuleType("fake_rag")

    class FakeHybridAgent(EchoAgent):
        name = "fake-hybrid-rag"

    module.build_agent = lambda settings: FakeHybridAgent()
    monkeypatch.setitem(sys.modules, "fake_rag", module)

    agent = await build_agent(
        AgentSettings(
            backend=AgentBackend.HYBRID_RAG, import_path="fake_rag:build_agent"
        )
    )

    assert agent.name == "fake-hybrid-rag"
