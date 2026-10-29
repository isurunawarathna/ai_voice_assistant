"""End-to-end conversation flow tests against a fake transport."""

from __future__ import annotations

import pytest

from app.conversation.orchestrator import (
    CLOSE_POLICY_VIOLATION,
    ConversationOrchestrator,
)
from app.conversation.transport import InboundFrame
from app.domain.enums import ConversationState, Language, PromptKey, Speaker
from app.domain.models import Transcription
from app.domain.protocol import (
    PingMessage,
    ResetMessage,
    SelectLanguageMessage,
    TextMessage,
)
from app.session.memory_store import InMemorySessionStore
from tests.conftest import FakeTransport, ScriptedSTT, StubAgent, make_wav

pytestmark = pytest.mark.asyncio


async def test_greets_and_asks_for_language(
    orchestrator: ConversationOrchestrator,
) -> None:
    transport = FakeTransport()

    await orchestrator.run(transport)

    assert transport.event_types()[:4] == [
        "session.created",
        "state.changed",
        "prompt.audio",
        "state.changed",
    ]

    created = transport.first("session.created")
    assert created.supported_languages == [Language.SINHALA, Language.ENGLISH]
    assert created.protocol_version == 1

    # The welcome clip is sent before selection is requested.
    header, clip = transport.clips[0]
    assert header.prompt == PromptKey.WELCOME
    assert header.size_bytes == clip.size_bytes

    states = [e.state for e in transport.events if e.type == "state.changed"]
    assert states == [ConversationState.WELCOME, ConversationState.LANGUAGE_SELECTION]


async def test_spoken_language_selection_then_question(
    orchestrator: ConversationOrchestrator,
    stt: ScriptedSTT,
    agent: StubAgent,
) -> None:
    stt.queue = [
        Transcription(text="Sinhala", detected_language="en", language_probability=0.9),
        Transcription(text="ලියාපදිංචිය කවදාද", detected_language="si"),
    ]
    transport = FakeTransport(
        [
            InboundFrame.audio_frame(make_wav(1.0)),
            InboundFrame.audio_frame(make_wav(2.0)),
        ]
    )

    await orchestrator.run(transport)

    selected = transport.first("language.selected")
    assert selected.language == Language.SINHALA
    assert selected.display_name == "Sinhala"

    # Confirmation audio is played in the chosen language.
    confirm = next(
        h for h, _ in transport.clips if h.prompt == PromptKey.LANGUAGE_CONFIRM
    )
    assert confirm.language == Language.SINHALA

    assert transport.has("agent.thinking")
    message = transport.first("agent.message")
    assert message.text == agent.reply
    assert message.language == Language.SINHALA

    # The question is transcribed in the selected language, not auto-detected.
    assert stt.calls[0][1] is None
    assert stt.calls[1][1] is Language.SINHALA

    request = agent.requests[0]
    assert request.query == "ලියාපදිංචිය කවදාද"
    assert request.language is Language.SINHALA
    assert request.history == ()


async def test_retries_then_fails_after_max_attempts(
    orchestrator: ConversationOrchestrator, stt: ScriptedSTT, settings
) -> None:
    attempts = settings.conversation.language_max_retries
    stt.queue = [Transcription(text="what?") for _ in range(attempts)]
    transport = FakeTransport(
        [InboundFrame.audio_frame(make_wav(1.0)) for _ in range(attempts)]
    )

    await orchestrator.run(transport)

    rejections = [e for e in transport.events if e.type == "language.rejected"]
    assert len(rejections) == attempts - 1
    assert rejections[0].attempt == 1
    assert rejections[0].attempts_remaining == attempts - 1

    # Every rejection replays the retry prompt.
    retries = [h for h, _ in transport.clips if h.prompt == PromptKey.LANGUAGE_RETRY]
    assert len(retries) == attempts - 1

    failure = transport.first("language.failed")
    assert failure.attempts == attempts
    assert transport.closed_with == (CLOSE_POLICY_VIOLATION, "language selection failed")


async def test_recovers_after_one_failed_attempt(
    orchestrator: ConversationOrchestrator, stt: ScriptedSTT
) -> None:
    stt.queue = [Transcription(text="mumble"), Transcription(text="English")]
    transport = FakeTransport(
        [
            InboundFrame.audio_frame(make_wav(1.0)),
            InboundFrame.audio_frame(make_wav(1.0)),
        ]
    )

    await orchestrator.run(transport)

    assert transport.has("language.rejected")
    assert not transport.has("language.failed")
    assert transport.first("language.selected").language == Language.ENGLISH


async def test_explicit_language_selection_skips_speech(
    orchestrator: ConversationOrchestrator, stt: ScriptedSTT
) -> None:
    transport = FakeTransport(
        [
            InboundFrame.control_frame(
                SelectLanguageMessage(type="select_language", language=Language.ENGLISH)
            )
        ]
    )

    await orchestrator.run(transport)

    assert transport.first("language.selected").language == Language.ENGLISH
    assert stt.calls == []


async def test_text_channel_drives_the_same_flow(
    orchestrator: ConversationOrchestrator, agent: StubAgent
) -> None:
    transport = FakeTransport(
        [
            InboundFrame.control_frame(TextMessage(type="text", text="english")),
            InboundFrame.control_frame(
                TextMessage(type="text", text="When do exams start?")
            ),
        ]
    )

    await orchestrator.run(transport)

    assert transport.first("language.selected").language == Language.ENGLISH
    assert transport.first("agent.message").text == agent.reply
    assert agent.requests[0].query == "When do exams start?"


async def test_history_is_passed_to_the_agent(
    orchestrator: ConversationOrchestrator, agent: StubAgent
) -> None:
    transport = FakeTransport(
        [
            InboundFrame.control_frame(TextMessage(type="text", text="english")),
            InboundFrame.control_frame(TextMessage(type="text", text="first question")),
            InboundFrame.control_frame(TextMessage(type="text", text="second question")),
        ]
    )

    await orchestrator.run(transport)

    second = agent.requests[1]
    assert [turn.speaker for turn in second.history] == [
        Speaker.USER,
        Speaker.ASSISTANT,
    ]
    assert second.history[0].text == "first question"
    assert second.query == "second question"


async def test_reset_returns_to_language_selection(
    orchestrator: ConversationOrchestrator,
) -> None:
    transport = FakeTransport(
        [
            InboundFrame.control_frame(
                SelectLanguageMessage(type="select_language", language=Language.ENGLISH)
            ),
            InboundFrame.control_frame(ResetMessage(type="reset")),
        ]
    )

    await orchestrator.run(transport)

    states = [e.state for e in transport.events if e.type == "state.changed"]
    assert states[-1] == ConversationState.LANGUAGE_SELECTION
    assert transport.clips[-1][0].prompt == PromptKey.LANGUAGE_RETRY


async def test_ping_is_answered_with_pong(
    orchestrator: ConversationOrchestrator,
) -> None:
    transport = FakeTransport([InboundFrame.control_frame(PingMessage(type="ping"))])

    await orchestrator.run(transport)

    assert transport.has("pong")


async def test_invalid_audio_is_reported_without_consuming_a_retry(
    orchestrator: ConversationOrchestrator, stt: ScriptedSTT
) -> None:
    stt.queue = [Transcription(text="English")]
    transport = FakeTransport(
        [
            InboundFrame.audio_frame(b"this is not audio"),
            InboundFrame.audio_frame(make_wav(1.0)),
        ]
    )

    await orchestrator.run(transport)

    error = transport.first("error")
    assert error.code == "invalid_audio"
    assert error.retryable is True
    assert not transport.has("language.rejected")
    assert transport.first("language.selected").language == Language.ENGLISH


async def test_silence_in_active_state_is_not_sent_to_the_agent(
    orchestrator: ConversationOrchestrator, stt: ScriptedSTT, agent: StubAgent
) -> None:
    stt.queue = [Transcription(text="English"), Transcription(text="   ")]
    transport = FakeTransport(
        [
            InboundFrame.audio_frame(make_wav(1.0)),
            InboundFrame.audio_frame(make_wav(1.0)),
        ]
    )

    await orchestrator.run(transport)

    assert transport.first("error").code == "no_speech_detected"
    assert agent.requests == []


async def test_invalid_control_frame_is_rejected(
    orchestrator: ConversationOrchestrator,
) -> None:
    transport = FakeTransport([InboundFrame.invalid_frame("Unsupported message.")])

    await orchestrator.run(transport)

    assert transport.first("error").code == "protocol_error"


async def test_session_is_removed_after_the_conversation(
    orchestrator: ConversationOrchestrator, sessions: InMemorySessionStore
) -> None:
    transport = FakeTransport()

    await orchestrator.run(transport)

    assert await sessions.active_count() == 0


async def test_capacity_limit_rejects_new_conversations(
    orchestrator: ConversationOrchestrator, sessions: InMemorySessionStore, settings
) -> None:
    settings.session.max_active_sessions = 1
    await sessions.create()

    transport = FakeTransport()
    await orchestrator.run(transport)

    error = transport.first("error")
    assert error.code == "capacity_exceeded"
    assert error.fatal is True
    assert transport.closed_with == (1013, "capacity")
