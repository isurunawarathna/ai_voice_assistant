"""Abuse budgets, state guards and bookkeeping of the welcome/language flow."""

from __future__ import annotations

import asyncio

import pytest

from app.conversation.orchestrator import (
    CLOSE_NORMAL,
    CLOSE_POLICY_VIOLATION,
    ConversationOrchestrator,
)
from app.conversation.transport import InboundFrame
from app.core.config import Settings
from app.core.metrics import Metrics
from app.domain.enums import ConversationState, Language, PromptKey
from app.domain.models import Transcription
from app.domain.protocol import ResetMessage, SelectLanguageMessage, TextMessage
from app.services.language import LanguageDetector
from app.services.tts.base import NullTextToSpeech
from app.session.memory_store import InMemorySessionStore
from tests.conftest import FakeTransport, ScriptedSTT, StubAgent, StubPrompts, make_wav

pytestmark = pytest.mark.asyncio


def control(message) -> InboundFrame:
    return InboundFrame.control_frame(message)


def reset() -> InboundFrame:
    return control(ResetMessage(type="reset"))


def choose(language: Language) -> InboundFrame:
    return control(SelectLanguageMessage(type="select_language", language=language))


def say(text: str) -> InboundFrame:
    return control(TextMessage(type="text", text=text))


def tiny_audio() -> InboundFrame:
    return InboundFrame.audio_frame(make_wav(0.05))  # below min_seconds


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


@pytest.fixture
def build(settings: Settings, stt: ScriptedSTT, agent: StubAgent, metrics: Metrics):
    def _build(**conversation: float) -> ConversationOrchestrator:
        for key, value in conversation.items():
            setattr(settings.conversation, key, value)
        return ConversationOrchestrator(
            settings=settings,
            sessions=InMemorySessionStore(settings.session),
            prompts=StubPrompts(settings.audio),
            stt=stt,
            agent=agent,
            tts=NullTextToSpeech(),
            language_detector=LanguageDetector(),
            metrics=metrics,
        )

    return _build


# --------------------------------------------------------------------------- #
# State guards
# --------------------------------------------------------------------------- #
async def test_language_cannot_be_switched_by_select_language_once_active(
    build,
) -> None:
    transport = FakeTransport([choose(Language.SINHALA), choose(Language.ENGLISH)])

    await build().run(transport)

    selected = [e for e in transport.events if e.type == "language.selected"]
    assert [e.language for e in selected] == [Language.SINHALA]

    error = transport.first("error")
    assert error.code == "unexpected_input"
    assert "reset" in error.message
    assert error.fatal is False

    # Only one confirmation was played, not two.
    confirms = [h for h, _ in transport.clips if h.prompt == PromptKey.LANGUAGE_CONFIRM]
    assert len(confirms) == 1


async def test_reset_is_the_supported_way_to_change_language(build) -> None:
    transport = FakeTransport(
        [choose(Language.SINHALA), reset(), choose(Language.ENGLISH)]
    )

    await build().run(transport)

    selected = [e.language for e in transport.events if e.type == "language.selected"]
    assert selected == [Language.SINHALA, Language.ENGLISH]


# --------------------------------------------------------------------------- #
# Abuse budgets
# --------------------------------------------------------------------------- #
async def test_reset_spam_is_cut_off(build) -> None:
    transport = FakeTransport([reset() for _ in range(50)])

    await build(max_resets=3).run(transport)

    # welcome + one retry prompt per allowed reset, then the socket is closed.
    assert len(transport.clips) == 1 + 3
    error = transport.events[-1]
    assert error.type == "error" and error.code == "too_many_resets" and error.fatal
    assert transport.closed_with == (CLOSE_POLICY_VIOLATION, "too many resets")


async def test_endless_unreadable_audio_is_cut_off(build, metrics: Metrics) -> None:
    transport = FakeTransport([tiny_audio() for _ in range(100)])

    await build(max_invalid_inputs=4).run(transport)

    errors = [e for e in transport.events if e.type == "error"]
    assert errors[-1].code == "too_many_invalid_inputs" and errors[-1].fatal
    assert len(errors) == 4 + 1  # four rejections, then the fatal notice
    assert transport.closed_with == (CLOSE_POLICY_VIOLATION, "too many invalid inputs")
    assert metrics.invalid_inputs.value(kind="audio") == 4
    assert (
        metrics.onboarding_outcomes.value(outcome="too_many_invalid_inputs") == 1
    )


async def test_garbage_control_frames_are_cut_off(build) -> None:
    transport = FakeTransport([InboundFrame.invalid_frame("nope") for _ in range(50)])

    await build(max_invalid_inputs=3).run(transport)

    assert transport.closed_with == (CLOSE_POLICY_VIOLATION, "too many invalid inputs")


async def test_one_good_input_resets_the_invalid_streak(build, stt: ScriptedSTT) -> None:
    stt.queue = [Transcription(text="what?")]
    # 2 bad, a usable (but unrecognised) utterance, 2 bad: never 3 in a row.
    transport = FakeTransport(
        [
            tiny_audio(),
            tiny_audio(),
            InboundFrame.audio_frame(make_wav(1.0)),
            tiny_audio(),
            tiny_audio(),
            choose(Language.ENGLISH),
        ]
    )

    await build(max_invalid_inputs=3).run(transport)

    assert transport.has("language.selected")
    assert not any(
        e.type == "error" and e.code == "too_many_invalid_inputs"
        for e in transport.events
    )


async def test_selection_deadline_ends_a_client_that_never_answers(
    build, metrics: Metrics
) -> None:
    orchestrator = build(
        language_selection_timeout_seconds=5.0,
        language_selection_deadline_seconds=0.2,
    )

    # A client that stays connected but never sends anything: without the
    # overall deadline the 5 s per-frame wait would be the only limit.
    class Silent(FakeTransport):
        async def receive(self):  # type: ignore[override]
            await asyncio.sleep(60)

    silent = Silent()
    await orchestrator.run(silent)

    error = [e for e in silent.events if e.type == "error"][-1]
    assert error.code == "language_timeout" and error.fatal
    assert silent.closed_with == (CLOSE_NORMAL, "idle timeout")
    assert metrics.onboarding_outcomes.value(outcome="timeout") == 1


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("frames", "method"),
    [
        ([choose(Language.ENGLISH)], "ui"),
        ([say("english please")], "text"),
    ],
)
async def test_language_selected_reports_how_it_was_chosen(
    build, frames, method: str
) -> None:
    transport = FakeTransport(frames)

    await build().run(transport)

    selected = transport.first("language.selected")
    assert selected.method == method
    assert 0 < selected.confidence <= 1.0


async def test_spoken_selection_is_marked_as_voice(build, stt: ScriptedSTT) -> None:
    stt.queue = [Transcription(text="Sinhala")]
    transport = FakeTransport([InboundFrame.audio_frame(make_wav(1.0))])

    await build().run(transport)

    selected = transport.first("language.selected")
    assert selected.method == "voice"
    assert selected.reason == "keyword_exact"


async def test_onboarding_funnel_metrics(
    build, stt: ScriptedSTT, metrics: Metrics
) -> None:
    stt.queue = [Transcription(text="hmm"), Transcription(text="English")]
    transport = FakeTransport(
        [InboundFrame.audio_frame(make_wav(1.0)), InboundFrame.audio_frame(make_wav(1.0))]
    )

    await build().run(transport)

    assert metrics.language_attempts.value(reason="no_match") == 1
    assert metrics.language_selected.value(language="en-US", method="voice") == 1
    assert metrics.onboarding_outcomes.value(outcome="selected") == 1
    assert metrics.selection_seconds.count(method="voice") == 1
    assert metrics.prompts_played.value(prompt="welcome", language="neutral") == 1
    assert metrics.active_conversations.value == 0

    text = metrics.render()
    assert "# TYPE voice_onboarding_outcomes_total counter" in text
    assert 'voice_language_selected_total{language="en-US",method="voice"} 1' in text
    assert 'voice_language_selection_seconds_bucket{method="voice",le="+Inf"} 1' in text


async def test_hanging_up_before_choosing_counts_as_abandoned(
    build, metrics: Metrics
) -> None:
    await build().run(FakeTransport())  # client disconnects right after the welcome

    assert metrics.onboarding_outcomes.value(outcome="abandoned") == 1
    assert metrics.onboarding_outcomes.value(outcome="selected") == 0


async def test_failed_selection_is_not_double_counted(
    build, stt: ScriptedSTT, metrics: Metrics, settings: Settings
) -> None:
    attempts = settings.conversation.language_max_retries
    stt.queue = [Transcription(text="what?") for _ in range(attempts)]
    transport = FakeTransport(
        [InboundFrame.audio_frame(make_wav(1.0)) for _ in range(attempts)]
    )

    await build().run(transport)

    assert metrics.onboarding_outcomes.value(outcome="failed") == 1
    assert metrics.onboarding_outcomes.value(outcome="abandoned") == 0


async def test_agent_receives_trace_id_and_selection_method(
    build, agent: StubAgent
) -> None:
    transport = FakeTransport([choose(Language.SINHALA), say("fees?")])

    await build().run(transport)

    request = agent.requests[0]
    assert request.language is Language.SINHALA
    assert request.trace_id  # hand-off keeps the log correlation id
    assert request.metadata["language_method"] == "ui"


async def test_state_is_active_after_a_completed_onboarding(build) -> None:
    transport = FakeTransport([choose(Language.ENGLISH)])

    await build().run(transport)

    states = [e.state for e in transport.events if e.type == "state.changed"]
    assert states[-1] == ConversationState.ACTIVE
