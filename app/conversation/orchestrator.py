"""Conversation orchestrator: the production onboarding + Q&A state machine.

    CONNECTING
        |  session created, welcome prompt played
        v
    LANGUAGE_SELECTION  <--+ retry prompt (bounded by LANGUAGE_MAX_RETRIES)
        |  language resolved |
        v                    |
    ACTIVE --------- reset --+
        |  question received
        v
    THINKING -> agent answers -> ACTIVE
        |
        v
    CLOSED

The orchestrator owns flow and policy only. Audio decoding, transcription,
language interpretation, answering and synthesis are all injected ports, and
the transport is abstract, so this class is fully unit-testable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

from app.audio import codec
from app.audio.prompts import AudioPromptLibrary
from app.conversation.transport import FrameKind, InboundFrame, VoiceTransport
from app.core.config import Settings
from app.core.context import bind_context, get_trace_id, new_trace_id
from app.core.errors import (
    AgentError,
    AgentTimeoutError,
    AudioValidationError,
    CapacityExceededError,
    VoiceAssistantError,
)
from app.core.metrics import Metrics
from app.domain.enums import ConversationState, Language, PromptKey, Speaker
from app.domain.models import (
    AgentRequest,
    AudioClip,
    DecodedAudio,
    LanguageDecision,
    Transcription,
    VoiceSession,
)
from app.domain.protocol import (
    AgentMessage,
    AgentSource,
    AgentThinking,
    AudioHeader,
    ErrorEvent,
    LanguageFailed,
    LanguageRejected,
    LanguageSelected,
    PongEvent,
    SessionClosed,
    SessionCreated,
    StateChanged,
    TranscriptEvent,
)
from app.services.agent.base import ConversationAgent
from app.services.language import LanguageDetector
from app.services.stt.base import SpeechToTextEngine
from app.services.tts.base import TextToSpeechEngine
from app.session.store import SessionStore

logger = logging.getLogger(__name__)

# Close codes (RFC 6455 + application range).
CLOSE_NORMAL = 1000
CLOSE_POLICY_VIOLATION = 1008
CLOSE_INTERNAL_ERROR = 1011
CLOSE_TRY_AGAIN_LATER = 1013

# Biases Whisper towards the two words we actually expect during selection.
LANGUAGE_SELECTION_PROMPT = "Sinhala. English. සිංහල. ඉංග්‍රීසි."

PROMPT_TEXT = {
    ConversationState.LANGUAGE_SELECTION: "Please say Sinhala or English.",
    ConversationState.ACTIVE: "Ask your question.",
}


class ConversationOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        sessions: SessionStore,
        prompts: AudioPromptLibrary,
        stt: SpeechToTextEngine,
        agent: ConversationAgent,
        tts: TextToSpeechEngine,
        language_detector: LanguageDetector,
        metrics: Metrics | None = None,
    ) -> None:
        self._settings = settings
        self._conversation = settings.conversation
        self._sessions = sessions
        self._prompts = prompts
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._detector = language_detector
        self._metrics = metrics or Metrics()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def run(self, transport: VoiceTransport) -> None:
        """Drive one conversation from greeting to close."""

        trace_id = new_trace_id()

        with bind_context(trace_id=trace_id):
            try:
                session = await self._sessions.create()
            except CapacityExceededError as exc:
                logger.warning("Rejected connection: %s", exc)
                await self._send_error(transport, exc, fatal=True)
                await transport.close(CLOSE_TRY_AGAIN_LATER, "capacity")
                return

            with bind_context(trace_id=trace_id, session_id=session.session_id):
                logger.info("Conversation started", extra={"peer": transport.peer})
                self._metrics.active_conversations.inc()
                try:
                    await self._greet(transport, session)
                    await self._loop(transport, session)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Conversation failed")
                    await self._send_error(
                        transport,
                        VoiceAssistantError("Unhandled conversation error"),
                        fatal=True,
                    )
                    await transport.close(CLOSE_INTERNAL_ERROR, "internal error")
                finally:
                    # A caller who hangs up before choosing never reached the
                    # assistant; that is the number to watch for a bad greeting.
                    if session.language is None:
                        self._record_outcome(session, "abandoned")
                    session.transition(ConversationState.CLOSED)
                    await self._sessions.delete(session.session_id)
                    self._metrics.active_conversations.dec()
                    logger.info(
                        "Conversation ended",
                        extra={
                            "turns": len(session.turns),
                            "onboarding_outcome": session.onboarding_outcome,
                            "language": session.language.value
                            if session.language
                            else None,
                        },
                    )

    async def _greet(self, transport: VoiceTransport, session: VoiceSession) -> None:
        await transport.send_event(
            SessionCreated(
                session_id=session.session_id,
                state=session.state,
                supported_languages=list(Language),
            )
        )

        await self._set_state(transport, session, ConversationState.WELCOME)
        await self._play_prompt(transport, PromptKey.WELCOME)
        await self._begin_language_selection(transport, session)

    async def _begin_language_selection(
        self, transport: VoiceTransport, session: VoiceSession
    ) -> None:
        """Enter LANGUAGE_SELECTION and start its overall deadline."""

        session.selection_started_at = datetime.now(tz=UTC)
        await self._set_state(transport, session, ConversationState.LANGUAGE_SELECTION)

    async def _loop(self, transport: VoiceTransport, session: VoiceSession) -> None:
        while session.state is not ConversationState.CLOSED:
            try:
                frame = await asyncio.wait_for(
                    transport.receive(), timeout=self._timeout_for(session)
                )
            except TimeoutError:
                selecting = session.state is ConversationState.LANGUAGE_SELECTION
                logger.info("Closing idle conversation", extra={"state": session.state})
                if selecting:
                    self._record_outcome(session, "timeout")
                await transport.send_event(
                    ErrorEvent(
                        code="language_timeout" if selecting else "idle_timeout",
                        message=(
                            "No language was selected. Please reconnect."
                            if selecting
                            else "No audio received. Please reconnect."
                        ),
                        fatal=True,
                        retryable=False,
                    )
                )
                await transport.close(CLOSE_NORMAL, "idle timeout")
                return

            if frame.kind is FrameKind.DISCONNECT:
                logger.info("Client disconnected")
                return

            if frame.kind is FrameKind.INVALID:
                await transport.send_event(
                    ErrorEvent(
                        code="protocol_error",
                        message=frame.error or "Unexpected message.",
                    )
                )
                if await self._reject_input(transport, session, "protocol"):
                    return
                continue

            if frame.kind is FrameKind.CONTROL:
                if await self._handle_control(transport, session, frame):
                    return
                continue

            await self._handle_audio(transport, session, frame)

    def _timeout_for(self, session: VoiceSession) -> float:
        if session.state is ConversationState.LANGUAGE_SELECTION:
            per_frame = self._conversation.language_selection_timeout_seconds
            if session.selection_started_at is None:
                return per_frame
            elapsed = (
                datetime.now(tz=UTC) - session.selection_started_at
            ).total_seconds()
            remaining = self._conversation.language_selection_deadline_seconds - elapsed
            # Never zero: wait_for(0) would time out before reading a frame
            # that is already queued.
            return max(0.05, min(per_frame, remaining))
        return self._conversation.idle_timeout_seconds

    # ------------------------------------------------------------------ #
    # Control messages
    # ------------------------------------------------------------------ #
    async def _handle_control(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        frame: InboundFrame,
    ) -> bool:
        """Handle a JSON control message. Returns ``True`` to end the loop."""

        message = frame.message
        assert message is not None  # guaranteed by FrameKind.CONTROL

        match message.type:
            case "ping":
                await transport.send_event(PongEvent())

            case "end":
                await transport.send_event(SessionClosed(reason="client_request"))
                await transport.close(CLOSE_NORMAL, "client ended session")
                return True

            case "reset":
                session.resets += 1
                if session.resets > self._conversation.max_resets:
                    logger.warning(
                        "Reset limit reached", extra={"resets": session.resets}
                    )
                    self._record_outcome(session, "too_many_resets")
                    await transport.send_event(
                        ErrorEvent(
                            code="too_many_resets",
                            message="Too many restarts. Please reconnect.",
                            fatal=True,
                            retryable=False,
                        )
                    )
                    await transport.close(CLOSE_POLICY_VIOLATION, "too many resets")
                    session.transition(ConversationState.CLOSED)
                    return True

                session.language = None
                session.language_method = None
                session.language_retries = 0
                session.invalid_inputs = 0
                session.onboarding_outcome = None  # each pass is its own funnel entry
                session.turns.clear()
                await self._begin_language_selection(transport, session)
                await self._play_prompt(transport, PromptKey.LANGUAGE_RETRY)
                logger.info("Conversation reset by client")

            case "select_language":
                if session.state is not ConversationState.LANGUAGE_SELECTION:
                    await transport.send_event(
                        ErrorEvent(
                            code="unexpected_input",
                            message=(
                                "A language can only be chosen during language "
                                "selection. Send 'reset' to choose again."
                            ),
                        )
                    )
                    return False

                await self._apply_language(
                    transport,
                    session,
                    LanguageDecision(message.language, 1.0, "client_selected"),
                    method="ui",
                )

            case "text":
                await self._handle_text(transport, session, message.text)

        return False

    async def _handle_text(
        self, transport: VoiceTransport, session: VoiceSession, text: str
    ) -> None:
        """Treat a typed message exactly like a spoken one."""

        if session.state is ConversationState.LANGUAGE_SELECTION:
            decision = self._detector.detect(Transcription(text=text))
            await self._resolve_language(transport, session, decision, method="text")
            return

        if session.state is ConversationState.ACTIVE:
            await self._answer(transport, session, text)
            return

        await transport.send_event(
            ErrorEvent(
                code="unexpected_input",
                message=f"Text is not accepted while in state '{session.state.value}'.",
            )
        )

    # ------------------------------------------------------------------ #
    # Audio turns
    # ------------------------------------------------------------------ #
    async def _handle_audio(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        frame: InboundFrame,
    ) -> None:
        payload = frame.audio or b""

        try:
            decoded = await asyncio.to_thread(
                codec.decode, payload, self._settings.audio
            )
        except AudioValidationError as exc:
            logger.info(
                "Rejected inbound audio",
                extra={"reason": str(exc), "size_bytes": len(payload)},
            )
            await self._send_error(transport, exc)
            await self._reject_input(transport, session, "audio")
            return

        if session.state is ConversationState.LANGUAGE_SELECTION:
            await self._handle_language_audio(transport, session, decoded)
            return

        if session.state is ConversationState.ACTIVE:
            await self._handle_question_audio(transport, session, decoded)
            return

        logger.debug("Dropped audio for state %s", session.state)
        await transport.send_event(
            ErrorEvent(
                code="unexpected_audio",
                message=f"Audio is not accepted while in state '{session.state.value}'.",
            )
        )

    async def _handle_language_audio(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        decoded: DecodedAudio,
    ) -> None:
        try:
            transcription = await self._stt.transcribe(
                decoded, language=None, prompt=LANGUAGE_SELECTION_PROMPT
            )
        except VoiceAssistantError as exc:
            await self._send_error(transport, exc)
            await self._reject_input(transport, session, "stt")
            return

        await self._send_transcript(transport, session, transcription)

        decision = self._detector.detect(transcription)
        await self._resolve_language(transport, session, decision, method="voice")

    async def _handle_question_audio(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        decoded: DecodedAudio,
    ) -> None:
        try:
            transcription = await self._stt.transcribe(
                decoded, language=session.language
            )
        except VoiceAssistantError as exc:
            await self._send_error(transport, exc)
            await self._reject_input(transport, session, "stt")
            return

        if transcription.is_empty:
            logger.info(
                "Empty transcript",
                extra={"no_speech_probability": transcription.no_speech_probability},
            )
            await transport.send_event(
                ErrorEvent(
                    code="no_speech_detected",
                    message="Nothing was recognised. Please speak again.",
                )
            )
            await self._reject_input(transport, session, "silence")
            return

        await self._send_transcript(transport, session, transcription)
        await self._answer(transport, session, transcription.text)

    # ------------------------------------------------------------------ #
    # Language selection
    # ------------------------------------------------------------------ #
    async def _resolve_language(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        decision: LanguageDecision,
        *,
        method: str,
    ) -> None:
        # The input was usable audio/text, so the invalid-input streak is over.
        session.invalid_inputs = 0

        if decision.resolved:
            await self._apply_language(transport, session, decision, method=method)
            return

        session.language_retries += 1
        self._metrics.language_attempts.inc(reason=decision.reason)
        attempts_remaining = max(
            0, self._conversation.language_max_retries - session.language_retries
        )
        await self._sessions.save(session)

        logger.info(
            "Language not recognised",
            extra={
                "attempt": session.language_retries,
                "attempts_remaining": attempts_remaining,
                "reason": decision.reason,
            },
        )

        if attempts_remaining == 0:
            self._record_outcome(session, "failed")
            await transport.send_event(
                LanguageFailed(attempts=session.language_retries)
            )
            await transport.close(CLOSE_POLICY_VIOLATION, "language selection failed")
            session.transition(ConversationState.CLOSED)
            return

        await transport.send_event(
            LanguageRejected(
                attempt=session.language_retries,
                attempts_remaining=attempts_remaining,
                reason=decision.reason,
            )
        )
        await self._play_prompt(transport, PromptKey.LANGUAGE_RETRY)

    async def _apply_language(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        decision: LanguageDecision,
        *,
        method: str,
    ) -> None:
        language = decision.language
        assert language is not None

        attempts = session.language_retries + 1
        session.language = language
        session.language_method = method
        session.language_retries = 0
        session.invalid_inputs = 0
        session.transition(ConversationState.ACTIVE)
        await self._sessions.save(session)

        self._record_outcome(session, "selected")
        self._metrics.language_selected.inc(language=language.value, method=method)
        if session.selection_started_at is not None:
            self._metrics.selection_seconds.observe(
                (datetime.now(tz=UTC) - session.selection_started_at)
                .total_seconds(),
                method=method,
            )

        logger.info(
            "Language selected",
            extra={
                "language": language.value,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "method": method,
                "attempts": attempts,
            },
        )

        await transport.send_event(
            LanguageSelected(
                language=language,
                display_name=language.display_name,
                state=session.state,
                method=method,  # type: ignore[arg-type]
                confidence=decision.confidence,
                reason=decision.reason,
            )
        )
        await self._play_prompt(transport, PromptKey.LANGUAGE_CONFIRM, language)
        await self._set_state(transport, session, ConversationState.ACTIVE)

    # ------------------------------------------------------------------ #
    # Answering (hybrid RAG seam)
    # ------------------------------------------------------------------ #
    async def _answer(
        self, transport: VoiceTransport, session: VoiceSession, query: str
    ) -> None:
        language = session.language or Language.ENGLISH

        session.invalid_inputs = 0
        session.add_turn(
            Speaker.USER, query, max_turns=self._settings.session.max_history_turns
        )
        await self._set_state(
            transport, session, ConversationState.THINKING, announce=False
        )
        await transport.send_event(AgentThinking())

        request = AgentRequest(
            session_id=session.session_id,
            language=language,
            query=query,
            history=session.history()[:-1],
            trace_id=get_trace_id(),
            metadata={
                "top_k": self._settings.agent.top_k,
                "language_method": session.language_method,
            },
        )

        started = time.perf_counter()

        try:
            reply = await asyncio.wait_for(
                self._agent.answer(request),
                timeout=self._conversation.agent_timeout_seconds,
            )
        except TimeoutError:
            logger.error(
                "Agent timed out",
                extra={"timeout_seconds": self._conversation.agent_timeout_seconds},
            )
            await self._send_error(transport, AgentTimeoutError())
            await self._set_state(transport, session, ConversationState.ACTIVE)
            return
        except AgentError as exc:
            logger.warning("Agent error: %s", exc)
            await self._send_error(transport, exc)
            await self._set_state(transport, session, ConversationState.ACTIVE)
            return
        except Exception as exc:
            logger.exception("Agent raised an unexpected error")
            await self._send_error(transport, AgentError(str(exc)))
            await self._set_state(transport, session, ConversationState.ACTIVE)
            return

        logger.info(
            "Agent replied",
            extra={
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "chunks": len(reply.chunks),
                "answer_chars": len(reply.text),
            },
        )

        session.add_turn(
            Speaker.ASSISTANT,
            reply.text,
            max_turns=self._settings.session.max_history_turns,
        )
        await self._sessions.save(session)

        await transport.send_event(
            AgentMessage(
                text=reply.text,
                language=reply.language or language,
                sources=[
                    AgentSource(id=chunk.id, source=chunk.source, score=chunk.score)
                    for chunk in reply.chunks
                ],
                metadata=dict(reply.metadata),
            )
        )

        await self._speak(transport, reply.text, language)
        await self._set_state(transport, session, ConversationState.ACTIVE)

    async def _speak(
        self, transport: VoiceTransport, text: str, language: Language
    ) -> None:
        if not self._tts.is_available or not text.strip():
            return

        try:
            clip = await asyncio.wait_for(
                self._tts.synthesize(text, language),
                timeout=self._settings.tts.timeout_seconds,
            )
        except Exception:
            # A missing voice must not cost the student their text answer.
            logger.exception("Speech synthesis failed")
            return

        await self._send_clip(
            transport, clip, header_type="agent.audio", language=language
        )

    # ------------------------------------------------------------------ #
    # Abuse budget and outcome bookkeeping
    # ------------------------------------------------------------------ #
    async def _reject_input(
        self, transport: VoiceTransport, session: VoiceSession, kind: str
    ) -> bool:
        """Count one unusable input; close the call when the streak is too long.

        Returns ``True`` when the conversation was closed.
        """

        self._metrics.invalid_inputs.inc(kind=kind)
        session.invalid_inputs += 1

        if session.invalid_inputs < self._conversation.max_invalid_inputs:
            return False

        logger.warning(
            "Closing conversation after repeated unusable input",
            extra={"invalid_inputs": session.invalid_inputs, "last_kind": kind},
        )
        self._record_outcome(session, "too_many_invalid_inputs")
        await transport.send_event(
            ErrorEvent(
                code="too_many_invalid_inputs",
                message="Too many unreadable inputs. Please reconnect.",
                fatal=True,
                retryable=False,
            )
        )
        await transport.close(CLOSE_POLICY_VIOLATION, "too many invalid inputs")
        session.transition(ConversationState.CLOSED)
        return True

    def _record_outcome(self, session: VoiceSession, outcome: str) -> None:
        """Record how onboarding ended, exactly once per session."""

        if session.onboarding_outcome is not None:
            return
        session.onboarding_outcome = outcome
        self._metrics.onboarding_outcomes.inc(outcome=outcome)

    # ------------------------------------------------------------------ #
    # Outbound helpers
    # ------------------------------------------------------------------ #
    async def _set_state(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        state: ConversationState,
        *,
        announce: bool = True,
    ) -> None:
        session.transition(state)
        await self._sessions.save(session)

        if announce:
            await transport.send_event(
                StateChanged(state=state, message=PROMPT_TEXT.get(state))
            )

    async def _play_prompt(
        self,
        transport: VoiceTransport,
        key: PromptKey,
        language: Language | None = None,
    ) -> None:
        clip = self._prompts.get(key, language)
        self._metrics.prompts_played.inc(
            prompt=key.value, language=language.value if language else "neutral"
        )
        await self._send_clip(
            transport, clip, header_type="prompt.audio", prompt=key, language=language
        )

    async def _send_clip(
        self,
        transport: VoiceTransport,
        clip: AudioClip,
        *,
        header_type: str,
        prompt: PromptKey | None = None,
        language: Language | None = None,
    ) -> None:
        header = AudioHeader(
            type=header_type,  # type: ignore[arg-type]
            prompt=prompt,
            language=language,
            content_type=clip.content_type,
            size_bytes=clip.size_bytes,
            duration_seconds=round(clip.duration_seconds, 3),
            sample_rate=clip.sample_rate,
            checksum=clip.checksum,
        )
        await transport.send_clip(header, clip)

    async def _send_transcript(
        self,
        transport: VoiceTransport,
        session: VoiceSession,
        transcription: Transcription,
    ) -> None:
        if not self._conversation.send_transcripts:
            return

        await transport.send_event(
            TranscriptEvent(
                text=transcription.text,
                language=session.language,
                detected_language=transcription.detected_language,
                language_probability=round(transcription.language_probability, 3),
            )
        )

    async def _send_error(
        self,
        transport: VoiceTransport,
        error: VoiceAssistantError,
        *,
        fatal: bool = False,
    ) -> None:
        await transport.send_event(
            ErrorEvent(
                code=error.code,
                message=error.client_message,
                fatal=fatal,
                retryable=not fatal,
            )
        )
