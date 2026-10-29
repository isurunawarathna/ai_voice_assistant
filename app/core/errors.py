"""Domain errors.

Each error carries a stable ``code`` that is safe to put on the wire, plus a
``client_message`` that is safe to show a student. Internal details stay in the
logs and never reach the WebSocket.
"""

from __future__ import annotations


class VoiceAssistantError(Exception):
    """Base class for every expected failure in the voice pipeline."""

    code = "internal_error"
    client_message = "Voice processing failed."

    def __init__(self, message: str | None = None, *, client_message: str | None = None):
        super().__init__(message or self.client_message)
        if client_message is not None:
            self.client_message = client_message


class ConfigurationError(VoiceAssistantError):
    code = "configuration_error"
    client_message = "The assistant is misconfigured."


class AudioValidationError(VoiceAssistantError):
    code = "invalid_audio"
    client_message = "The audio could not be read."


class AudioTooLargeError(AudioValidationError):
    code = "audio_too_large"
    client_message = "The recording is too large."


class AudioTooShortError(AudioValidationError):
    code = "audio_too_short"
    client_message = "The recording is too short. Please speak again."


class TranscriptionError(VoiceAssistantError):
    code = "transcription_failed"
    client_message = "Speech could not be recognised. Please try again."


class TranscriptionTimeoutError(TranscriptionError):
    code = "transcription_timeout"
    client_message = "Speech recognition timed out. Please try again."


class AgentError(VoiceAssistantError):
    code = "agent_failed"
    client_message = "The assistant could not answer that. Please try again."


class AgentTimeoutError(AgentError):
    code = "agent_timeout"
    client_message = "The assistant took too long to answer. Please try again."


class SynthesisError(VoiceAssistantError):
    code = "synthesis_failed"
    client_message = "The reply could not be spoken."


class SessionNotFoundError(VoiceAssistantError):
    code = "session_not_found"
    client_message = "The session has expired. Please reconnect."


class CapacityExceededError(VoiceAssistantError):
    code = "capacity_exceeded"
    client_message = "The assistant is busy. Please try again shortly."


class PromptNotFoundError(ConfigurationError):
    code = "prompt_not_found"
    client_message = "An audio prompt is unavailable."


class ProtocolError(VoiceAssistantError):
    code = "protocol_error"
    client_message = "Unexpected message received."
