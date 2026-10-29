"""Core domain enumerations shared by every layer."""

from __future__ import annotations

from enum import StrEnum


class Language(StrEnum):
    """Supported conversation languages, keyed by BCP-47 tag."""

    SINHALA = "si-LK"
    ENGLISH = "en-US"

    @property
    def whisper_code(self) -> str:
        return _WHISPER_CODES[self]

    @property
    def display_name(self) -> str:
        return _DISPLAY_NAMES[self]

    @classmethod
    def from_whisper_code(cls, code: str | None) -> Language | None:
        if not code:
            return None
        return _BY_WHISPER_CODE.get(code.lower())


_WHISPER_CODES: dict[Language, str] = {
    Language.SINHALA: "si",
    Language.ENGLISH: "en",
}

_DISPLAY_NAMES: dict[Language, str] = {
    Language.SINHALA: "Sinhala",
    Language.ENGLISH: "English",
}

_BY_WHISPER_CODE: dict[str, Language] = {
    code: language for language, code in _WHISPER_CODES.items()
}


class ConversationState(StrEnum):
    """States of the conversation state machine."""

    CONNECTING = "connecting"
    WELCOME = "welcome"
    LANGUAGE_SELECTION = "language_selection"
    ACTIVE = "active"
    THINKING = "thinking"
    CLOSED = "closed"


class PromptKey(StrEnum):
    """Pre-recorded audio prompts used by the onboarding flow."""

    WELCOME = "welcome"
    LANGUAGE_RETRY = "language_retry"
    LANGUAGE_CONFIRM = "language_confirm"


class Speaker(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
