"""Pre-recorded prompt library.

The onboarding flow (welcome, language selection, confirmation) is driven by
fixed WAV files rather than TTS so it is instant, free and identical on every
call. Files are validated and cached in memory at startup, so a missing or
corrupt prompt fails the deploy instead of the first student call.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.audio import codec
from app.core.config import AudioSettings
from app.core.errors import PromptNotFoundError
from app.domain.enums import Language, PromptKey
from app.domain.models import AudioClip

logger = logging.getLogger(__name__)

PromptId = tuple[PromptKey, Language | None]

# ``None`` as the language means "language-neutral" (the bilingual prompts
# played before the student has chosen).
PROMPT_MANIFEST: dict[PromptId, str] = {
    (PromptKey.WELCOME, None): "welcome_message.wav",
    (PromptKey.LANGUAGE_RETRY, None): "select_language_again.wav",
    (PromptKey.LANGUAGE_CONFIRM, Language.SINHALA): "sinhala_confirm.wav",
    (PromptKey.LANGUAGE_CONFIRM, Language.ENGLISH): "english_confirm.wav",
}


class AudioPromptLibrary:
    def __init__(self, settings: AudioSettings) -> None:
        self._settings = settings
        self._validation_settings = settings.model_copy(
            update={"max_seconds": 600.0, "min_seconds": 0.0}
        )
        self._clips: dict[PromptId, AudioClip] = {}
        self._loaded = False

    @property
    def directory(self) -> Path:
        return self._settings.assets_dir

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """Read, validate and cache every prompt declared in the manifest."""

        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        clips: dict[PromptId, AudioClip] = {}
        missing: list[str] = []

        for prompt_id, filename in PROMPT_MANIFEST.items():
            path = self.directory / filename

            if not path.is_file():
                missing.append(str(path))
                continue

            try:
                clips[prompt_id] = codec.load_clip(
                    path.read_bytes(), self._validation_settings
                )
            except Exception as exc:
                missing.append(f"{path} ({exc})")
                continue

            key, language = prompt_id
            logger.info(
                "Loaded prompt",
                extra={
                    "prompt": key.value,
                    "language": language.value if language else None,
                    "duration_seconds": round(clips[prompt_id].duration_seconds, 2),
                    "size_bytes": clips[prompt_id].size_bytes,
                },
            )

        if missing:
            message = "Unusable audio prompts: " + ", ".join(missing)
            if self._settings.strict_prompt_loading:
                raise PromptNotFoundError(message)
            logger.warning(message)

        self._clips = clips
        self._loaded = True

    def get(self, key: PromptKey, language: Language | None = None) -> AudioClip:
        """Return a prompt, falling back to the language-neutral recording."""

        for candidate in ((key, language), (key, None)):
            clip = self._clips.get(candidate)
            if clip is not None:
                return clip

        raise PromptNotFoundError(
            f"No audio prompt for {key.value} (language={language})"
        )

    def has(self, key: PromptKey, language: Language | None = None) -> bool:
        return (key, language) in self._clips or (key, None) in self._clips

    def describe(self) -> list[dict[str, object]]:
        """Inventory for the readiness endpoint."""

        return [
            {
                "prompt": key.value,
                "language": language.value if language else None,
                "duration_seconds": round(clip.duration_seconds, 2),
                "size_bytes": clip.size_bytes,
                "checksum": clip.checksum,
            }
            for (key, language), clip in sorted(
                self._clips.items(), key=lambda item: (item[0][0].value, str(item[0][1]))
            )
        ]
