"""faster-whisper speech-to-text engine.

Design notes
------------
* The model is loaded once during startup (``warmup``) so the first student
  call does not pay the download/initialisation cost.
* CPU inference is serialised behind a semaphore sized by
  ``STT__MAX_CONCURRENCY``; oversubscribing CPU threads makes every concurrent
  call slower rather than increasing throughput.
* Inference runs in a worker thread and is wrapped in a timeout so a wedged
  decode surfaces as an error event instead of a hung WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from app.core.config import STTSettings
from app.core.errors import TranscriptionError, TranscriptionTimeoutError
from app.domain.enums import Language
from app.domain.models import DecodedAudio, Transcription
from app.services.stt.base import SpeechToTextEngine

logger = logging.getLogger(__name__)


class FasterWhisperEngine(SpeechToTextEngine):
    name = "faster-whisper"

    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._model = None
        self._semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
        self._load_lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    async def warmup(self) -> None:
        await self._ensure_model()

        if not self._settings.warmup_on_startup:
            return

        silence = DecodedAudio(
            samples=np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            duration_seconds=1.0,
            source_sample_rate=16_000,
            channels=1,
        )

        started = time.perf_counter()
        try:
            await self.transcribe(silence, language=Language.ENGLISH)
        except Exception:
            logger.warning("STT warmup inference failed", exc_info=True)
        else:
            logger.info(
                "STT warmup complete",
                extra={"elapsed_ms": round((time.perf_counter() - started) * 1000)},
            )

    async def aclose(self) -> None:
        self._model = None

    async def transcribe(
        self,
        audio: DecodedAudio,
        *,
        language: Language | None = None,
        prompt: str | None = None,
    ) -> Transcription:
        await self._ensure_model()

        started = time.perf_counter()

        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._transcribe_sync,
                        audio.samples,
                        language.whisper_code if language else None,
                        prompt,
                    ),
                    timeout=self._settings.timeout_seconds,
                )
            except TimeoutError as exc:
                logger.error(
                    "STT timed out",
                    extra={
                        "timeout_seconds": self._settings.timeout_seconds,
                        "audio_seconds": round(audio.duration_seconds, 2),
                    },
                )
                raise TranscriptionTimeoutError("Whisper decode timed out") from exc
            except Exception as exc:
                logger.exception("STT inference failed")
                raise TranscriptionError(str(exc)) from exc

        logger.info(
            "Transcribed utterance",
            extra={
                "text": result.text,
                "audio_seconds": round(audio.duration_seconds, 2),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "detected_language": result.detected_language,
                "language_probability": round(result.language_probability, 3),
                "forced_language": language.value if language else None,
            },
        )
        return result

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return

        async with self._load_lock:
            if self._model is not None:
                return
            self._model = await asyncio.to_thread(self._load_model)

    def _load_model(self):
        from faster_whisper import WhisperModel

        settings = self._settings
        started = time.perf_counter()

        model = WhisperModel(
            settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
            cpu_threads=settings.cpu_threads,
            num_workers=settings.num_workers,
            download_root=str(settings.download_root) if settings.download_root else None,
        )

        logger.info(
            "Whisper model loaded",
            extra={
                "model_size": settings.model_size,
                "device": settings.device,
                "compute_type": settings.compute_type,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return model

    def _transcribe_sync(
        self,
        samples: np.ndarray,
        language_code: str | None,
        prompt: str | None,
    ) -> Transcription:
        settings = self._settings

        segments, info = self._model.transcribe(
            samples,
            language=language_code,
            task="transcribe",
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            condition_on_previous_text=False,
            log_prob_threshold=settings.log_prob_threshold,
            no_speech_threshold=settings.no_speech_threshold,
            initial_prompt=prompt,
        )

        kept: list[str] = []
        no_speech = 0.0

        for segment in segments:
            no_speech = max(no_speech, getattr(segment, "no_speech_prob", 0.0) or 0.0)

            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None and avg_logprob < settings.log_prob_threshold:
                # Low-confidence segments on near-silence are Whisper
                # hallucinations ("Thank you.", subtitle credits, ...).
                logger.debug(
                    "Dropped low-confidence segment",
                    extra={"text": segment.text, "avg_logprob": avg_logprob},
                )
                continue

            text = segment.text.strip()
            if text:
                kept.append(text)

        return Transcription(
            text=" ".join(kept).strip(),
            detected_language=info.language,
            language_probability=info.language_probability or 0.0,
            duration_seconds=getattr(info, "duration", 0.0) or 0.0,
            no_speech_probability=no_speech,
            segments=tuple(kept),
        )
