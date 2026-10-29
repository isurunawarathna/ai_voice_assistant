"""Language selection.

Whisper transcripts of a single spoken word are noisy, especially for Sinhala
spoken into a laptop microphone: "English" may come back as "Inglis", as two
words ("Sin hala"), or written in Sinhala script ("ඉංග්ලිෂ්"). Rather than an
exact set membership test, the detector applies a ladder of signals and reports
how it decided, so the logs explain every retry a student experiences.

Signal order (first match wins):
    1. Explicit keyword, exact token match            -> 0.95
    2. Explicit keyword, split by Whisper ("sin hala") -> 0.85
    3. Explicit keyword, fuzzy token match             -> 0.80
    4. IVR-style numeric choice ("one"/"1"/"එක")        -> 0.75
    5. Sinhala script present, no English-like token   -> 0.70
    6. Whisper's own language id above threshold       -> 0.55

Guard rails (each one is covered by a test):
    * A keyword for *both* languages in one utterance is ambiguous, not a guess.
    * Fuzzy matching ignores very short tokens, so "singles" is not "singhalese".
    * Numeric choices only count in a short utterance, so "I have two questions"
      is not a vote for English.
    * Sinhala script is only treated as Sinhala when no token looks like a
      Sinhala-script spelling of "English".
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from itertools import pairwise

from app.domain.enums import Language
from app.domain.models import LanguageDecision, Transcription

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\w඀-෿]+", re.UNICODE)
_SINHALA_RANGE = re.compile(r"[඀-෿]")
_ZERO_WIDTH = ("‍", "‌")  # ZWJ / ZWNJ


def _normalize(text: str) -> str:
    """Casefold and strip Sinhala zero-width joiners so tokens compare cleanly."""

    folded = unicodedata.normalize("NFC", text).casefold().strip()
    for char in _ZERO_WIDTH:
        folded = folded.replace(char, "")
    return folded


def _keywords(*words: str) -> frozenset[str]:
    return frozenset(_normalize(word) for word in words)


SINHALA_KEYWORDS: frozenset[str] = _keywords(
    "සිංහල",
    "සිංහලෙන්",
    "සිංහලයි",
    "සිහල",
    "sinhala",
    "sinhalese",
    "singhala",
    "singhalese",
    "sinhalen",
    "sinhale",
    "cinhala",
    "shinhala",
    "senhala",
)

ENGLISH_KEYWORDS: frozenset[str] = _keywords(
    "english",
    "englisch",
    "inglish",
    "ingreesi",
    "ingrisi",
    "ingirisi",
    "ඉංග්‍රීසි",
    "ඉංග්‍රීසියෙන්",
    "ඉංගිරිසි",
    # Sinhala-script spellings of the English word itself, which is what
    # Whisper tends to emit when it decides the audio is Sinhala.
    "ඉංග්ලිෂ්",
    "ඉංග්ලිශ්",
    "ඉන්ග්ලිෂ්",
    "ඉංග්ලීෂ්",
    "ඉංග්ලීසි",
    "ඉංගිලිෂ්",
)

# Spoken/keyed menu choices. The welcome prompt announces Sinhala first.
NUMERIC_CHOICES: dict[str, Language] = {
    "1": Language.SINHALA,
    "one": Language.SINHALA,
    "එක": Language.SINHALA,
    "එකයි": Language.SINHALA,
    "2": Language.ENGLISH,
    "two": Language.ENGLISH,
    "දෙක": Language.ENGLISH,
    "දෙකයි": Language.ENGLISH,
}

FUZZY_THRESHOLD = 0.85
WHISPER_PROBABILITY_THRESHOLD = 0.80

# A language answer is a word or two. Anything much longer is a sentence about
# something else, so numeric choices are not honoured in it.
MAX_NUMERIC_TOKENS = 3
# Bound the work done for an arbitrarily long typed/transcribed message.
MAX_TOKENS_CONSIDERED = 24
# Latin tokens shorter than this are too easy to match by accident.
MIN_FUZZY_TOKEN_LENGTH = 5
# Looser bar used only to *veto* the Sinhala-script heuristic.
SCRIPT_VETO_THRESHOLD = 0.55


def _ratio(token: str, keyword: str) -> float:
    return SequenceMatcher(None, token, keyword).ratio()


def _best_fuzzy_match(token: str, keywords: frozenset[str]) -> float:
    return max((_ratio(token, keyword) for keyword in keywords), default=0.0)


def _is_fuzzy_candidate(token: str) -> bool:
    if _SINHALA_RANGE.search(token):
        return len(token) >= 3
    return len(token) >= MIN_FUZZY_TOKEN_LENGTH


class LanguageDetector:
    """Maps a transcript (plus Whisper metadata) onto a supported language."""

    def __init__(
        self,
        *,
        fuzzy_threshold: float = FUZZY_THRESHOLD,
        whisper_probability_threshold: float = WHISPER_PROBABILITY_THRESHOLD,
    ) -> None:
        self._fuzzy_threshold = fuzzy_threshold
        self._whisper_threshold = whisper_probability_threshold

    def detect(self, transcription: Transcription) -> LanguageDecision:
        text = transcription.text

        if not text.strip():
            return LanguageDecision(None, 0.0, "empty_transcript")

        # Normalise before tokenising: a zero-width joiner inside a Sinhala
        # word is not a word character, so tokenising first would split it.
        normalized = _normalize(text)
        tokens = _TOKEN_RE.findall(normalized)[:MAX_TOKENS_CONSIDERED]

        if not tokens:
            return LanguageDecision(None, 0.0, "no_tokens")

        token_set = set(tokens)

        # 1. exact keyword
        sinhala_hit = bool(token_set & SINHALA_KEYWORDS)
        english_hit = bool(token_set & ENGLISH_KEYWORDS)

        if sinhala_hit and english_hit:
            return LanguageDecision(None, 0.0, "ambiguous_both_languages")
        if sinhala_hit:
            return LanguageDecision(Language.SINHALA, 0.95, "keyword_exact")
        if english_hit:
            return LanguageDecision(Language.ENGLISH, 0.95, "keyword_exact")

        # 2. a keyword that Whisper split in two ("sin hala")
        joined = [left + right for left, right in pairwise(tokens)]
        joined_set = set(joined)
        sinhala_joined = bool(joined_set & SINHALA_KEYWORDS)
        english_joined = bool(joined_set & ENGLISH_KEYWORDS)

        if sinhala_joined and english_joined:
            return LanguageDecision(None, 0.0, "ambiguous_both_languages")
        if sinhala_joined:
            return LanguageDecision(Language.SINHALA, 0.85, "keyword_joined")
        if english_joined:
            return LanguageDecision(Language.ENGLISH, 0.85, "keyword_joined")

        # 3. fuzzy keyword
        candidates = [t for t in (*tokens, *joined) if _is_fuzzy_candidate(t)]
        sinhala_score = max(
            (_best_fuzzy_match(t, SINHALA_KEYWORDS) for t in candidates), default=0.0
        )
        english_score = max(
            (_best_fuzzy_match(t, ENGLISH_KEYWORDS) for t in candidates), default=0.0
        )

        if (
            max(sinhala_score, english_score) >= self._fuzzy_threshold
            and abs(sinhala_score - english_score) >= 0.05
        ):
            language = (
                Language.SINHALA if sinhala_score > english_score else Language.ENGLISH
            )
            return LanguageDecision(language, 0.80, "keyword_fuzzy")

        # 4. numeric menu choice, only in a short utterance
        if len(tokens) <= MAX_NUMERIC_TOKENS:
            numeric = {
                NUMERIC_CHOICES[token] for token in tokens if token in NUMERIC_CHOICES
            }
            if len(numeric) == 1:
                return LanguageDecision(numeric.pop(), 0.75, "numeric_choice")

        # 5. script heuristic, vetoed when a Sinhala-script token looks like
        #    a spelling of "English" that the keyword list did not anticipate.
        if _SINHALA_RANGE.search(text):
            script_tokens = [t for t in tokens if _SINHALA_RANGE.search(t)]
            looks_english = any(
                _best_fuzzy_match(t, ENGLISH_KEYWORDS) >= SCRIPT_VETO_THRESHOLD
                for t in script_tokens
            )
            if looks_english:
                return LanguageDecision(None, 0.0, "ambiguous_script")
            return LanguageDecision(Language.SINHALA, 0.70, "sinhala_script")

        # 6. Whisper's own language identification
        detected = Language.from_whisper_code(transcription.detected_language)
        if (
            detected is not None
            and transcription.language_probability >= self._whisper_threshold
        ):
            return LanguageDecision(detected, 0.55, "whisper_language_id")

        return LanguageDecision(None, 0.0, "no_match")
