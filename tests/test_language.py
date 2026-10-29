from __future__ import annotations

import pytest

from app.domain.enums import Language
from app.domain.models import Transcription
from app.services.language import LanguageDetector


@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector()


@pytest.mark.parametrize(
    ("text", "expected", "reason"),
    [
        ("English", Language.ENGLISH, "keyword_exact"),
        ("english please", Language.ENGLISH, "keyword_exact"),
        ("Sinhala", Language.SINHALA, "keyword_exact"),
        ("සිංහල", Language.SINHALA, "keyword_exact"),
        ("sinhalese", Language.SINHALA, "keyword_exact"),
        # Whisper mishears single words constantly; fuzzy matching absorbs it.
        ("sinhalaa", Language.SINHALA, "keyword_fuzzy"),
        ("englsh", Language.ENGLISH, "keyword_fuzzy"),
        # IVR style fallback.
        ("one", Language.SINHALA, "numeric_choice"),
        ("2", Language.ENGLISH, "numeric_choice"),
        # Any Sinhala script at all is a strong signal.
        ("මට උදව් කරන්න", Language.SINHALA, "sinhala_script"),
    ],
)
def test_resolves_language(
    detector: LanguageDetector, text: str, expected: Language, reason: str
) -> None:
    decision = detector.detect(Transcription(text=text))

    assert decision.language is expected
    assert decision.reason == reason
    assert decision.confidence > 0.5


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty_transcript"),
        ("   ", "empty_transcript"),
        ("...", "no_tokens"),
        ("I would like to know about the campus fees", "no_match"),
        ("sinhala or english", "ambiguous_both_languages"),
    ],
)
def test_refuses_ambiguous_input(
    detector: LanguageDetector, text: str, reason: str
) -> None:
    decision = detector.detect(Transcription(text=text))

    assert decision.language is None
    assert decision.resolved is False
    assert decision.reason == reason


def test_falls_back_to_whisper_language_id(detector: LanguageDetector) -> None:
    decision = detector.detect(
        Transcription(
            text="mata denna",
            detected_language="si",
            language_probability=0.93,
        )
    )

    assert decision.language is Language.SINHALA
    assert decision.reason == "whisper_language_id"


def test_ignores_low_confidence_whisper_language_id(
    detector: LanguageDetector,
) -> None:
    decision = detector.detect(
        Transcription(
            text="hmm okay",
            detected_language="si",
            language_probability=0.41,
        )
    )

    assert decision.resolved is False


def test_normalises_zero_width_joiners(detector: LanguageDetector) -> None:
    # The confirm prompt spelling uses a ZWJ; a transcript may not.
    decision = detector.detect(Transcription(text="ඉංග්‍රීසි"))

    assert decision.language is Language.ENGLISH


@pytest.mark.parametrize(
    "text",
    ["ඉංග්ලිෂ්", "ඉන්ග්ලිෂ්", "ඉංග්ලීසි", "ඉංග්ලිශ්"],
)
def test_sinhala_script_spellings_of_english_are_english(
    detector: LanguageDetector, text: str
) -> None:
    """Whisper writes a spoken "English" in Sinhala script when it guesses si.

    This used to be classified as Sinhala by the script heuristic.
    """

    decision = detector.detect(Transcription(text=text))

    assert decision.language is Language.ENGLISH


def test_unknown_sinhala_script_spelling_of_english_is_not_guessed(
    detector: LanguageDetector,
) -> None:
    # Close to an English transliteration but not in the keyword list: the
    # safe answer is to ask again, not to pick Sinhala.
    decision = detector.detect(Transcription(text="ඉංග්ලිස්"))

    assert decision.language is not Language.SINHALA


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sin hala", Language.SINHALA),
        ("Eng lish", Language.ENGLISH),
        ("in glish", Language.ENGLISH),
    ],
)
def test_keywords_split_in_two_by_whisper(
    detector: LanguageDetector, text: str, expected: Language
) -> None:
    decision = detector.detect(Transcription(text=text))

    assert decision.language is expected
    assert decision.reason == "keyword_joined"


@pytest.mark.parametrize(
    "text",
    ["singles", "singing", "finnish", "england", "ingles", "sinner"],
)
def test_similar_looking_words_are_not_a_language(
    detector: LanguageDetector, text: str
) -> None:
    assert detector.detect(Transcription(text=text)).language is None


@pytest.mark.parametrize(
    "text",
    [
        "I have two questions",
        "I need one more minute please",
        "the fee is two thousand rupees",
    ],
)
def test_numeric_choice_ignored_inside_a_sentence(
    detector: LanguageDetector, text: str
) -> None:
    assert detector.detect(Transcription(text=text)).language is None


def test_numeric_choice_still_works_in_a_short_reply(
    detector: LanguageDetector,
) -> None:
    decision = detector.detect(Transcription(text="number two"))

    assert decision.language is Language.ENGLISH
    assert decision.reason == "numeric_choice"


def test_long_input_is_bounded(detector: LanguageDetector) -> None:
    # 2,000 characters of noise must not turn into thousands of fuzzy matches.
    decision = detector.detect(Transcription(text="blah " * 400))

    assert decision.language is None
