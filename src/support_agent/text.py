"""Deterministic text normalization and lightweight language detection."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
MASKED_CONTACT_PATTERN = re.compile(r"__(?:email|phone_number|contact)__", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


class LanguageResult(BaseModel):
    """Language-detection output with an explicit conservative fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    probability: float = Field(ge=0.0, le=1.0)


def normalize_customer_text(text: str) -> str:
    """Normalize only stable surface forms while preserving meaning-bearing cues."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = URL_PATTERN.sub("__url__", normalized)
    normalized = MENTION_PATTERN.sub("__user__", normalized)
    normalized = MASKED_CONTACT_PATTERN.sub("__contact__", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def extract_urls(text: str) -> tuple[str, ...]:
    """Return URLs in source order without normalization."""
    return tuple(URL_PATTERN.findall(text))


def extract_numeric_tokens(text: str) -> tuple[str, ...]:
    """Return decimal numeric tokens in source order."""
    return tuple(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))


def detect_language(text: str) -> LanguageResult:
    """Detect language deterministically; report unknown instead of raising."""
    from langdetect import DetectorFactory, detect_langs  # type: ignore[import-untyped]
    from langdetect.lang_detect_exception import LangDetectException  # type: ignore[import-untyped]

    DetectorFactory.seed = 1729
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return LanguageResult(language="unknown", probability=0.0)
    if not candidates:
        return LanguageResult(language="unknown", probability=0.0)
    best = candidates[0]
    return LanguageResult(language=best.lang, probability=float(best.prob))
