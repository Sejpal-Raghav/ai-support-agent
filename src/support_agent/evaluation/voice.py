"""Held-out voice evaluation with corpus WER and explicit assignment acceptance gates."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceEvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recording_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    noise: Literal["quiet", "moderate", "noisy"]
    split: Literal["held_out"]
    reference: str = Field(min_length=1)
    transcript: str
    asr_model_version: str = Field(min_length=1)
    duration_ms: int = Field(gt=0, le=30500)
    latency_ms: int = Field(ge=0)


def words(text: str) -> list[str]:
    """Case-fold and ignore punctuation consistently for reference and hypothesis."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return Levenshtein word edits and reference length, counting empty ASR as deletions."""
    expected, actual = words(reference), words(hypothesis)
    if not expected:
        raise ValueError("voice references must contain at least one English word")
    previous = list(range(len(actual) + 1))
    for index, word in enumerate(expected, start=1):
        current = [index]
        for position, candidate in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[position] + 1,
                    previous[position - 1] + (word != candidate),
                )
            )
        previous = current
    return previous[-1], len(expected)


def evaluate_voice(records: Sequence[VoiceEvaluationRecord]) -> dict[str, object]:
    """Aggregate edits rather than averaging clip WER; do not tune ASR on held-out audio."""
    if not records:
        raise ValueError("voice evaluation requires held-out recordings")
    if len({record.recording_id for record in records}) != len(records):
        raise ValueError("voice recording IDs must be unique")
    versions = {record.asr_model_version for record in records}
    if len(versions) != 1:
        raise ValueError("evaluate exactly one ASR model version at a time")
    counts = Counter(record.noise for record in records)
    groups: dict[str, object] = {}
    overall_edits = overall_words = 0
    for noise in ("quiet", "moderate", "noisy"):
        edits = reference_words = 0
        for record in records:
            if record.noise == noise:
                errors, length = word_errors(record.reference, record.transcript)
                edits += errors
                reference_words += length
        groups[noise] = {
            "recordings": counts[noise],
            "word_errors": edits,
            "reference_words": reference_words,
            "wer": edits / reference_words if reference_words else None,
        }
        overall_edits += edits
        overall_words += reference_words
    wer = overall_edits / overall_words
    gates = {
        "fifty_recordings": len(records) == 50,
        "five_speakers": len({record.speaker_id for record in records}) == 5,
        "noise_distribution": counts == {"quiet": 25, "moderate": 15, "noisy": 10},
        "wer_at_most_30_percent": wer <= 0.30,
    }
    return {
        "schema_version": "1",
        "asr_model_version": next(iter(versions)),
        "recordings": len(records),
        "speakers": len({record.speaker_id for record in records}),
        "word_errors": overall_edits,
        "reference_words": overall_words,
        "wer": wer,
        "by_noise": groups,
        "acceptance_gates": gates,
        "accepted": all(gates.values()),
        "normalization": "lowercase_ascii_words_preserve_apostrophes_v1",
    }
