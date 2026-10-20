"""Validation-only operating-point and transformer challenger selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    artifact_dir: str
    validation_macro_f1: float
    warm_p95_latency_ms: float


def select_model_candidate(
    candidates: Sequence[ModelCandidate], *, f1_tie_margin: float = 0.005
) -> ModelCandidate:
    """Select by validation macro-F1, using latency only inside the declared tie margin."""
    if not candidates:
        raise ValueError("model selection requires at least one candidate")
    if f1_tie_margin < 0:
        raise ValueError("f1_tie_margin must be non-negative")
    best_f1 = max(candidate.validation_macro_f1 for candidate in candidates)
    competitive = [
        candidate
        for candidate in candidates
        if candidate.validation_macro_f1 == best_f1
        or best_f1 - candidate.validation_macro_f1 < f1_tie_margin
    ]
    return min(
        competitive,
        key=lambda candidate: (
            candidate.warm_p95_latency_ms,
            -candidate.validation_macro_f1,
            candidate.model_id,
        ),
    )


def select_confidence_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    target_precision: float = 0.90,
    minimum_accepted: int = 20,
) -> float:
    """Return the lowest validation threshold meeting the fixed selective-precision target."""
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("threshold selection requires matched non-empty validation outcomes")
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be within [0, 1]")
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    if any(not 0.0 <= confidence <= 1.0 for confidence in confidences):
        raise ValueError("confidences must be within [0, 1]")

    for threshold in sorted(set(float(value) for value in confidences)):
        accepted = [index for index, value in enumerate(confidences) if value >= threshold]
        if len(accepted) < minimum_accepted:
            continue
        precision = sum(bool(correct[index]) for index in accepted) / len(accepted)
        if precision >= target_precision:
            return threshold
    raise ValueError("no confidence threshold meets the validation precision and support gates")
