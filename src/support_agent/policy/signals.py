"""Explicit risk-signal extraction with no hidden model decisions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from support_agent.schemas import EscalationReason, RetrievedCase, RiskSignals


@dataclass(frozen=True)
class RawRiskFeatures:
    """Uncalibrated features used by the documented fixed risk formula."""

    intent_max_probability: float
    intent_entropy_normalized: float
    top_retrieval_similarity: float | None
    retrieval_adequacy_probability: float
    retrieval_gap_risk: float
    multi_turn_risk: float
    severe_negative_risk: float
    hard_scope_categories: tuple[EscalationReason, ...]
    draft_validation_passed: bool
    raw_risk_score: float


def load_hard_scope_patterns(path: Path) -> dict[EscalationReason, tuple[re.Pattern[str], ...]]:
    """Load the reviewed risk-keyword configuration and reject unknown policy categories."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("hard_scope_categories"), dict):
        raise ValueError("risk keyword configuration has no hard_scope_categories mapping")
    patterns: dict[EscalationReason, tuple[re.Pattern[str], ...]] = {}
    for name, expressions in raw["hard_scope_categories"].items():
        if not isinstance(expressions, list) or not all(
            isinstance(item, str) for item in expressions
        ):
            raise ValueError(f"risk patterns for {name!r} must be a list of strings")
        try:
            reason = EscalationReason(str(name))
        except ValueError as error:
            raise ValueError(f"unknown hard-scope escalation reason {name!r}") from error
        patterns[reason] = tuple(
            re.compile(expression, re.IGNORECASE) for expression in expressions
        )
    return patterns


def _normalized_entropy(probabilities: Mapping[str, float]) -> float:
    values = [value for value in probabilities.values() if value > 0.0]
    if len(values) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in values)
    return min(1.0, max(0.0, entropy / math.log(len(values))))


def _hard_scope_categories(
    text: str, patterns: Mapping[EscalationReason, Sequence[re.Pattern[str]]]
) -> tuple[EscalationReason, ...]:
    return tuple(
        reason
        for reason, category_patterns in sorted(patterns.items(), key=lambda item: item[0].value)
        if any(pattern.search(text) for pattern in category_patterns)
    )


def raw_risk_features(
    customer_text: str,
    intent_probabilities: Mapping[str, float],
    retrieved_cases: Sequence[RetrievedCase],
    retrieval_adequacy_probability: float,
    customer_turn_count: int,
    severe_negative: bool,
    draft_validation_passed: bool,
    hard_scope_patterns: Mapping[EscalationReason, Sequence[re.Pattern[str]]],
) -> RawRiskFeatures:
    """Apply the Phase 6 fixed formula before validation-only safe-probability calibration.

    ``raw_risk = clamp(.35*intent + .20*entropy + .25*retrieval + .10*gap
    + .10*multi_turn + .15*severe_negative, 0, 1)``. Hard scope and failed draft
    validation are not scored away: they are mandatory escalation conditions in the policy.
    """
    if not intent_probabilities:
        raise ValueError("intent probabilities cannot be empty")
    if any(value < 0.0 or value > 1.0 for value in intent_probabilities.values()):
        raise ValueError("intent probabilities must lie in [0, 1]")
    if not 0.0 <= retrieval_adequacy_probability <= 1.0:
        raise ValueError("retrieval adequacy probability must lie in [0, 1]")
    top_similarity = retrieved_cases[0].similarity if retrieved_cases else None
    if len(retrieved_cases) < 2:
        gap_risk = 1.0
    else:
        gap = retrieved_cases[0].similarity - retrieved_cases[1].similarity
        gap_risk = 1.0 - min(max(gap, 0.0) / 0.15, 1.0)
    intent_max = max(intent_probabilities.values())
    intent_risk = 1.0 - intent_max
    entropy = _normalized_entropy(intent_probabilities)
    retrieval_risk = 1.0 - retrieval_adequacy_probability
    multi_turn_risk = 1.0 if customer_turn_count > 1 else 0.0
    severe_risk = 1.0 if severe_negative else 0.0
    raw = min(
        1.0,
        max(
            0.0,
            0.35 * intent_risk
            + 0.20 * entropy
            + 0.25 * retrieval_risk
            + 0.10 * gap_risk
            + 0.10 * multi_turn_risk
            + 0.15 * severe_risk,
        ),
    )
    return RawRiskFeatures(
        intent_max_probability=intent_max,
        intent_entropy_normalized=entropy,
        top_retrieval_similarity=top_similarity,
        retrieval_adequacy_probability=retrieval_adequacy_probability,
        retrieval_gap_risk=gap_risk,
        multi_turn_risk=multi_turn_risk,
        severe_negative_risk=severe_risk,
        hard_scope_categories=_hard_scope_categories(customer_text, hard_scope_patterns),
        draft_validation_passed=draft_validation_passed,
        raw_risk_score=raw,
    )


def make_risk_signals(features: RawRiskFeatures, calibrated_safe_probability: float) -> RiskSignals:
    """Convert transparent raw features into the persisted policy schema."""
    if not 0.0 <= calibrated_safe_probability <= 1.0:
        raise ValueError("calibrated safe probability must lie in [0, 1]")
    return RiskSignals(
        intent_max_probability=features.intent_max_probability,
        intent_entropy_normalized=features.intent_entropy_normalized,
        intent_risk=1.0 - features.intent_max_probability,
        top_retrieval_similarity=features.top_retrieval_similarity,
        retrieval_adequacy_probability=features.retrieval_adequacy_probability,
        retrieval_risk=1.0 - features.retrieval_adequacy_probability,
        retrieval_gap_risk=features.retrieval_gap_risk,
        multi_turn_risk=features.multi_turn_risk,
        severe_negative_risk=features.severe_negative_risk,
        hard_scope_categories=features.hard_scope_categories,
        draft_validation_passed=features.draft_validation_passed,
        raw_risk_score=features.raw_risk_score,
        calibrated_safe_probability=calibrated_safe_probability,
    )
