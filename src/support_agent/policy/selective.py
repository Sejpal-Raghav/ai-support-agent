"""Validation-only isotonic calibration and conservative policy threshold selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from support_agent.policy.signals import RawRiskFeatures, make_risk_signals
from support_agent.schemas import EscalationReason, RiskSignals


@dataclass(frozen=True)
class IsotonicProbabilityModel:
    """Serializable validation-only isotonic probability model."""

    estimator: Any
    direction: str

    def predict_one(self, value: float) -> float:
        return float(np.clip(self.estimator.predict([value])[0], 0.0, 1.0))


@dataclass(frozen=True)
class PolicyDecision:
    """The auto/escalate action and reasons produced without a test-set threshold choice."""

    action: str
    decision_reasons: tuple[EscalationReason, ...]
    signals: RiskSignals


@dataclass(frozen=True)
class PolicyThreshold:
    """Validation-selected operating point and the safety evidence supporting it."""

    safe_probability_threshold: float
    validation_auto_count: int
    validation_coverage: float
    validation_auto_error_rate: float
    validation_auto_error_upper_95: float
    validation_escalation_recall: float


def fit_retrieval_adequacy_model(
    similarities: Sequence[float], evidence_adequate: Sequence[bool]
) -> IsotonicProbabilityModel:
    """Fit adequacy from the dedicated validation-calibration annotation partition only."""
    if len(similarities) < 20 or len(similarities) != len(evidence_adequate):
        raise ValueError("retrieval adequacy calibration requires at least 20 matched labels")
    if len(set(evidence_adequate)) < 2:
        raise ValueError(
            "retrieval adequacy calibration requires both adequate and inadequate labels"
        )
    estimator = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        list(similarities), [int(value) for value in evidence_adequate]
    )
    return IsotonicProbabilityModel(estimator=estimator, direction="increasing")


def fit_safe_probability_model(
    raw_risks: Sequence[float], requires_escalation: Sequence[bool]
) -> IsotonicProbabilityModel:
    """Calibrate P(no escalation required) only on the validation-policy partition."""
    if len(raw_risks) < 50 or len(raw_risks) != len(requires_escalation):
        raise ValueError("safe-probability calibration requires at least 50 matched policy labels")
    safe_labels = [int(not value) for value in requires_escalation]
    if len(set(safe_labels)) < 2:
        raise ValueError("safe-probability calibration requires both safe and escalation labels")
    estimator = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
        raw_risks, safe_labels
    )
    return IsotonicProbabilityModel(estimator=estimator, direction="decreasing")


def _wilson_upper(errors: int, total: int, z: float = 1.959963984540054) -> float:
    if total == 0:
        return 1.0
    proportion = errors / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
    return min(1.0, (center + margin / denominator))


def tune_policy_threshold(
    safe_probabilities: Sequence[float],
    requires_escalation: Sequence[bool],
    mandatory_escalate: Sequence[bool] | None = None,
    target_error_upper: float = 0.05,
    minimum_escalation_recall: float = 0.90,
    minimum_auto_count: int = 50,
) -> PolicyThreshold:
    """Choose the highest validation coverage satisfying exact Wilson and recall safety gates."""
    if len(safe_probabilities) != len(requires_escalation) or not safe_probabilities:
        raise ValueError(
            "threshold tuning requires matched non-empty validation-policy predictions"
        )
    mandatory = list(mandatory_escalate or [False] * len(safe_probabilities))
    if len(mandatory) != len(safe_probabilities):
        raise ValueError("mandatory escalation gates must align with validation-policy predictions")
    thresholds = sorted(set(float(value) for value in safe_probabilities))
    valid: list[PolicyThreshold] = []
    risk_total = sum(requires_escalation)
    for threshold in thresholds:
        auto = [
            index
            for index, value in enumerate(safe_probabilities)
            if value >= threshold and not mandatory[index]
        ]
        auto_count = len(auto)
        errors = sum(requires_escalation[index] for index in auto)
        escalated_risk = sum(
            requires_escalation[index]
            for index, value in enumerate(safe_probabilities)
            if value < threshold or mandatory[index]
        )
        recall = 1.0 if risk_total == 0 else escalated_risk / risk_total
        upper = _wilson_upper(errors, auto_count)
        candidate = PolicyThreshold(
            safe_probability_threshold=threshold,
            validation_auto_count=auto_count,
            validation_coverage=auto_count / len(safe_probabilities),
            validation_auto_error_rate=errors / auto_count if auto_count else 0.0,
            validation_auto_error_upper_95=upper,
            validation_escalation_recall=recall,
        )
        if (
            auto_count >= minimum_auto_count
            and upper <= target_error_upper
            and recall >= minimum_escalation_recall
        ):
            valid.append(candidate)
    if not valid:
        raise ValueError("no validation threshold satisfies the fixed selective-prediction gates")
    return sorted(
        valid,
        key=lambda item: (-item.validation_coverage, -item.safe_probability_threshold),
    )[0]


def decide_action(
    features: RawRiskFeatures,
    safe_probability_model: IsotonicProbabilityModel,
    threshold: PolicyThreshold,
) -> PolicyDecision:
    """Apply mandatory escalation gates before the frozen validation-selected threshold."""
    safe_probability = safe_probability_model.predict_one(features.raw_risk_score)
    signals = make_risk_signals(features, safe_probability)
    reasons = list(features.hard_scope_categories)
    if not features.draft_validation_passed:
        reasons.append(EscalationReason.DRAFT_VALIDATION_FAILED)
    if not reasons and safe_probability < threshold.safe_probability_threshold:
        reasons.append(EscalationReason.INSUFFICIENT_EVIDENCE)
    return PolicyDecision(
        action="escalate" if reasons else "auto_handle",
        decision_reasons=tuple(reasons),
        signals=signals,
    )
