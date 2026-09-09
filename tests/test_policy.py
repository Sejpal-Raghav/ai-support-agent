from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from support_agent.policy.selective import (
    IsotonicProbabilityModel,
    PolicyThreshold,
    decide_action,
    tune_policy_threshold,
)
from support_agent.policy.signals import load_hard_scope_patterns, raw_risk_features


def test_hard_scope_keyword_forces_escalation() -> None:
    patterns = load_hard_scope_patterns(Path("configs/risk_keywords.yaml"))
    features = raw_risk_features(
        "I need a refund for a duplicate charge",
        {"billing": 0.95, "other": 0.05},
        [],
        retrieval_adequacy_probability=0.95,
        customer_turn_count=1,
        severe_negative=False,
        draft_validation_passed=True,
        hard_scope_patterns=patterns,
    )
    model = IsotonicProbabilityModel(
        estimator=IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [1.0, 0.0]),
        direction="decreasing",
    )
    threshold = PolicyThreshold(0.5, 50, 0.25, 0.0, 0.05, 1.0)
    decision = decide_action(features, model, threshold)
    assert decision.action == "escalate"
    assert decision.decision_reasons[0].value == "money_or_billing_risk"


def test_threshold_uses_conservative_validation_gates() -> None:
    safe = [0.99] * 100 + [0.10] * 100
    labels = [False] * 100 + [True] * 100
    threshold = tune_policy_threshold(safe, labels)
    assert threshold.validation_auto_count == 100
    assert threshold.validation_auto_error_upper_95 <= 0.05
    assert threshold.validation_escalation_recall == 1.0


def test_threshold_respects_mandatory_escalation_gates() -> None:
    safe = [0.99] * 100 + [0.10] * 100
    labels = [False] * 100 + [True] * 100
    mandatory = [True] + [False] * 199
    threshold = tune_policy_threshold(safe, labels, mandatory)
    assert threshold.validation_auto_count == 99


def test_failed_draft_validation_escalates_even_with_high_safe_probability() -> None:
    patterns = load_hard_scope_patterns(Path("configs/risk_keywords.yaml"))
    features = raw_risk_features(
        "The app is slow",
        {"app_issue": 0.99, "other": 0.01},
        [],
        retrieval_adequacy_probability=0.99,
        customer_turn_count=1,
        severe_negative=False,
        draft_validation_passed=False,
        hard_scope_patterns=patterns,
    )
    model = IsotonicProbabilityModel(
        estimator=IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [1.0, 0.0]),
        direction="decreasing",
    )
    decision = decide_action(features, model, PolicyThreshold(0.5, 50, 0.25, 0.0, 0.05, 1.0))
    assert decision.action == "escalate"
    assert decision.decision_reasons[0].value == "draft_validation_failed"
    assert np.isfinite(decision.signals.raw_risk_score)
