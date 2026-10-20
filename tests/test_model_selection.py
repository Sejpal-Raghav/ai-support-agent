import pytest

from support_agent.intent.selection import (
    ModelCandidate,
    select_confidence_threshold,
    select_model_candidate,
)


def test_model_selection_prefers_f1_outside_tie_margin() -> None:
    fast = ModelCandidate("fast", "fast", 0.80, 10.0)
    accurate = ModelCandidate("accurate", "accurate", 0.81, 100.0)
    assert select_model_candidate([fast, accurate]).model_id == "accurate"


def test_model_selection_prefers_latency_inside_tie_margin() -> None:
    fast = ModelCandidate("fast", "fast", 0.806, 10.0)
    accurate = ModelCandidate("accurate", "accurate", 0.81, 100.0)
    assert select_model_candidate([accurate, fast]).model_id == "fast"


def test_confidence_threshold_uses_lowest_safe_validation_cutoff() -> None:
    confidences = [0.95] * 20 + [0.70] * 5
    correct = [True] * 19 + [False] + [False] * 5
    assert select_confidence_threshold(confidences, correct) == pytest.approx(0.95)


def test_confidence_threshold_fails_closed_when_support_is_too_small() -> None:
    with pytest.raises(ValueError, match="no confidence threshold"):
        select_confidence_threshold([0.99] * 10, [True] * 10)


def test_zero_tie_margin_selects_the_best_f1() -> None:
    candidate = ModelCandidate("best", "best", 0.81, 10.0)
    assert select_model_candidate([candidate], f1_tie_margin=0.0) == candidate
