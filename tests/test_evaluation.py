from __future__ import annotations

import pytest

from support_agent.evaluation.metrics import (
    expected_calibration_error,
    judge_agreement,
    weighted_intent_f1,
)
from support_agent.schemas import ExampleEvaluationRecord, JudgeScores


def _record(index: int, correct: bool) -> ExampleEvaluationRecord:
    score = JudgeScores(
        relevance=4, grounding=4, usefulness=4, tone=4, safety=4,
        critical_violation=False, critical_violation_category=None, rationale="fixture",
        accepted=True, overall_ordinal=4,
    )
    return ExampleEvaluationRecord(
        example_id=f"example-{index}", thread_id=f"thread-{index}", system_id="main",
        sampling_stratum="intent_a|edge=false", sampling_weight=1.0,
        gold_intent="intent_a", predicted_intent="intent_a" if correct else "intent_b",
        intent_correct=correct, gold_requires_escalation=False, predicted_action="auto_handle",
        judge_scores=score, primary_human_scores=None, independent_human_scores=None,
        composite_safe_to_auto_handle=correct, auto_handle_error=not correct,
    )


def test_weighted_f1_and_ece_are_deterministic() -> None:
    macro, per_class = weighted_intent_f1(
        [_record(1, True), _record(2, False)], ["intent_a", "intent_b"]
    )
    assert 0.0 <= macro <= 1.0
    assert set(per_class) == {"intent_a", "intent_b"}
    assert expected_calibration_error([0.9, 0.1], [True, False], [1.0, 1.0]) == pytest.approx(0.1)


def test_judge_agreement_requires_and_scores_fifty_independent_ratings() -> None:
    paired = []
    for index in range(50):
        record = _record(index, True)
        score = record.judge_scores
        if index % 2:
            score = (
                score.model_copy(update={"accepted": False, "overall_ordinal": 2})
                if score
                else None
            )
        paired.append(
            record.model_copy(update={"judge_scores": score, "independent_human_scores": score})
        )
    binary, ordinal, count = judge_agreement(paired)
    assert (binary, ordinal, count) == (1.0, 1.0, 50)
