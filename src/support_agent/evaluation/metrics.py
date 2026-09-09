"""Weighted held-out metrics, calibration, bootstrap intervals, and judge agreement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from support_agent.schemas import ExampleEvaluationRecord, MetricEstimate


def weighted_intent_f1(
    records: Sequence[ExampleEvaluationRecord], labels: Sequence[str]
) -> tuple[float, dict[str, float]]:
    """Return design-weighted macro F1 and one-vs-rest per-intent F1 on golden records."""
    if not records:
        raise ValueError("intent F1 requires at least one evaluation record")
    truth = [record.gold_intent for record in records]
    predicted = [record.predicted_intent for record in records]
    weights = [record.sampling_weight for record in records]
    macro = float(
        f1_score(
            truth,
            predicted,
            labels=list(labels),
            average="macro",
            sample_weight=weights,
            zero_division=0,
        )
    )
    per_class_values = f1_score(
        truth, predicted, labels=list(labels), average=None, sample_weight=weights, zero_division=0
    )
    return macro, {
        label: float(value) for label, value in zip(labels, per_class_values, strict=True)
    }


def weighted_rate(
    records: Sequence[ExampleEvaluationRecord], predicate: Callable[[ExampleEvaluationRecord], bool]
) -> float:
    """Estimate a population rate with inverse-probability sampling weights."""
    denominator = sum(record.sampling_weight for record in records)
    if denominator == 0.0:
        raise ValueError("weighted rate requires at least one record")
    return sum(record.sampling_weight for record in records if predicate(record)) / denominator


def coverage(records: Sequence[ExampleEvaluationRecord]) -> float:
    return weighted_rate(records, lambda record: record.predicted_action == "auto_handle")


def escalation_recall(records: Sequence[ExampleEvaluationRecord]) -> float:
    risky = [record for record in records if record.gold_requires_escalation]
    if not risky:
        raise ValueError("escalation recall is undefined without risky golden examples")
    return weighted_rate(risky, lambda record: record.predicted_action == "escalate")


def auto_handle_error_rate(records: Sequence[ExampleEvaluationRecord]) -> float:
    automated = [record for record in records if record.predicted_action == "auto_handle"]
    if not automated or any(record.auto_handle_error is None for record in automated):
        raise ValueError(
            "auto-handle error requires judged composite labels for every automated record"
        )
    return weighted_rate(automated, lambda record: bool(record.auto_handle_error))


def acceptable_reply_rate(records: Sequence[ExampleEvaluationRecord]) -> float:
    automated = [record for record in records if record.predicted_action == "auto_handle"]
    if not automated or any(record.judge_scores is None for record in automated):
        raise ValueError("acceptable reply rate requires a judge score for every automated record")
    return weighted_rate(
        automated, lambda record: bool(record.judge_scores and record.judge_scores.accepted)
    )


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], weights: Sequence[float], bins: int = 10
) -> float:
    """Compute ECE = sum_b weight_b/N * |accuracy_b - confidence_b| over ten equal bins."""
    if not (len(confidences) == len(correct) == len(weights)) or not confidences:
        raise ValueError("ECE requires matched non-empty confidence, correctness, and weight lists")
    if bins != 10:
        raise ValueError("ECE uses exactly ten equal-width bins")
    total_weight = sum(weights)
    ece = 0.0
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if not indices:
            continue
        bin_weight = sum(weights[index] for index in indices)
        accuracy = sum(weights[index] * float(correct[index]) for index in indices) / bin_weight
        confidence = sum(weights[index] * confidences[index] for index in indices) / bin_weight
        ece += bin_weight / total_weight * abs(accuracy - confidence)
    return ece


def judge_agreement(records: Sequence[ExampleEvaluationRecord]) -> tuple[float, float, int]:
    """Return binary and quadratic ordinal Cohen kappa on independently blind-rated items."""
    paired = [
        record
        for record in records
        if record.judge_scores is not None and record.independent_human_scores is not None
    ]
    if len(paired) < 50:
        raise ValueError("judge agreement requires at least 50 blind independent human ratings")
    machine_binary = [
        record.judge_scores.accepted for record in paired if record.judge_scores
    ]
    human_binary = [
        record.independent_human_scores.accepted
        for record in paired
        if record.independent_human_scores
    ]
    machine_ordinal = [
        record.judge_scores.overall_ordinal for record in paired if record.judge_scores
    ]
    human_ordinal = [
        record.independent_human_scores.overall_ordinal
        for record in paired
        if record.independent_human_scores
    ]
    return (
        float(cohen_kappa_score(human_binary, machine_binary)),
        float(cohen_kappa_score(human_ordinal, machine_ordinal, weights="quadratic")),
        len(paired),
    )


def stratified_bootstrap_interval(
    records: Sequence[ExampleEvaluationRecord],
    metric_name: str,
    system_id: str,
    estimator: Callable[[Sequence[ExampleEvaluationRecord]], float],
    seed: int = 314159,
    replicates: int = 2_000,
) -> MetricEstimate:
    """Use deterministic within-stratum bootstrap percentiles for a weighted point estimator."""
    if not records:
        raise ValueError("bootstrap requires non-empty records")
    by_stratum: dict[str, list[ExampleEvaluationRecord]] = defaultdict(list)
    for record in records:
        by_stratum[record.sampling_stratum].append(record)
    generator = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(replicates):
        resampled: list[ExampleEvaluationRecord] = []
        for stratum in sorted(by_stratum):
            source = by_stratum[stratum]
            positions = generator.integers(0, len(source), size=len(source))
            resampled.extend(source[int(position)] for position in positions)
        samples.append(estimator(resampled))
    return MetricEstimate(
        metric_name=metric_name,
        system_id=system_id,
        point_estimate=estimator(records),
        ci_lower=float(np.quantile(samples, 0.025)),
        ci_upper=float(np.quantile(samples, 0.975)),
        ci_method="stratified_nonparametric_bootstrap_2000_seed_314159",
        sample_size=len(records),
        weighted=True,
    )
