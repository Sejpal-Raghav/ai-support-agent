"""Blind annotation packet construction and completed-label validation."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from support_agent.hashing import canonical_sha256
from support_agent.schemas import (
    EscalationReason,
    GoldenExample,
    IntentTaxonomy,
    RawMessage,
    ReconstructedThread,
)


class DecisionPoint(BaseModel):
    """An inference-time customer turn with only past context available."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    incoming_tweet_id: str
    brand_id: str
    context_messages: tuple[RawMessage, ...]
    provisional_intent: str
    edge_flag: bool
    edge_reasons: tuple[str, ...]
    turn_count_in_thread: int = Field(default=1, ge=1)


class AnnotationTask(BaseModel):
    """Label-free packet row delivered to a human annotator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str
    thread_id: str
    incoming_tweet_id: str
    brand_id: str
    context_messages: tuple[RawMessage, ...]
    provisional_intent: str
    sampling_stratum: str
    stage_one_probability: float = Field(gt=0.0, le=1.0)
    stage_two_probability: float = Field(gt=0.0, le=1.0)
    inclusion_probability: float = Field(gt=0.0, le=1.0)
    sampling_weight: float = Field(gt=0.0)
    annotation_partition: str


class AnnotationSample(BaseModel):
    """Frozen packet plus allocation provenance for a validation or golden draw."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_name: str
    tasks: tuple[AnnotationTask, ...]
    frame_counts: dict[str, int]
    selected_counts: dict[str, int]
    seed: int


class AnnotationAudit(BaseModel):
    """Completed-label quality checks required before model tuning or evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=0)
    partitions: dict[str, int]
    invalid_intent_ids: tuple[str, ...]
    invalid_escalation_examples: tuple[str, ...]
    duplicate_example_ids: tuple[str, ...]


_EDGE_PATTERN = re.compile(
    r"\b(refund|charge|charged|payment|bill|billing|card|account|password|login|"
    r"security|fraud|police|legal|injury|emergency|suicide)\b",
    flags=re.IGNORECASE,
)


def build_decision_points(
    threads: Sequence[ReconstructedThread], provisional_intents: Mapping[str, str]
) -> list[DecisionPoint]:
    """Create leakage-safe customer decision points that expose no future messages."""
    points: list[DecisionPoint] = []
    for thread in threads:
        prior_inbound = 0
        for index, message in enumerate(thread.messages):
            if not message.inbound:
                continue
            prior_inbound += 1
            text = message.text
            reasons: list[str] = []
            if prior_inbound > 1:
                reasons.append("multi_turn_customer_request")
            if _EDGE_PATTERN.search(text):
                reasons.append("hard_risk_keyword")
            if provisional_intents.get(message.tweet_id, "other") == "other":
                reasons.append("provisional_other")
            points.append(
                DecisionPoint(
                    thread_id=thread.thread_id,
                    incoming_tweet_id=message.tweet_id,
                    brand_id=thread.brand_id,
                    context_messages=tuple(thread.messages[: index + 1]),
                    provisional_intent=provisional_intents.get(message.tweet_id, "other"),
                    edge_flag=bool(reasons),
                    edge_reasons=tuple(reasons),
                    turn_count_in_thread=prior_inbound,
                )
            )
    return points


def _task_id(point: DecisionPoint) -> str:
    return canonical_sha256(f"annotation:{point.thread_id}:{point.incoming_tweet_id}")[:20]


def _one_turn_per_thread(
    points: Sequence[DecisionPoint], seed_namespace: str, seed: int
) -> list[tuple[DecisionPoint, float]]:
    by_thread: dict[str, list[DecisionPoint]] = defaultdict(list)
    for point in points:
        by_thread[point.thread_id].append(point)
    selected: list[tuple[DecisionPoint, float]] = []
    for thread_id, candidates in sorted(by_thread.items()):
        ordered = sorted(
            candidates,
            key=lambda item: canonical_sha256(
                f"{seed_namespace}:{seed}:{thread_id}:{item.incoming_tweet_id}"
            ),
        )
        selected.append((ordered[0], 1.0 / len(candidates)))
    return selected


def _allocate_square_root(
    frame_counts: Mapping[str, int], total: int, minimum: int
) -> dict[str, int]:
    nonempty = {stratum: count for stratum, count in frame_counts.items() if count > 0}
    if total > sum(nonempty.values()):
        raise ValueError("sample size exceeds available annotation frame")
    allocations = {stratum: min(minimum, count) for stratum, count in nonempty.items()}
    remaining = total - sum(allocations.values())
    while remaining > 0:
        candidates = {
            stratum: math.sqrt(count)
            for stratum, count in nonempty.items()
            if allocations[stratum] < count
        }
        if not candidates:
            raise AssertionError("allocation exhausted before filling requested sample")
        denominator = sum(candidates.values())
        ideal = {
            stratum: remaining * weight / denominator
            for stratum, weight in candidates.items()
        }
        additions = {
            stratum: min(int(math.floor(value)), nonempty[stratum] - allocations[stratum])
            for stratum, value in ideal.items()
        }
        added = sum(additions.values())
        for stratum, value in additions.items():
            allocations[stratum] += value
        remaining -= added
        if remaining == 0:
            break
        for stratum in sorted(
            candidates,
            key=lambda item: (-(ideal[item] - math.floor(ideal[item])), item),
        ):
            if remaining == 0:
                break
            if allocations[stratum] < nonempty[stratum]:
                allocations[stratum] += 1
                remaining -= 1
    return allocations


def _stratified_tasks(
    points_with_stage_one: Sequence[tuple[DecisionPoint, float]],
    sample_name: str,
    partition: str,
    size: int,
    seed: int,
    minimum_per_stratum: int,
) -> AnnotationSample:
    by_stratum: dict[str, list[tuple[DecisionPoint, float]]] = defaultdict(list)
    for point, stage_one_probability in points_with_stage_one:
        stratum = f"{point.provisional_intent}|edge={str(point.edge_flag).lower()}"
        by_stratum[stratum].append((point, stage_one_probability))
    frame_counts = {stratum: len(items) for stratum, items in by_stratum.items()}
    allocations = _allocate_square_root(frame_counts, size, minimum_per_stratum)
    tasks: list[AnnotationTask] = []
    for stratum, count in sorted(allocations.items()):
        frame = sorted(
            by_stratum[stratum],
            key=lambda item: canonical_sha256(
                f"{sample_name}:{seed}:{item[0].thread_id}:{item[0].incoming_tweet_id}"
            ),
        )
        for point, stage_one_probability in frame[:count]:
            stage_two_probability = count / len(frame)
            inclusion_probability = stage_one_probability * stage_two_probability
            tasks.append(
                AnnotationTask(
                    example_id=_task_id(point),
                    thread_id=point.thread_id,
                    incoming_tweet_id=point.incoming_tweet_id,
                    brand_id=point.brand_id,
                    context_messages=point.context_messages,
                    provisional_intent=point.provisional_intent,
                    sampling_stratum=stratum,
                    stage_one_probability=stage_one_probability,
                    stage_two_probability=stage_two_probability,
                    inclusion_probability=inclusion_probability,
                    sampling_weight=1.0 / inclusion_probability,
                    annotation_partition=partition,
                )
            )
    return AnnotationSample(
        sample_name=sample_name,
        tasks=tuple(sorted(tasks, key=lambda item: item.example_id)),
        frame_counts=dict(sorted(frame_counts.items())),
        selected_counts=dict(sorted(allocations.items())),
        seed=seed,
    )


def select_validation_annotation_set(
    points: Sequence[DecisionPoint], size: int = 400, seed: int = 1729
) -> tuple[AnnotationSample, AnnotationSample, AnnotationSample]:
    """Create non-overlapping 100/100/200 stratified validation partitions."""
    if size != 400:
        raise ValueError("validation annotation size must be exactly 400")
    stage_one = _one_turn_per_thread(points, "validation_turn", seed)
    model_selection = _stratified_tasks(
        stage_one,
        "validation_model_selection",
        "validation_model_selection",
        100,
        seed,
        3,
    )
    selected_threads = {task.thread_id for task in model_selection.tasks}
    calibration_frame = [item for item in stage_one if item[0].thread_id not in selected_threads]
    calibration = _stratified_tasks(
        calibration_frame,
        "validation_calibration",
        "validation_calibration",
        100,
        seed,
        3,
    )
    selected_threads.update(task.thread_id for task in calibration.tasks)
    policy_frame = [item for item in stage_one if item[0].thread_id not in selected_threads]
    policy = _stratified_tasks(
        policy_frame,
        "validation_policy",
        "validation_policy",
        200,
        seed,
        3,
    )
    return model_selection, calibration, policy


def select_golden_annotation_set(
    points: Sequence[DecisionPoint], size: int = 200, seed: int = 2718
) -> AnnotationSample:
    """Select exactly 100 edge and 100 non-edge held-out test tasks when available."""
    if size != 200:
        raise ValueError("golden annotation size must be exactly 200")
    stage_one = _one_turn_per_thread(points, "gold_turn", seed)
    edge = [item for item in stage_one if item[0].edge_flag]
    non_edge = [item for item in stage_one if not item[0].edge_flag]
    edge_size = min(100, len(edge))
    non_edge_size = min(100, len(non_edge))
    shortage = size - edge_size - non_edge_size
    if shortage > 0:
        if len(edge) - edge_size >= shortage:
            edge_size += shortage
        elif len(non_edge) - non_edge_size >= shortage:
            non_edge_size += shortage
        else:
            raise ValueError("golden frame contains fewer than 200 eligible test threads")
    edge_sample = _stratified_tasks(edge, "golden_edge", "golden_test", edge_size, seed, 5)
    non_edge_sample = _stratified_tasks(
        non_edge, "golden_non_edge", "golden_test", non_edge_size, seed, 5
    )
    tasks = tuple(
        sorted((*edge_sample.tasks, *non_edge_sample.tasks), key=lambda item: item.example_id)
    )
    return AnnotationSample(
        sample_name="golden_test",
        tasks=tasks,
        frame_counts={
            **{f"edge::{key}": value for key, value in edge_sample.frame_counts.items()},
            **{f"non_edge::{key}": value for key, value in non_edge_sample.frame_counts.items()},
        },
        selected_counts={
            **{f"edge::{key}": value for key, value in edge_sample.selected_counts.items()},
            **{f"non_edge::{key}": value for key, value in non_edge_sample.selected_counts.items()},
        },
        seed=seed,
    )


def write_annotation_tasks(sample: AnnotationSample, output_path: Path) -> None:
    """Write a label-free CSV packet; context is JSON to avoid spreadsheet ambiguity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example_id",
        "thread_id",
        "incoming_tweet_id",
        "brand_id",
        "context_messages_json",
        "provisional_intent",
        "sampling_stratum",
        "stage_one_probability",
        "stage_two_probability",
        "inclusion_probability",
        "sampling_weight",
        "annotation_partition",
        "intent_label",
        "intent_ambiguous",
        "requires_escalation",
        "escalation_reasons_json",
        "evidence_adequate",
        "required_reply_elements_json",
        "forbidden_reply_elements_json",
        "annotation_notes",
        "annotator_id",
        "annotation_guideline_version",
        "annotated_at",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in sample.tasks:
            writer.writerow(
                {
                    "example_id": task.example_id,
                    "thread_id": task.thread_id,
                    "incoming_tweet_id": task.incoming_tweet_id,
                    "brand_id": task.brand_id,
                    "context_messages_json": json.dumps(
                        [message.model_dump(mode="json") for message in task.context_messages]
                    ),
                    "provisional_intent": task.provisional_intent,
                    "sampling_stratum": task.sampling_stratum,
                    "stage_one_probability": task.stage_one_probability,
                    "stage_two_probability": task.stage_two_probability,
                    "inclusion_probability": task.inclusion_probability,
                    "sampling_weight": task.sampling_weight,
                    "annotation_partition": task.annotation_partition,
                }
            )


def _json_list(value: str, field_name: str) -> list[str]:
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError(f"{field_name} must be a JSON list of strings")
    return decoded


def load_completed_annotations(path: Path) -> list[GoldenExample]:
    """Load completed human labels into strict schema records."""
    records: list[GoldenExample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            context = tuple(
                RawMessage.model_validate(item)
                for item in json.loads(row["context_messages_json"])
            )
            evidence_raw = row["evidence_adequate"].strip().lower()
            if evidence_raw not in {"", "true", "false"}:
                raise ValueError("evidence_adequate must be blank, true, or false")
            evidence_adequate = None if evidence_raw == "" else evidence_raw == "true"
            intent_ambiguous = row["intent_ambiguous"].strip().lower()
            requires_escalation = row["requires_escalation"].strip().lower()
            if intent_ambiguous not in {"true", "false"}:
                raise ValueError("intent_ambiguous must be true or false")
            if requires_escalation not in {"true", "false"}:
                raise ValueError("requires_escalation must be true or false")
            partition = row["annotation_partition"]
            if partition not in {
                "validation_model_selection",
                "validation_calibration",
                "validation_policy",
                "golden_test",
            }:
                raise ValueError("annotation_partition is not a supported packet name")
            reasons = tuple(
                EscalationReason(item)
                for item in _json_list(row["escalation_reasons_json"], "escalation_reasons_json")
            )
            records.append(
                GoldenExample(
                    example_id=row["example_id"],
                    thread_id=row["thread_id"],
                    incoming_tweet_id=row["incoming_tweet_id"],
                    brand_id=row["brand_id"],
                    context_messages=context,
                    intent_label=row["intent_label"].strip(),
                    intent_ambiguous=intent_ambiguous == "true",
                    requires_escalation=requires_escalation == "true",
                    escalation_reasons=reasons,
                    evidence_adequate=evidence_adequate,
                    required_reply_elements=tuple(
                        _json_list(
                            row["required_reply_elements_json"], "required_reply_elements_json"
                        )
                    ),
                    forbidden_reply_elements=tuple(
                        _json_list(
                            row["forbidden_reply_elements_json"], "forbidden_reply_elements_json"
                        )
                    ),
                    annotation_notes=row["annotation_notes"],
                    sampling_stratum=row["sampling_stratum"],
                    stage_one_probability=float(row["stage_one_probability"]),
                    stage_two_probability=float(row["stage_two_probability"]),
                    inclusion_probability=float(row["inclusion_probability"]),
                    sampling_weight=float(row["sampling_weight"]),
                    annotation_partition=cast(Literal[
                        "validation_model_selection",
                        "validation_calibration",
                        "validation_policy",
                        "golden_test",
                    ], partition),
                    annotator_id=row["annotator_id"],
                    annotation_guideline_version=row["annotation_guideline_version"],
                    annotated_at=_parse_utc_datetime(row["annotated_at"]),
                )
            )
    return records


def _parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("annotated_at must include a timezone offset")
    return parsed.astimezone(UTC)


def validate_annotation_consistency(
    records: Sequence[GoldenExample], taxonomy: IntentTaxonomy
) -> AnnotationAudit:
    """Reject label, escalation, and partition inconsistencies before downstream use."""
    valid_intents = {intent.intent_id for intent in taxonomy.intents}
    invalid_intents = sorted(
        {record.intent_label for record in records if record.intent_label not in valid_intents}
    )
    invalid_escalations = tuple(
        record.example_id
        for record in records
        if (record.requires_escalation and not record.escalation_reasons)
        or (not record.requires_escalation and record.escalation_reasons)
    )
    ids = [record.example_id for record in records]
    duplicate_ids = tuple(sorted({identifier for identifier in ids if ids.count(identifier) > 1}))
    partitions: dict[str, int] = defaultdict(int)
    for record in records:
        partitions[record.annotation_partition] += 1
    return AnnotationAudit(
        record_count=len(records),
        partitions=dict(sorted(partitions.items())),
        invalid_intent_ids=tuple(invalid_intents),
        invalid_escalation_examples=invalid_escalations,
        duplicate_example_ids=duplicate_ids,
    )
