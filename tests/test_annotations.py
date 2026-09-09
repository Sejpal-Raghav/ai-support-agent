from __future__ import annotations

from datetime import UTC, datetime, timedelta

from support_agent.evaluation.annotations import (
    DecisionPoint,
    select_golden_annotation_set,
    select_validation_annotation_set,
)
from support_agent.schemas import RawMessage


def _point(index: int, edge: bool) -> DecisionPoint:
    message = RawMessage(
        tweet_id=f"tweet-{index}",
        author_id=f"customer-{index}",
        inbound=True,
        created_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        text=f"Please help with issue {index}",
        response_tweet_ids=(),
        in_response_to_tweet_id=None,
    )
    return DecisionPoint(
        thread_id=f"thread-{index}",
        incoming_tweet_id=message.tweet_id,
        brand_id="BrandHelp",
        context_messages=(message,),
        provisional_intent=f"intent_{index % 5}",
        edge_flag=edge,
        edge_reasons=("fixture_edge",) if edge else (),
    )


def test_validation_packets_are_disjoint_and_exactly_sized() -> None:
    points = [_point(index, edge=index % 3 == 0) for index in range(600)]
    selection, calibration, policy = select_validation_annotation_set(points)
    assert [len(sample.tasks) for sample in (selection, calibration, policy)] == [100, 100, 200]
    thread_sets = [
        {task.thread_id for task in sample.tasks} for sample in (selection, calibration, policy)
    ]
    assert not thread_sets[0] & thread_sets[1]
    assert not thread_sets[0] & thread_sets[2]
    assert not thread_sets[1] & thread_sets[2]
    assert all(task.sampling_weight > 0.0 for task in policy.tasks)


def test_golden_packet_balances_edge_and_non_edge() -> None:
    points = [_point(index, edge=index < 250) for index in range(500)]
    sample = select_golden_annotation_set(points)
    assert len(sample.tasks) == 200
    assert sum("edge=true" in task.sampling_stratum for task in sample.tasks) == 100
