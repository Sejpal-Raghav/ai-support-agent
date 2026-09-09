from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from support_agent.data.brand_selection import (
    BrandStatistics,
    NoEligibleBrandError,
    PilotMetrics,
    classify_resolution_proxy,
    load_brand_selection_settings,
    load_risk_rules,
    select_brand,
)
from support_agent.schemas import RawMessage, ReconstructedThread, ResolutionProxy


def _thread(messages: tuple[RawMessage, ...]) -> ReconstructedThread:
    return ReconstructedThread(
        thread_id="thread",
        brand_id="BrandHelp",
        root_tweet_ids=("1",),
        messages=messages,
        started_at=messages[0].created_at,
        ended_at=messages[-1].created_at,
        customer_turn_count=sum(message.inbound for message in messages),
        brand_turn_count=sum(not message.inbound for message in messages),
        is_multi_turn=False,
        has_cycle=False,
        dangling_reference_count=0,
        reconstruction_version="1",
    )


def _message(tweet_id: str, inbound: bool, minute: int, parent: str | None) -> RawMessage:
    return RawMessage(
        tweet_id=tweet_id,
        author_id="customer" if inbound else "BrandHelp",
        inbound=inbound,
        created_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=minute),
        text="Thanks, that worked" if inbound and minute > 1 else "Try restarting the app",
        response_tweet_ids=(),
        in_response_to_tweet_id=parent,
    )


def test_positive_ack_proxy_is_not_silence() -> None:
    customer = _message("1", True, 0, None)
    reply = _message("2", False, 1, "1")
    acknowledgement = _message("3", True, 2, "2")
    result = classify_resolution_proxy(
        _thread((customer, reply, acknowledgement)),
        customer.tweet_id,
        reply.tweet_id,
        load_risk_rules(Path("configs/risk_keywords.yaml")),
    )
    assert result is ResolutionProxy.POSITIVE_ACK


def test_select_brand_tie_breaks_by_brand_id() -> None:
    settings = load_brand_selection_settings(Path("configs/brand_selection.yaml"))
    pilot = PilotMetrics(
        sample_size=5000,
        english_share=0.9,
        non_noise_cluster_count=6,
        non_noise_cluster_coverage=0.8,
        largest_cluster_share=0.2,
    )
    common = dict(
        thread_count=20_000,
        inbound_message_count=50_000,
        direct_reply_pair_count=10_000,
        resolution_proxy_count=20_000,
        positive_ack_count=1,
        terminal_action_proxy_count=19_999,
        multi_turn_rate=0.25,
        median_reply_minutes=1.0,
        p90_reply_minutes=2.0,
        pilot=pilot,
        eligibility_failures=(),
        score=0.95,
    )
    selection = select_brand(
        [
            BrandStatistics(brand_id="ZooHelp", **common),
            BrandStatistics(brand_id="AlphaHelp", **common),
        ],
        settings,
    )
    assert selection.selected_brand_id == "AlphaHelp"


def test_select_brand_refuses_to_relax_criteria() -> None:
    settings = load_brand_selection_settings(Path("configs/brand_selection.yaml"))
    with pytest.raises(NoEligibleBrandError):
        select_brand([], settings)
