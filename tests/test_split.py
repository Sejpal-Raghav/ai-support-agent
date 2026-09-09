from datetime import UTC, datetime, timedelta

import pytest

from support_agent.data.split import assert_no_split_leakage, chronological_split, select_cohort
from support_agent.schemas import RawMessage, ReconstructedThread


def _thread(index: int) -> ReconstructedThread:
    start = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    customer = RawMessage(
        tweet_id=f"{index}-c",
        author_id="customer",
        inbound=True,
        created_at=start,
        text=f"Need help {index}",
        response_tweet_ids=(f"{index}-b",),
        in_response_to_tweet_id=None,
    )
    brand = RawMessage(
        tweet_id=f"{index}-b",
        author_id="BrandHelp",
        inbound=False,
        created_at=start + timedelta(minutes=1),
        text="Try restarting.",
        response_tweet_ids=(),
        in_response_to_tweet_id=customer.tweet_id,
    )
    return ReconstructedThread(
        thread_id=str(index),
        brand_id="BrandHelp",
        root_tweet_ids=(customer.tweet_id,),
        messages=(customer, brand),
        started_at=start,
        ended_at=brand.created_at,
        customer_turn_count=1,
        brand_turn_count=1,
        is_multi_turn=False,
        has_cycle=False,
        dangling_reference_count=0,
        reconstruction_version="1",
    )


def test_select_cohort_is_hash_deterministic() -> None:
    threads = [_thread(index) for index in range(20)]
    first = select_cohort(threads, size=10, seed=1729)
    second = select_cohort(list(reversed(threads)), size=10, seed=1729)
    assert [thread.thread_id for thread in first] == [thread.thread_id for thread in second]


def test_chronological_split_has_no_thread_leakage() -> None:
    splits = chronological_split([_thread(index) for index in range(20)], (0.70, 0.15, 0.15))
    audit = assert_no_split_leakage(splits)
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (14, 3, 3)
    assert not audit.train_validation_overlap
    assert audit.train_end <= audit.validation_start
    assert audit.validation_end <= audit.test_start


def test_select_cohort_rejects_insufficient_threads() -> None:
    with pytest.raises(ValueError, match="Need at least 2"):
        select_cohort([_thread(1)], size=2, seed=1729)
