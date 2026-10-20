from datetime import UTC, datetime, timedelta

from support_agent.data.deduplicate import (
    audit_cross_split_near_duplicates,
    remove_exact_duplicate_threads,
)
from support_agent.data.split import ThreadSplits
from support_agent.schemas import RawMessage, ReconstructedThread


def _thread(thread_id: str, minute: int, text: str) -> ReconstructedThread:
    start = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=minute)
    customer = RawMessage(
        tweet_id=f"{thread_id}-c",
        author_id="customer",
        inbound=True,
        created_at=start,
        text=text,
        response_tweet_ids=(f"{thread_id}-b",),
        in_response_to_tweet_id=None,
    )
    brand = RawMessage(
        tweet_id=f"{thread_id}-b",
        author_id="BrandHelp",
        inbound=False,
        created_at=start + timedelta(minutes=1),
        text="Try restarting.",
        response_tweet_ids=(),
        in_response_to_tweet_id=customer.tweet_id,
    )
    return ReconstructedThread(
        thread_id=thread_id,
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


def test_exact_duplicate_keeps_earliest_thread() -> None:
    newer = _thread("newer", 2, "I NEED help")
    earlier = _thread("earlier", 1, "I need help")
    retained, audit = remove_exact_duplicate_threads([newer, earlier])
    assert [thread.thread_id for thread in retained] == ["earlier"]
    assert audit.exact_groups[0].discarded_thread_ids == ("newer",)


def test_near_duplicate_audit_handles_batch_boundaries_and_excludes_same_split() -> None:
    import numpy as np

    class Encoder:
        def encode(self, texts):
            return np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (len(texts), 1))

    splits = ThreadSplits(
        train=tuple(_thread(str(index), index, "same request") for index in range(130)),
        validation=(_thread("validation", 131, "same request"),),
        test=(_thread("test", 132, "same request"),),
    )
    audit = audit_cross_split_near_duplicates(splits, Encoder())
    assert len(audit.near_duplicate_pairs) == 261
    assert all(pair.left_split != pair.right_split for pair in audit.near_duplicate_pairs)
