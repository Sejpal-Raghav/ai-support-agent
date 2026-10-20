from datetime import UTC, datetime, timedelta

from support_agent.data.threads import reconstruct_threads, validate_thread_graph
from support_agent.schemas import RawMessage


def _message(
    tweet_id: str,
    author_id: str,
    inbound: bool,
    minute: int,
    parent: str | None = None,
    children: tuple[str, ...] = (),
) -> RawMessage:
    return RawMessage(
        tweet_id=tweet_id,
        author_id=author_id,
        inbound=inbound,
        created_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=minute),
        text=f"message {tweet_id}",
        response_tweet_ids=children,
        in_response_to_tweet_id=parent,
    )


def test_reconstruction_uses_parent_and_child_references() -> None:
    threads = reconstruct_threads(
        [
            _message("1", "customer", True, 0, children=("2",)),
            _message("2", "BrandHelp", False, 1, parent="1"),
        ]
    )
    assert len(threads) == 1
    assert threads[0].thread_id == "1"
    assert validate_thread_graph(threads[0]).message_count == 2


def test_reconstruction_handles_branch() -> None:
    threads = reconstruct_threads(
        [
            _message("1", "customer", True, 0, children=("2", "3")),
            _message("2", "BrandHelp", False, 1, parent="1"),
            _message("3", "BrandHelp", False, 2, parent="1"),
        ]
    )
    assert len(threads) == 1
    assert threads[0].brand_turn_count == 2


def test_reconstruction_marks_cycle_and_orders_deterministically() -> None:
    threads = reconstruct_threads(
        [
            _message("2", "BrandHelp", False, 1, parent="1"),
            _message("1", "customer", True, 0, parent="2"),
        ]
    )
    assert len(threads) == 1
    assert threads[0].has_cycle
    assert threads[0].root_tweet_ids == ()
    assert [message.tweet_id for message in threads[0].messages] == ["1", "2"]


def test_multi_brand_component_is_excluded() -> None:
    threads = reconstruct_threads(
        [
            _message("1", "customer", True, 0, children=("2", "3")),
            _message("2", "BrandOne", False, 1, parent="1"),
            _message("3", "BrandTwo", False, 2, parent="1"),
        ]
    )
    assert threads == []


def test_distinct_cyclic_components_have_unique_stable_ids() -> None:
    messages = [
        _message("1", "customer", True, 0, parent="2"),
        _message("2", "BrandHelp", False, 1, parent="1"),
        _message("3", "customer", True, 2, parent="4"),
        _message("4", "BrandHelp", False, 3, parent="3"),
    ]
    threads = reconstruct_threads(messages)
    assert len({thread.thread_id for thread in threads}) == 2
    assert [thread.thread_id for thread in threads] == [
        thread.thread_id for thread in reconstruct_threads(list(reversed(messages)))
    ]
