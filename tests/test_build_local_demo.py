from support_agent.data.demo import partition_messages
from support_agent.demo_runtime import is_support_exchange


def test_support_exchange_rejects_social_praise() -> None:
    assert not is_support_exchange(
        "Your team is amazing and I appreciate the followup. Merry Christmas!",
        "It was our pleasure! Merry Christmas from our family to yours!",
    )


def test_support_exchange_accepts_problem_solving_reply() -> None:
    assert is_support_exchange(
        "I received the wrong item. Can you help?",
        "We're sorry about that. Please send us the order details so we can investigate.",
    )


def test_demo_partitions_keep_both_reference_directions_together() -> None:
    import pandas as pd

    rows = []
    for index in range(10):
        rows.extend(
            [
                {
                    "tweet_id": f"c{index}",
                    "author_id": "customer",
                    "inbound": True,
                    "created_at": f"2020-01-{index + 1:02d}",
                    "response_tweet_ids": [f"r{index}"],
                    "in_response_to_tweet_id": None,
                },
                {
                    "tweet_id": f"r{index}",
                    "author_id": "AmazonHelp",
                    "inbound": False,
                    "created_at": f"2020-01-{index + 1:02d}",
                    "response_tweet_ids": [],
                    "in_response_to_tweet_id": None,
                },
            ]
        )
    result = partition_messages(pd.DataFrame(rows), "AmazonHelp")
    for index in range(10):
        assert result[f"c{index}"] == result[f"r{index}"]
    assert result["c0"][1] == "train"
    assert result["c7"][1] == "validation"
    assert result["c9"][1] == "test"
