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
