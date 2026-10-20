import pytest

from support_agent.demo_runtime import (
    is_relevant_historical_reply,
    is_safe_historical_reply,
    sanitize_historical_reply,
)


def test_sanitize_historical_reply_removes_handles_and_links() -> None:
    assert sanitize_historical_reply("Hi @person see https://example.com please") == "Hi see please"


def test_sanitize_historical_reply_removes_names_and_agent_signatures() -> None:
    original = "We're here to help, Scott! Please check again. ^AL"
    assert sanitize_historical_reply(original) == "We're here to help! Please check again."


def test_sanitize_historical_reply_removes_dangling_link_punctuation() -> None:
    assert sanitize_historical_reply("Please contact us here: https://example.com") == (
        "Please contact us"
    )


def test_sanitize_historical_reply_repairs_removed_inline_link() -> None:
    original = "Create a return here: https://example.com Also, keep your order details private."
    assert sanitize_historical_reply(original) == (
        "Create a return. Also, keep your order details private."
    )


@pytest.mark.parametrize(
    "reply",
    [
        "We have refunded your order.",
        "I've cancelled your subscription.",
        "Your refund has been issued.",
        "Please send your OTP to support@example.com.",
        "Call 1800-123-4567 to share your card number.",
        "Have you tried cancelling it through your account Bonnie?",
        "I understand you don't want your order canceled. Use the previous link.",
    ],
)
def test_unsafe_historical_replies_are_rejected(reply) -> None:
    assert not is_safe_historical_reply(reply)


def test_private_contact_details_are_removed() -> None:
    assert "example.com" not in sanitize_historical_reply("Email jane@example.com for help")
    assert "123-4567" not in sanitize_historical_reply("Call 1800-123-4567 for help")


def test_reply_must_contain_relevant_intent_guidance() -> None:
    assert not is_relevant_historical_reply("Did you get a new delivery date?", "product_or_stock")
    assert is_relevant_historical_reply("Check the product availability page.", "product_or_stock")


def test_refund_only_guidance_cannot_answer_a_return_request() -> None:
    assert not is_relevant_historical_reply(
        "An agent will review a refund if the parcel is lost.",
        "return_refund",
        "I want to return an item for a refund",
    )
