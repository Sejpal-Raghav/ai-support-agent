from support_agent.demo_runtime import sanitize_historical_reply


def test_sanitize_historical_reply_removes_handles_and_links() -> None:
    assert sanitize_historical_reply("Hi @person see https://example.com please") == "Hi see please"


def test_sanitize_historical_reply_removes_names_and_agent_signatures() -> None:
    original = "We're here to help, Scott! Please check again. ^AL"
    assert sanitize_historical_reply(original) == "We're here to help! Please check again."


def test_sanitize_historical_reply_removes_dangling_link_punctuation() -> None:
    assert sanitize_historical_reply("Please contact us here: https://example.com") == (
        "Please contact us here"
    )


def test_sanitize_historical_reply_repairs_removed_inline_link() -> None:
    original = "Create a return here: https://example.com Also, keep your order details private."
    assert sanitize_historical_reply(original) == (
        "Create a return here. Also, keep your order details private."
    )
