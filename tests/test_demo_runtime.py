from support_agent.demo_runtime import sanitize_historical_reply


def test_sanitize_historical_reply_removes_handles_and_links() -> None:
    assert sanitize_historical_reply("Hi @person see https://example.com please") == "Hi see please"


def test_sanitize_historical_reply_removes_names_and_agent_signatures() -> None:
    original = "We're here to help, Scott! Please check again. ^AL"
    assert sanitize_historical_reply(original) == "We're here to help! Please check again."
