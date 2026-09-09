from __future__ import annotations

from datetime import UTC, datetime

import pytest

from support_agent.schemas import RawMessage


@pytest.fixture
def inbound_message() -> RawMessage:
    return RawMessage(
        tweet_id="1",
        author_id="customer_1",
        inbound=True,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        text="I need help",
        response_tweet_ids=("2",),
        in_response_to_tweet_id=None,
    )
