from datetime import datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from support_agent.schemas import IntentTaxonomy, RawMessage, make_intent_enum


def test_raw_message_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        RawMessage(
            tweet_id="1",
            author_id="customer_1",
            inbound=True,
            created_at=datetime(2020, 1, 1),
            text="I need help",
            response_tweet_ids=(),
            in_response_to_tweet_id=None,
        )


def test_raw_message_preserves_tweet_id_as_string(inbound_message: RawMessage) -> None:
    assert inbound_message.tweet_id == "1"
    assert isinstance(inbound_message.tweet_id, str)


def test_dynamic_taxonomy_enum_contains_other() -> None:
    raw = yaml.safe_load(Path("data/fixtures/tiny_taxonomy.yaml").read_text(encoding="utf-8"))
    taxonomy = IntentTaxonomy.model_validate(raw)
    intent_enum = make_intent_enum(taxonomy)
    assert intent_enum.OTHER.value == "other"
    assert len(intent_enum) == 6
