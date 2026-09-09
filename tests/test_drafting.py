from __future__ import annotations

from pathlib import Path

from support_agent.drafting.drafter import generate_grounded_draft
from support_agent.drafting.validate import validate_draft
from support_agent.schemas import RetrievedCase


def _evidence() -> RetrievedCase:
    return RetrievedCase(
        case_id="case_1",
        source_thread_id="historical-thread",
        customer_tweet_id="customer-1",
        brand_reply_tweet_id="brand-1",
        customer_text="app will not start",
        brand_reply_text="Try restarting the app at https://example.test/help.",
        intent_id="app_issue",
        similarity=0.92,
        mmr_score=0.92,
        rank=1,
        source_uri="tweet://customer-1",
        index_version="retrieval-fixture",
    )


class _FakeGenerator:
    def create(self, *, instructions: str, input_text: str) -> str:
        assert "historical_evidence" in input_text
        assert "case_1" in input_text
        assert "Return JSON only" in instructions
        return '{"draft_reply":"Sorry you\'re having trouble. Try restarting the app at https://example.test/help.","cited_case_ids":["case_1"]}'


def test_drafter_restricts_citations_and_uses_cache(tmp_path: Path) -> None:
    first = generate_grounded_draft(
        "BrandHelp",
        "CUSTOMER: app is broken",
        "app_issue",
        [_evidence()],
        "fixture-model",
        tmp_path,
        _FakeGenerator(),
    )
    second = generate_grounded_draft(
        "BrandHelp",
        "CUSTOMER: app is broken",
        "app_issue",
        [_evidence()],
        "fixture-model",
        tmp_path,
    )
    assert not first.from_cache
    assert second.from_cache
    assert second.cited_case_ids == ("case_1",)


def test_draft_validation_rejects_unsupported_literals_and_account_claims() -> None:
    validation = validate_draft(
        "We will refund 99 today at https://invented.example/test.", [_evidence()]
    )
    assert not validation.passed
    assert "account_action_claim" in validation.failures
    assert "unsupported_literal" in validation.failures
