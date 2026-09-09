"""Fail-closed deterministic validation for proposed public support drafts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from support_agent.schemas import RetrievedCase
from support_agent.text import extract_numeric_tokens, extract_urls

_ACCOUNT_ACTION = re.compile(
    r"\b(we (?:have |will )?(?:refund|cancel|close|unlock|update|reset)|"
    r"your (?:account|refund|order) (?:is|has been))\b",
    flags=re.IGNORECASE,
)
_PRIVATE_DATA = re.compile(
    r"\b(password|passcode|pin|cvv|card number|social security|send us a dm)\b",
    flags=re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass(frozen=True)
class DraftValidation:
    """Validation outcome surfaced verbatim to the later selective-prediction policy."""

    passed: bool
    failures: tuple[str, ...]
    unsupported_information: tuple[str, ...]


def validate_draft(draft_reply: str, evidence: Sequence[RetrievedCase]) -> DraftValidation:
    """Reject unsupported claims and scope violations before an auto-handling decision exists."""
    failures: list[str] = []
    unsupported: list[str] = []
    if not draft_reply.strip():
        failures.append("empty_draft")
    if len(draft_reply) > 280:
        failures.append("twitter_length_exceeded")
    sentence_count = len(_SENTENCE_END.findall(draft_reply))
    if sentence_count < 1 or sentence_count > 2:
        failures.append("sentence_count_out_of_bounds")
    if _ACCOUNT_ACTION.search(draft_reply):
        failures.append("account_action_claim")
    if _PRIVATE_DATA.search(draft_reply):
        failures.append("private_data_request")
    evidence_text = "\n".join(
        f"{case.customer_text}\n{case.brand_reply_text}" for case in evidence
    )
    evidence_urls = set(extract_urls(evidence_text))
    for url in extract_urls(draft_reply):
        if url not in evidence_urls:
            unsupported.append(f"url:{url}")
    evidence_numbers = set(extract_numeric_tokens(evidence_text))
    for number in extract_numeric_tokens(draft_reply):
        if number not in evidence_numbers:
            unsupported.append(f"number:{number}")
    if unsupported:
        failures.append("unsupported_literal")
    return DraftValidation(
        passed=not failures,
        failures=tuple(sorted(failures)),
        unsupported_information=tuple(sorted(unsupported)),
    )
