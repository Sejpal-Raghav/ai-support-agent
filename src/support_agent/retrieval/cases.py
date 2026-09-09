"""Build auditable, safe historical resolution cases from train threads only."""

from __future__ import annotations

import re
from collections.abc import Sequence

from support_agent.hashing import canonical_sha256
from support_agent.schemas import (
    IntentTaxonomy,
    ReconstructedThread,
    ResolutionCase,
    ResolutionProxy,
)
from support_agent.taxonomy.cluster import ClusterResult
from support_agent.text import normalize_customer_text

_HARD_RISK_PATTERN = re.compile(
    r"\b(password|passcode|pin|cvv|card number|social security|verify identity|"
    r"verification code|2fa|refund|chargeback|charged twice|duplicate charge|"
    r"card charged|lawyer|legal action|sue|police|unsafe|self harm|suicide|threat)\b",
    flags=re.IGNORECASE,
)
_POSITIVE_ACK_PATTERN = re.compile(r"\b(thanks|thank you|worked|fixed|sorted|resolved)\b", re.I)
_NEGATIVE_FOLLOWUP_PATTERN = re.compile(
    r"\b(not working|still|didn't work|did not work|worse|unresolved)\b", re.I
)
_ACTIONABLE_PATTERN = re.compile(
    r"\b(try|check|update|restart|visit|use|follow|enable|disable|clear)\b", re.I
)
_GENERIC_ONLY_PATTERN = re.compile(
    r"^(thanks|thank you|sorry|we're sorry|we are sorry)[!. ]*$", re.I
)


def cluster_intent_map(cluster_result: ClusterResult, taxonomy: IntentTaxonomy) -> dict[str, str]:
    """Map only reviewed train assignments to their final taxonomy intent IDs."""
    cluster_to_intent = {
        cluster_id: intent.intent_id
        for intent in taxonomy.intents
        for cluster_id in intent.source_cluster_ids
    }
    return {
        assignment.tweet_id: cluster_to_intent.get(assignment.cluster_id, taxonomy.other_intent_id)
        for assignment in cluster_result.assignments
    }


def _context_text(thread: ReconstructedThread, incoming_index: int) -> str:
    start = max(0, incoming_index - 2)
    return "\n".join(
        f"{'CUSTOMER' if message.inbound else 'BRAND'}: {normalize_customer_text(message.text)}"
        for message in thread.messages[start : incoming_index + 1]
    )


def _resolution_proxy(
    thread: ReconstructedThread, reply_index: int, next_inbound_index: int | None
) -> ResolutionProxy:
    end = None if next_inbound_index is None else next_inbound_index + 1
    following = thread.messages[reply_index + 1 : end]
    following_text = " ".join(message.text for message in following if message.inbound)
    if _NEGATIVE_FOLLOWUP_PATTERN.search(following_text):
        return ResolutionProxy.UNRESOLVED_OR_UNKNOWN
    if _POSITIVE_ACK_PATTERN.search(following_text):
        return ResolutionProxy.POSITIVE_ACK
    reply = thread.messages[reply_index].text
    if _ACTIONABLE_PATTERN.search(reply):
        return ResolutionProxy.TERMINAL_ACTION_PROXY
    return ResolutionProxy.UNRESOLVED_OR_UNKNOWN


def build_resolution_cases(
    train_threads: Sequence[ReconstructedThread],
    taxonomy: IntentTaxonomy,
    cluster_result: ClusterResult,
) -> list[ResolutionCase]:
    """Extract only safe, resolved customer→brand pairs from the train split.

    A case needs the first brand reply before the next customer request, a non-generic reply,
    no hard risk cue in request/reply, and either a positive customer acknowledgement or an
    actionable reply without a later negative follow-up. Cases labelled ``other`` stay out of
    the index because no reviewed safe intent definition supports automatic handling.
    """
    intent_by_tweet = cluster_intent_map(cluster_result, taxonomy)
    cases: list[ResolutionCase] = []
    train_ids = {thread.thread_id for thread in train_threads}
    if len(train_ids) != len(train_threads):
        raise ValueError("train threads contain duplicate thread IDs")
    for thread in train_threads:
        for incoming_index, incoming in enumerate(thread.messages):
            if not incoming.inbound:
                continue
            next_inbound_index = next(
                (
                    index
                    for index in range(incoming_index + 1, len(thread.messages))
                    if thread.messages[index].inbound
                ),
                None,
            )
            reply_index = next(
                (
                    index
                    for index in range(
                        incoming_index + 1, next_inbound_index or len(thread.messages)
                    )
                    if not thread.messages[index].inbound
                ),
                None,
            )
            if reply_index is None:
                continue
            reply = thread.messages[reply_index]
            intent_id = intent_by_tweet.get(incoming.tweet_id, taxonomy.other_intent_id)
            proxy = _resolution_proxy(thread, reply_index, next_inbound_index)
            reasons: list[str] = []
            if intent_id == taxonomy.other_intent_id:
                reasons.append("unsupported_intent")
            if _HARD_RISK_PATTERN.search(incoming.text) or _HARD_RISK_PATTERN.search(reply.text):
                reasons.append("hard_scope_risk")
            if _GENERIC_ONLY_PATTERN.fullmatch(reply.text.strip()):
                reasons.append("generic_reply")
            if proxy == ResolutionProxy.UNRESOLVED_OR_UNKNOWN:
                reasons.append("unresolved_or_unknown")
            safe = not reasons
            case_id = canonical_sha256(
                {"thread_id": thread.thread_id, "customer_tweet_id": incoming.tweet_id,
                 "brand_reply_tweet_id": reply.tweet_id, "case_builder_version": "v1"}
            )[:20]
            cases.append(
                ResolutionCase(
                    case_id=f"case_{case_id}",
                    brand_id=thread.brand_id,
                    source_thread_id=thread.thread_id,
                    customer_tweet_id=incoming.tweet_id,
                    brand_reply_tweet_id=reply.tweet_id,
                    context_text=_context_text(thread, incoming_index),
                    customer_text=normalize_customer_text(incoming.text),
                    brand_reply_text=reply.text.strip(),
                    intent_id=intent_id,
                    resolution_proxy=proxy,
                    safe_for_auto_handle=safe,
                    filter_reasons=tuple(reasons),
                    source_uri=f"tweet://{incoming.tweet_id}",
                    case_builder_version="v1",
                )
            )
    return cases


def safe_index_cases(cases: Sequence[ResolutionCase]) -> list[ResolutionCase]:
    """Return the only cases eligible for retrieval-grounded automatic drafting."""
    return [case for case in cases if case.safe_for_auto_handle]
