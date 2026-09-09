from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from support_agent.retrieval.cases import build_resolution_cases, safe_index_cases
from support_agent.retrieval.index import RetrievalIndex, mmr_rerank, retrieve_cases
from support_agent.schemas import (
    IntentDefinition,
    IntentTaxonomy,
    RawMessage,
    ReconstructedThread,
    ResolutionCase,
    ResolutionProxy,
)
from support_agent.taxonomy.cluster import ClusterAssignment, ClusterResult, TaxonomySettings


def _taxonomy() -> IntentTaxonomy:
    definitions = tuple(
        IntentDefinition(
            intent_id=f"intent_{index}",
            display_name=f"Intent {index}",
            description="fixture",
            inclusion_rules=("in", "in2"),
            exclusion_rules=("out", "out2"),
            positive_examples=("yes",),
            counter_examples=("no",),
            source_cluster_ids=(index,),
            training_support=1,
        )
        for index in range(5)
    ) + (
        IntentDefinition(
            intent_id="other",
            display_name="Other",
            description="fixture",
            inclusion_rules=("in", "in2"),
            exclusion_rules=("out", "out2"),
            positive_examples=("yes",),
            counter_examples=("no",),
            source_cluster_ids=(-1,),
            training_support=1,
        ),
    )
    return IntentTaxonomy(
        version="taxonomy-fixture",
        brand_id="BrandHelp",
        created_from_split="train",
        intents=definitions,
        cluster_review_sha256="a" * 64,
        content_sha256="b" * 64,
    )


def _message(tweet_id: str, inbound: bool, text: str, minute: int) -> RawMessage:
    return RawMessage(
        tweet_id=tweet_id,
        author_id="customer" if inbound else "BrandHelp",
        inbound=inbound,
        created_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=minute),
        text=text,
        response_tweet_ids=(),
        in_response_to_tweet_id=None,
    )


def test_resolution_case_requires_safe_positive_or_actionable_history() -> None:
    thread = ReconstructedThread(
        thread_id="train-thread",
        brand_id="BrandHelp",
        root_tweet_ids=("1",),
        messages=(
            _message("1", True, "The app will not start", 0),
            _message("2", False, "Try restarting the app.", 1),
            _message("3", True, "Thanks, that worked!", 2),
        ),
        started_at=datetime(2020, 1, 1, tzinfo=UTC),
        ended_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=2),
        customer_turn_count=2,
        brand_turn_count=1,
        is_multi_turn=True,
        has_cycle=False,
        dangling_reference_count=0,
        reconstruction_version="fixture",
    )
    result = ClusterResult(
        assignments=(
            ClusterAssignment(
                tweet_id="1", normalized_text="the app will not start", cluster_id=0,
                membership_probability=1.0,
            ),
        ),
        embedding_path="unused.npy",
        reduced_embedding_path="unused_reduced.npy",
        settings_sha256="fixture",
    )
    cases = build_resolution_cases([thread], _taxonomy(), result)
    assert len(cases) == 1
    assert cases[0].safe_for_auto_handle
    assert cases[0].resolution_proxy == ResolutionProxy.POSITIVE_ACK
    assert safe_index_cases(cases)[0].source_thread_id == "train-thread"


def test_mmr_rerank_is_deterministic_and_limits_evidence_to_three() -> None:
    cases = tuple(
        ResolutionCase(
            case_id=f"case_{index}",
            brand_id="BrandHelp",
            source_thread_id=f"thread_{index}",
            customer_tweet_id=f"customer_{index}",
            brand_reply_tweet_id=f"brand_{index}",
            context_text="context",
            customer_text="request",
            brand_reply_text="reply",
            intent_id="intent_0",
            resolution_proxy=ResolutionProxy.TERMINAL_ACTION_PROXY,
            safe_for_auto_handle=True,
            filter_reasons=(),
            source_uri=f"tweet://{index}",
            case_builder_version="v1",
        )
        for index in range(4)
    )
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    scores = np.asarray([0.99, 0.98, 0.75, 0.74], dtype=np.float32)
    ranked = mmr_rerank([0, 1, 2, 3], scores, embeddings, cases)
    assert len(ranked) == 3
    assert ranked[0][0] == 0
    assert len({position for position, _ in ranked}) == 3


def test_retrieval_preserves_source_case_identifiers(monkeypatch) -> None:
    case = ResolutionCase(
        case_id="case_source",
        brand_id="BrandHelp",
        source_thread_id="historical-thread",
        customer_tweet_id="historical-customer",
        brand_reply_tweet_id="historical-brand",
        context_text="context",
        customer_text="app does not start",
        brand_reply_text="Try restarting the app.",
        intent_id="intent_0",
        resolution_proxy=ResolutionProxy.TERMINAL_ACTION_PROXY,
        safe_for_auto_handle=True,
        filter_reasons=(),
        source_uri="tweet://historical-customer",
        case_builder_version="v1",
    )

    def fake_embed(_: list[str], __: TaxonomySettings) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("support_agent.retrieval.index.embed_for_taxonomy", fake_embed)
    retrieved = retrieve_cases(
        "app is broken",
        "intent_0",
        "live-thread",
        RetrievalIndex(
            cases=(case,),
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            index_version="retrieval-fixture",
        ),
        TaxonomySettings(),
    )
    assert retrieved[0].case_id == "case_source"
    assert retrieved[0].source_thread_id == "historical-thread"
    assert retrieved[0].source_uri == "tweet://historical-customer"
