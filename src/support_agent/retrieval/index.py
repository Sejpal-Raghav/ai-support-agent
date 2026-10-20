"""FAISS inner-product retrieval with deterministic same-intent MMR reranking."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from support_agent.hashing import canonical_sha256
from support_agent.schemas import ResolutionCase, RetrievedCase
from support_agent.taxonomy.cluster import TaxonomySettings, embed_for_taxonomy


@dataclass(frozen=True)
class RetrievalIndex:
    """In-memory immutable index metadata and corresponding safe train cases."""

    cases: tuple[ResolutionCase, ...]
    embeddings: NDArray[np.float32]
    index_version: str


def case_embedding_text(case: ResolutionCase) -> str:
    """Use past context plus request, never the historical reply as a query feature."""
    return f"{case.context_text}\nCUSTOMER_REQUEST: {case.customer_text}"


def build_retrieval_index(
    cases: Sequence[ResolutionCase], settings: TaxonomySettings
) -> RetrievalIndex:
    """Embed safe train cases and create versioned metadata for a FAISS IndexFlatIP artifact."""
    safe_cases = tuple(case for case in cases if case.safe_for_auto_handle)
    if not safe_cases:
        raise ValueError("retrieval index requires at least one safe resolution case")
    embeddings = embed_for_taxonomy([case_embedding_text(case) for case in safe_cases], settings)
    version = canonical_sha256(
        {
            "case_ids": [case.case_id for case in safe_cases],
            "embedding_model_id": settings.embedding_model_id,
            "embedding_revision": settings.embedding_revision,
            "index_type": "IndexFlatIP",
            "retrieval_version": "v1",
        }
    )
    return RetrievalIndex(
        cases=safe_cases,
        embeddings=embeddings,
        index_version=f"retrieval-{version[:12]}",
    )


def save_retrieval_index(index: RetrievalIndex, output_dir: Path) -> None:
    """Persist explicit case metadata, vectors, and an exact FAISS IndexFlatIP artifact."""
    import faiss  # type: ignore[import-untyped]

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss_index = faiss.IndexFlatIP(index.embeddings.shape[1])
    faiss_index.add(np.ascontiguousarray(index.embeddings))
    faiss.write_index(faiss_index, str(output_dir / "resolution_cases.faiss"))
    np.save(output_dir / "resolution_case_embeddings.npy", index.embeddings)
    with (output_dir / "resolution_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in index.cases:
            handle.write(case.model_dump_json() + "\n")
    (output_dir / "retrieval_manifest.json").write_text(
        json.dumps(
            {
                "index_version": index.index_version,
                "index_type": "IndexFlatIP",
                "case_count": len(index.cases),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_retrieval_index(input_dir: Path) -> RetrievalIndex:
    """Load metadata/vectors and verify artifact row alignment before serving retrieval."""
    cases = tuple(
        ResolutionCase.model_validate_json(line)
        for line in (input_dir / "resolution_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    embeddings = np.load(input_dir / "resolution_case_embeddings.npy")
    manifest = json.loads((input_dir / "retrieval_manifest.json").read_text(encoding="utf-8"))
    if len(cases) != len(embeddings) or embeddings.ndim != 2:
        raise ValueError("retrieval case metadata and embedding rows are inconsistent")
    return RetrievalIndex(
        cases=cases,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        index_version=str(manifest["index_version"]),
    )


def mmr_rerank(
    candidate_indices: Sequence[int],
    query_scores: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    cases: Sequence[ResolutionCase],
    top_k: int = 3,
    lambda_mult: float = 0.75,
) -> list[tuple[int, float]]:
    """Select up to three cases using MMR, breaking ties by stable public case ID."""
    selected: list[int] = []
    remaining = list(candidate_indices)
    while remaining and len(selected) < top_k:
        scored: list[tuple[float, str, int]] = []
        for index in remaining:
            diversity = max(
                (float(embeddings[index] @ embeddings[chosen]) for chosen in selected), default=0.0
            )
            mmr_score = lambda_mult * float(query_scores[index]) - (1.0 - lambda_mult) * diversity
            scored.append((mmr_score, cases[index].case_id, index))
        _, _, chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return [
        (
            index,
            lambda_mult * float(query_scores[index])
            - (1.0 - lambda_mult)
            * max(
                (float(embeddings[index] @ embeddings[chosen]) for chosen in selected[:rank]),
                default=0.0,
            ),
        )
        for rank, index in enumerate(selected)
    ]


def retrieve_cases(
    query_text: str,
    predicted_intent: str,
    query_thread_id: str,
    index: RetrievalIndex,
    settings: TaxonomySettings,
) -> tuple[RetrievedCase, ...]:
    """Return max-three same-intent, other-thread evidence records with source IDs preserved."""
    candidates = [
        position
        for position, case in enumerate(index.cases)
        if case.intent_id == predicted_intent and case.source_thread_id != query_thread_id
    ]
    if not candidates:
        return ()
    query = embed_for_taxonomy([query_text], settings)[0]
    scores = np.asarray(index.embeddings @ query, dtype=np.float32)
    ranked = sorted(
        candidates, key=lambda item: (-float(scores[item]), index.cases[item].case_id)
    )[:30]
    selected = mmr_rerank(ranked, scores, index.embeddings, index.cases)
    return tuple(
        RetrievedCase(
            case_id=index.cases[position].case_id,
            source_thread_id=index.cases[position].source_thread_id,
            customer_tweet_id=index.cases[position].customer_tweet_id,
            brand_reply_tweet_id=index.cases[position].brand_reply_tweet_id,
            customer_text=index.cases[position].customer_text,
            brand_reply_text=index.cases[position].brand_reply_text,
            intent_id=index.cases[position].intent_id,
            similarity=float(np.clip(scores[position], -1.0, 1.0)),
            mmr_score=mmr_score,
            rank=rank,
            source_uri=index.cases[position].source_uri,
            index_version=index.index_version,
        )
        for rank, (position, mmr_score) in enumerate(selected, start=1)
    )
