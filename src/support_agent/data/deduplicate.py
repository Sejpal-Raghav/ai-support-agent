"""Exact first-contact deduplication and cross-split near-duplicate auditing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from support_agent.data.split import ThreadSplits
from support_agent.schemas import ReconstructedThread
from support_agent.text import normalize_customer_text


class DuplicateGroup(BaseModel):
    """One normalized first-contact text and its retained/discarded threads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized_first_contact: str
    retained_thread_id: str
    discarded_thread_ids: tuple[str, ...]


class NearDuplicatePair(BaseModel):
    """Potential cross-split semantic duplicate for manual audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_split: str
    left_thread_id: str
    right_split: str
    right_thread_id: str
    cosine_similarity: float = Field(ge=-1.0, le=1.0)


class DuplicateAudit(BaseModel):
    """Deduplication and leakage-audit outcomes preserved for reporting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exact_groups: tuple[DuplicateGroup, ...]
    near_duplicate_pairs: tuple[NearDuplicatePair, ...] = ()


class EmbedderProtocol(Protocol):
    """Embedding contract used by the audit without importing a model library."""

    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


def _first_inbound_text(thread: ReconstructedThread) -> str:
    for message in thread.messages:
        if message.inbound:
            return normalize_customer_text(message.text)
    raise ValueError(f"Thread {thread.thread_id} has no inbound message")


def remove_exact_duplicate_threads(
    threads: Sequence[ReconstructedThread],
) -> tuple[list[ReconstructedThread], DuplicateAudit]:
    """Keep only the earliest thread for an exact normalized first-contact duplicate."""
    groups: dict[str, list[ReconstructedThread]] = defaultdict(list)
    for thread in sorted(threads, key=lambda item: (item.started_at, item.thread_id)):
        groups[_first_inbound_text(thread)].append(thread)
    retained: list[ReconstructedThread] = []
    audit_groups: list[DuplicateGroup] = []
    for normalized, group in sorted(groups.items()):
        retained.append(group[0])
        if len(group) > 1:
            audit_groups.append(
                DuplicateGroup(
                    normalized_first_contact=normalized,
                    retained_thread_id=group[0].thread_id,
                    discarded_thread_ids=tuple(item.thread_id for item in group[1:]),
                )
            )
    return retained, DuplicateAudit(exact_groups=tuple(audit_groups))


def _normalize_embeddings(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("near-duplicate audit received a zero embedding")
    return np.asarray(embeddings / norms, dtype=np.float32)


def audit_cross_split_near_duplicates(
    splits: ThreadSplits,
    embedder: EmbedderProtocol,
    cosine_threshold: float = 0.95,
) -> DuplicateAudit:
    """Flag semantic first-contact duplicates across different chronological splits."""
    if not -1.0 <= cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must be within [-1, 1]")
    entries: list[tuple[str, ReconstructedThread, str]] = []
    for split_name, threads in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        entries.extend((split_name, thread, _first_inbound_text(thread)) for thread in threads)
    if len(entries) < 2:
        return DuplicateAudit(exact_groups=())
    embeddings = _normalize_embeddings(embedder.encode([entry[2] for entry in entries]))
    similarities = embeddings @ embeddings.T
    pairs: list[NearDuplicatePair] = []
    for left_index, (left_split, left_thread, _) in enumerate(entries):
        for right_index in range(left_index + 1, len(entries)):
            right_split, right_thread, _ = entries[right_index]
            score = float(similarities[left_index, right_index])
            if left_split != right_split and score >= cosine_threshold:
                pairs.append(
                    NearDuplicatePair(
                        left_split=left_split,
                        left_thread_id=left_thread.thread_id,
                        right_split=right_split,
                        right_thread_id=right_thread.thread_id,
                        cosine_similarity=score,
                    )
                )
    return DuplicateAudit(exact_groups=(), near_duplicate_pairs=tuple(pairs))
