"""Reproducible MiniLM/UMAP/HDBSCAN clustering for train inbound messages."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from support_agent.schemas import RawMessage
from support_agent.text import normalize_customer_text


class TaxonomySettings(BaseModel):
    """Fixed clustering hyperparameters from the approved plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, default=1729)
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "fa97f6e"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    umap_n_neighbors: int = 30
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    hdbscan_min_samples: int = 15


class ClusterAssignment(BaseModel):
    """One train inbound message assigned to a cluster or HDBSCAN noise."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tweet_id: str
    normalized_text: str
    cluster_id: int
    membership_probability: float = Field(ge=0.0, le=1.0)


class ClusterResult(BaseModel):
    """Cluster result metadata; arrays are persisted separately as .npy files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignments: tuple[ClusterAssignment, ...]
    embedding_path: str
    reduced_embedding_path: str
    settings_sha256: str


def train_inbound_messages(threads: Sequence[RawMessage]) -> list[RawMessage]:
    """Return inbound messages only; callers must pass train split messages."""
    return [
        message for message in threads if message.inbound and normalize_customer_text(message.text)
    ]


def embed_for_taxonomy(
    messages: Sequence[str], settings: TaxonomySettings
) -> NDArray[np.float32]:
    """Encode normalized train messages with the fixed local semantic encoder."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(
        settings.embedding_model_id,
        revision=settings.embedding_revision,
        device="cpu",
    )
    embeddings = embedder.encode(
        list(messages),
        batch_size=settings.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if embeddings.shape != (len(messages), settings.embedding_dimension):
        raise ValueError("taxonomy embedding output has an unexpected shape")
    return np.asarray(embeddings, dtype=np.float32)


def cluster_embeddings(
    embeddings: NDArray[np.float32], settings: TaxonomySettings
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float32]]:
    """Return HDBSCAN labels/probabilities and the persisted UMAP representation."""
    if embeddings.ndim != 2 or len(embeddings) < 50:
        raise ValueError("taxonomy clustering requires at least 50 embeddings")
    from hdbscan import HDBSCAN  # type: ignore[import-untyped]
    from umap import UMAP  # type: ignore[import-untyped]

    reduced = UMAP(
        n_neighbors=settings.umap_n_neighbors,
        n_components=settings.umap_n_components,
        min_dist=settings.umap_min_dist,
        metric="cosine",
        random_state=settings.seed,
        transform_seed=settings.seed,
    ).fit_transform(embeddings)
    clusterer = HDBSCAN(
        min_cluster_size=max(50, math.ceil(0.01 * len(embeddings))),
        min_samples=settings.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        prediction_data=True,
    ).fit(reduced)
    return np.asarray(clusterer.labels_, dtype=np.int64), np.asarray(
        clusterer.probabilities_, dtype=np.float64
    ), np.asarray(reduced, dtype=np.float32)


def cluster_messages(
    messages: Sequence[RawMessage], output_dir: Path, settings: TaxonomySettings
) -> ClusterResult:
    """Embed and cluster train-only messages, materializing deterministic array artifacts."""
    inbound = train_inbound_messages(messages)
    if len(inbound) < 50:
        raise ValueError("need at least 50 train inbound messages for taxonomy clustering")
    normalized = [normalize_customer_text(message.text) for message in inbound]
    embeddings = embed_for_taxonomy(normalized, settings)
    labels, probabilities, reduced = cluster_embeddings(embeddings, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "train_messages.npy"
    reduced_path = output_dir / "train_messages_reduced.npy"
    np.save(embedding_path, embeddings)
    np.save(reduced_path, reduced)
    assignments = tuple(
        ClusterAssignment(
            tweet_id=message.tweet_id,
            normalized_text=text,
            cluster_id=int(label),
            membership_probability=float(probability),
        )
        for message, text, label, probability in zip(
            inbound, normalized, labels, probabilities, strict=True
        )
    )
    from support_agent.hashing import canonical_sha256

    return ClusterResult(
        assignments=assignments,
        embedding_path=embedding_path.as_posix(),
        reduced_embedding_path=reduced_path.as_posix(),
        settings_sha256=canonical_sha256(settings.model_dump(mode="json")),
    )


def _ctfidf_terms(assignments: Sequence[ClusterAssignment], cluster_id: int) -> list[str]:
    clustered = [item for item in assignments if item.cluster_id == cluster_id]
    documents_by_cluster: dict[int, Counter[str]] = {}
    for identifier in sorted({item.cluster_id for item in assignments if item.cluster_id >= 0}):
        tokens = Counter(
            token
            for item in assignments
            if item.cluster_id == identifier
            for token in item.normalized_text.split()
            if not token.startswith("__")
        )
        documents_by_cluster[identifier] = tokens
    token_counts = Counter(
        token
        for item in clustered
        for token in item.normalized_text.split()
        if not token.startswith("__")
    )
    total = sum(token_counts.values())
    if total == 0:
        return []
    document_frequency = Counter(
        token for counts in documents_by_cluster.values() for token in counts
    )
    cluster_count = len(documents_by_cluster)
    scores = {
        token: (count / total) * math.log((cluster_count + 1) / (document_frequency[token] + 1))
        for token, count in token_counts.items()
    }
    return [token for token, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:20]]


def export_cluster_review(
    result: ClusterResult, output_path: Path, cluster_report_path: Path
) -> None:
    """Write auditable cluster metrics and a human-completable review CSV."""
    embeddings = np.load(result.embedding_path)
    assignments = list(result.assignments)
    cluster_ids = sorted({item.cluster_id for item in assignments if item.cluster_id >= 0})
    rows: list[dict[str, object]] = []
    for cluster_id in cluster_ids:
        member_indices = [
            index for index, item in enumerate(assignments) if item.cluster_id == cluster_id
        ]
        member_embeddings = embeddings[member_indices]
        centroid = member_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        similarities = member_embeddings @ centroid
        medoid_positions = np.argsort(-similarities, kind="stable")[:25]
        exemplar_indices = [member_indices[int(position)] for position in medoid_positions]
        outside_indices = [
            index for index, item in enumerate(assignments) if item.cluster_id != cluster_id
        ]
        outside_similarities = embeddings[outside_indices] @ centroid
        counter_positions = np.argsort(-outside_similarities, kind="stable")[:10]
        counter_indices = [outside_indices[int(position)] for position in counter_positions]
        rows.append(
            {
                "cluster_id": cluster_id,
                "support": len(member_indices),
                "share": len(member_indices) / len(assignments),
                "mean_membership_probability": float(
                    np.mean([assignments[index].membership_probability for index in member_indices])
                ),
                "top_ctfidf_terms": json.dumps(_ctfidf_terms(assignments, cluster_id)),
                "medoid_examples": json.dumps(
                    [assignments[index].normalized_text for index in exemplar_indices]
                ),
                "counter_examples": json.dumps(
                    [assignments[index].normalized_text for index in counter_indices]
                ),
                "coherent": "",
                "candidate_intent_id": "",
                "display_name": "",
                "requested_resolution": "",
                "required_agent_capability": "",
                "inclusion_rule_1": "",
                "inclusion_rule_2": "",
                "exclusion_rule_1": "",
                "exclusion_rule_2": "",
                "same_safe_resolution_yes_no": "",
                "different_routing_value_yes_no": "",
                "merge_target_cluster_id": "",
                "review_notes": "",
            }
        )
    report = pd.DataFrame(rows)
    cluster_report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(cluster_report_path, index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
