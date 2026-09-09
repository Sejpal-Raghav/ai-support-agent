"""Train fair MiniLM and TF-IDF intent classifiers on train-only pseudo-labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from support_agent.hashing import canonical_sha256
from support_agent.schemas import GoldenExample, IntentTaxonomy, ReconstructedThread
from support_agent.taxonomy.cluster import ClusterResult, TaxonomySettings, embed_for_taxonomy
from support_agent.text import normalize_customer_text


@dataclass(frozen=True)
class IntentTrainingRecord:
    """One train-only customer message paired with a cluster-derived intent label."""

    thread_id: str
    tweet_id: str
    normalized_text: str
    intent_id: str


@dataclass(frozen=True)
class IntentClassifier:
    """Serializable classifier wrapper with the exact feature representation recorded."""

    kind: Literal["embedding_logistic", "tfidf_logistic", "majority"]
    estimator: Any
    class_labels: tuple[str, ...]
    selected_c: float | None
    model_version: str


def _latest_customer_text(record: GoldenExample) -> str:
    return normalize_customer_text(record.context_messages[-1].text)


def build_pseudo_labeled_training_set(
    cluster_result: ClusterResult,
    taxonomy: IntentTaxonomy,
    train_threads: Sequence[ReconstructedThread],
) -> list[IntentTrainingRecord]:
    """Map train cluster assignments to reviewed taxonomy labels, otherwise other."""
    cluster_to_intent: dict[int, str] = {}
    for intent in taxonomy.intents:
        if intent.intent_id == taxonomy.other_intent_id:
            continue
        for cluster_id in intent.source_cluster_ids:
            if cluster_id in cluster_to_intent:
                raise ValueError(f"cluster {cluster_id} maps to multiple substantive intents")
            cluster_to_intent[cluster_id] = intent.intent_id
    thread_by_tweet = {
        message.tweet_id: thread.thread_id
        for thread in train_threads
        for message in thread.messages
        if message.inbound
    }
    records: list[IntentTrainingRecord] = []
    for assignment in cluster_result.assignments:
        if assignment.tweet_id not in thread_by_tweet:
            raise ValueError("cluster assignment contains a non-train inbound message")
        records.append(
            IntentTrainingRecord(
                thread_id=thread_by_tweet[assignment.tweet_id],
                tweet_id=assignment.tweet_id,
                normalized_text=assignment.normalized_text,
                intent_id=cluster_to_intent.get(assignment.cluster_id, taxonomy.other_intent_id),
            )
        )
    if not records:
        raise ValueError("pseudo-labeled training set is empty")
    return records


def _selection_labels(records: Sequence[GoldenExample]) -> list[str]:
    if not records:
        raise ValueError("model selection requires its dedicated 100 human labels")
    return [record.intent_label for record in records]


def _select_c(
    fit_for_c: Any,
    selection_features: Any,
    selection_labels: Sequence[str],
    label_order: Sequence[str],
    c_grid: tuple[float, ...],
) -> tuple[Any, float]:
    scores: list[tuple[float, float, Any]] = []
    for c_value in c_grid:
        estimator = fit_for_c(c_value)
        predicted = estimator.predict(selection_features)
        score = float(
            f1_score(
                selection_labels,
                predicted,
                labels=list(label_order),
                average="macro",
                zero_division=0,
            )
        )
        scores.append((score, c_value, estimator))
    best_score = max(score for score, _, _ in scores)
    tied = [item for item in scores if abs(item[0] - best_score) <= 1e-6]
    _, chosen_c, chosen_estimator = sorted(tied, key=lambda item: item[1])[0]
    return chosen_estimator, chosen_c


def _embedding_matrix(
    records: Sequence[IntentTrainingRecord], settings: TaxonomySettings
) -> NDArray[np.float32]:
    return embed_for_taxonomy([record.normalized_text for record in records], settings)


def train_embedding_logistic(
    records: Sequence[IntentTrainingRecord],
    selection_records: Sequence[GoldenExample],
    taxonomy: IntentTaxonomy,
    settings: TaxonomySettings,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> IntentClassifier:
    """Train MiniLM-feature logistic regression and choose C only on its dedicated labels."""
    features = _embedding_matrix(records, settings)
    labels = [record.intent_id for record in records]
    selection_features = embed_for_taxonomy(
        [_latest_customer_text(record) for record in selection_records], settings
    )
    selection_labels = _selection_labels(selection_records)
    label_order = tuple(intent.intent_id for intent in taxonomy.intents)

    def fit_for_c(c_value: float) -> LogisticRegression:
        return LogisticRegression(
            C=c_value,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=1729,
        ).fit(features, labels)

    estimator, selected_c = _select_c(
        fit_for_c, selection_features, selection_labels, label_order, c_grid
    )
    model_version = canonical_sha256(
        {
            "kind": "embedding_logistic",
            "selected_c": selected_c,
            "taxonomy": taxonomy.content_sha256,
            "records": [(record.tweet_id, record.intent_id) for record in records],
        }
    )
    return IntentClassifier(
        kind="embedding_logistic",
        estimator=estimator,
        class_labels=tuple(str(label) for label in estimator.classes_),
        selected_c=selected_c,
        model_version=model_version,
    )


def train_tfidf_logistic(
    records: Sequence[IntentTrainingRecord],
    selection_records: Sequence[GoldenExample],
    taxonomy: IntentTaxonomy,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> IntentClassifier:
    """Train the fair word-plus-character TF-IDF logistic baseline."""
    train_texts = [record.normalized_text for record in records]
    labels = [record.intent_id for record in records]
    selection_texts = [_latest_customer_text(record) for record in selection_records]
    selection_labels = _selection_labels(selection_records)
    label_order = tuple(intent.intent_id for intent in taxonomy.intents)

    def fit_for_c(c_value: float) -> Pipeline:
        feature_union = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=3,
                        max_df=0.98,
                        sublinear_tf=True,
                        max_features=50_000,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        sublinear_tf=True,
                        max_features=50_000,
                    ),
                ),
            ]
        )
        return Pipeline(
            [
                ("features", feature_union),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=1729,
                    ),
                ),
            ]
        ).fit(train_texts, labels)

    estimator, selected_c = _select_c(
        fit_for_c, selection_texts, selection_labels, label_order, c_grid
    )
    classifier = estimator.named_steps["classifier"]
    model_version = canonical_sha256(
        {
            "kind": "tfidf_logistic",
            "selected_c": selected_c,
            "taxonomy": taxonomy.content_sha256,
            "records": [(record.tweet_id, record.intent_id) for record in records],
        }
    )
    return IntentClassifier(
        kind="tfidf_logistic",
        estimator=estimator,
        class_labels=tuple(str(label) for label in classifier.classes_),
        selected_c=selected_c,
        model_version=model_version,
    )


def train_majority_classifier(
    records: Sequence[IntentTrainingRecord], taxonomy: IntentTaxonomy
) -> IntentClassifier:
    """Build the deterministic majority-intent control from train pseudo-label counts."""
    labels = [record.intent_id for record in records]
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    majority = sorted(counts, key=lambda label: (-counts[label], label))[0]
    return IntentClassifier(
        kind="majority",
        estimator={"majority": majority, "counts": counts},
        class_labels=tuple(intent.intent_id for intent in taxonomy.intents),
        selected_c=None,
        model_version=canonical_sha256({"kind": "majority", "counts": counts}),
    )
