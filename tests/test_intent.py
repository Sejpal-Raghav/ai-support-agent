from __future__ import annotations

import numpy as np

from support_agent.intent.calibrate import fit_temperature_scaler, softmax
from support_agent.intent.predict import predict_intents
from support_agent.intent.train import (
    IntentClassifier,
    IntentTrainingRecord,
    train_majority_classifier,
)
from support_agent.schemas import IntentDefinition, IntentTaxonomy
from support_agent.taxonomy.cluster import TaxonomySettings


def _taxonomy() -> IntentTaxonomy:
    intents = tuple(
        IntentDefinition(
            intent_id=f"intent_{index}",
            display_name=f"Intent {index}",
            description="Fixture intent",
            inclusion_rules=("matches", "matches again"),
            exclusion_rules=("does not match", "does not match again"),
            positive_examples=("positive",),
            counter_examples=("counter",),
            source_cluster_ids=(index,),
            training_support=10,
        )
        for index in range(5)
    ) + (
        IntentDefinition(
            intent_id="other",
            display_name="Other",
            description="Other",
            inclusion_rules=("other", "other again"),
            exclusion_rules=("not substantive", "not substantive again"),
            positive_examples=("other",),
            counter_examples=("counter",),
            source_cluster_ids=(-1,),
            training_support=10,
        ),
    )
    return IntentTaxonomy(
        version="taxonomy-fixture",
        brand_id="BrandHelp",
        created_from_split="train",
        intents=intents,
        cluster_review_sha256="a" * 64,
        content_sha256="b" * 64,
    )


def test_temperature_scaler_returns_normalized_probabilities() -> None:
    logits = np.asarray([[4.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 4.0]])
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    scaler = fit_temperature_scaler(logits, labels)
    probabilities = softmax(logits, scaler.temperature)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert 0.05 <= scaler.temperature <= 10.0


def test_majority_prediction_exposes_all_taxonomy_probabilities() -> None:
    taxonomy = _taxonomy()
    records = [
        IntentTrainingRecord("thread-1", "tweet-1", "need help", "intent_1"),
        IntentTrainingRecord("thread-2", "tweet-2", "need help", "intent_1"),
        IntentTrainingRecord("thread-3", "tweet-3", "need help", "intent_2"),
    ]
    classifier: IntentClassifier = train_majority_classifier(records, taxonomy)
    prediction = predict_intents(
        ["new request"],
        classifier,
        fit_temperature_scaler(np.asarray([[1.0, 0.0]]), np.asarray([0], dtype=np.int64)),
        TaxonomySettings(),
    )[0]
    assert prediction.intent_id == "intent_1"
    assert set(prediction.probabilities) == {intent.intent_id for intent in taxonomy.intents}
