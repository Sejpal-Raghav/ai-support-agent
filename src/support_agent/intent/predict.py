"""Intent inference with calibrated confidence for main and baseline classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from support_agent.intent.calibrate import TemperatureScaler, softmax
from support_agent.intent.train import IntentClassifier
from support_agent.taxonomy.cluster import TaxonomySettings, embed_for_taxonomy
from support_agent.text import normalize_customer_text


@dataclass(frozen=True)
class IntentPrediction:
    """Intent label plus full calibrated probability distribution."""

    intent_id: str
    probabilities: dict[str, float]
    confidence: float


def _logits_for_texts(
    texts: list[str], classifier: IntentClassifier, settings: TaxonomySettings
) -> NDArray[np.float64]:
    if classifier.kind == "majority":
        counts = cast(dict[str, object], classifier.estimator)["counts"]
        if not isinstance(counts, dict):
            raise ValueError("majority classifier counts are malformed")
        total = sum(int(value) for value in counts.values())
        probabilities = np.array(
            [float(counts.get(label, 0)) / total for label in classifier.class_labels],
            dtype=np.float64,
        )
        return np.log(np.tile(probabilities + 1e-12, (len(texts), 1)))
    if classifier.kind == "embedding_logistic":
        features = embed_for_taxonomy(texts, settings)
        logits = classifier.estimator.decision_function(features)
    else:
        logits = classifier.estimator.decision_function(texts)
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim == 1:
        array = np.column_stack((-array, array))
    return array


def predict_intents(
    texts: list[str],
    classifier: IntentClassifier,
    calibrator: TemperatureScaler,
    settings: TaxonomySettings,
) -> list[IntentPrediction]:
    """Predict a calibrated intent distribution for a batch of normalized inputs."""
    normalized = [normalize_customer_text(text) for text in texts]
    logits = _logits_for_texts(normalized, classifier, settings)
    probabilities = softmax(logits, calibrator.temperature)
    predictions: list[IntentPrediction] = []
    for row in probabilities:
        index = int(np.argmax(row))
        distribution = {
            label: float(value) for label, value in zip(classifier.class_labels, row, strict=True)
        }
        predictions.append(
            IntentPrediction(
                intent_id=classifier.class_labels[index],
                probabilities=distribution,
                confidence=float(row[index]),
            )
        )
    return predictions


def classifier_logits(
    texts: list[str], classifier: IntentClassifier, settings: TaxonomySettings
) -> NDArray[np.float64]:
    """Expose uncalibrated logits solely for validation-only temperature fitting."""
    normalized = [normalize_customer_text(text) for text in texts]
    return _logits_for_texts(normalized, classifier, settings)
