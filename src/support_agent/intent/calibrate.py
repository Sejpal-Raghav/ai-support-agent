"""Validation-only temperature scaling for intent confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class TemperatureScaler:
    """Frozen scalar temperature and its validation-only fit evidence."""

    temperature: float
    uncalibrated_nll: float
    calibrated_nll: float
    applied: bool


def softmax(logits: NDArray[np.float64], temperature: float = 1.0) -> NDArray[np.float64]:
    """Numerically stable temperature-scaled softmax."""
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return np.asarray(
        exponentiated / np.sum(exponentiated, axis=1, keepdims=True), dtype=np.float64
    )


def negative_log_likelihood(
    logits: NDArray[np.float64], label_indices: NDArray[np.int64], temperature: float
) -> float:
    """Return mean multiclass NLL for a candidate temperature."""
    probabilities = softmax(logits, temperature)
    selected = probabilities[np.arange(len(label_indices)), label_indices]
    return float(-np.mean(np.log(selected + 1e-12)))


def fit_temperature_scaler(
    logits: NDArray[np.float64], label_indices: NDArray[np.int64]
) -> TemperatureScaler:
    """Fit T on calibration labels and retain identity when NLL does not improve."""
    if logits.ndim != 2 or len(logits) != len(label_indices):
        raise ValueError("logits and labels must have matching two-dimensional rows")
    uncalibrated = negative_log_likelihood(logits, label_indices, 1.0)
    result = minimize_scalar(
        lambda temperature: negative_log_likelihood(logits, label_indices, float(temperature)),
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    candidate = float(result.x)
    calibrated = negative_log_likelihood(logits, label_indices, candidate)
    applied = calibrated <= uncalibrated - 1e-6
    return TemperatureScaler(
        temperature=candidate if applied else 1.0,
        uncalibrated_nll=uncalibrated,
        calibrated_nll=calibrated if applied else uncalibrated,
        applied=applied,
    )
