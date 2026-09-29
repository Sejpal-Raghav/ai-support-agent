"""Compact local runtime backed by neural embeddings and real historical replies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from support_agent.api.app import ChatResult

INTENT_SEEDS: dict[str, tuple[str, ...]] = {
    "delivery_tracking": (
        "where is my order",
        "package is late",
        "tracking says delivered but missing",
    ),
    "return_refund": ("return an item", "refund has not arrived", "money back for order"),
    "damaged_or_wrong_item": (
        "item arrived damaged",
        "received the wrong product",
        "product is defective",
    ),
    "account_or_payment": (
        "payment was declined",
        "charged twice",
        "cannot access my account",
    ),
    "order_change_or_cancel": (
        "cancel my order",
        "change delivery address",
        "modify an order",
    ),
    "product_or_stock": (
        "is this product available",
        "item is out of stock",
        "product information",
    ),
}

_HANDLE = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_DIRECT_NAME = re.compile(r",\s+[A-Z][a-z]{1,20}(?=[!,.])")
_AGENT_SIGNATURE = re.compile(r"\s*\^[A-Z]{1,3}\s*$")
_UNSAFE = re.compile(r"\b(password|passcode|pin|cvv|card number|social security)\b", re.I)
_CUSTOMER_PROBLEM = re.compile(
    r"\b(help|issue|problem|wrong|damag|broken|defect|missing|late|delay|lost|"
    r"refund|return|cancel|change|charged|declin|cannot|can't|won't|not|never|"
    r"where|when|how|why|stock|available)\w*\b|\?",
    re.I,
)
_CUSTOMER_PRAISE = re.compile(
    r"\b(thank|appreciat|amazing|great|excellent|love|merry christmas|happy holiday)\w*\b",
    re.I,
)
_REPLY_SUPPORT = re.compile(
    r"\b(sorry|apolog|help|assist|check|look into|investigat|contact|send|share|"
    r"refund|return|replace|order|delivery|carrier|account|payment|cancel|stock)\w*\b",
    re.I,
)
_REPLY_SOCIAL = re.compile(r"\b(merry christmas|happy holiday|glad you|our pleasure)\b", re.I)


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    customer_text: str
    reply_text: str
    intent_id: str


def is_support_exchange(customer_text: str, reply_text: str) -> bool:
    """Keep problem-solving exchanges and reject praise/social chatter."""
    has_problem = bool(_CUSTOMER_PROBLEM.search(customer_text))
    praise_only = bool(_CUSTOMER_PRAISE.search(customer_text)) and not has_problem
    useful_reply = bool(_REPLY_SUPPORT.search(reply_text))
    social_reply = bool(_REPLY_SOCIAL.search(reply_text))
    return has_problem and not praise_only and useful_reply and not social_reply


def sanitize_historical_reply(text: str) -> str:
    """Remove user handles and links before showing a public historical reply."""
    cleaned = _HANDLE.sub("", text)
    cleaned = _URL.sub("", cleaned)
    cleaned = _DIRECT_NAME.sub("", cleaned)
    cleaned = _AGENT_SIGNATURE.sub("", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"\b(here|at):\s+(?=[A-Z])", r"\1. ", cleaned)
    cleaned = re.sub(r"\s+here(?=[.!?])", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*([:,-])\s*$", "", cleaned)
    cleaned = re.sub(r"\s+here$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


class DemoSupportEngine:
    """Classify and retrieve using a pretrained neural sentence encoder."""

    def __init__(
        self,
        cases_path: Path,
        *,
        transformer_dir: Path | None = None,
        intent_threshold: float = 0.62,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._cases = tuple(
            DemoCase(**json.loads(line))
            for line in cases_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not self._cases:
            raise ValueError("the local demo corpus is empty")
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self._case_embeddings = self._model.encode(
            [case.customer_text for case in self._cases], normalize_embeddings=True
        )
        self._intent_ids = tuple(INTENT_SEEDS)
        self._intent_embeddings = self._model.encode(
            [". ".join(INTENT_SEEDS[intent]) for intent in self._intent_ids],
            normalize_embeddings=True,
        )
        self._transformer = None
        self._intent_threshold = intent_threshold
        if transformer_dir is not None:
            from support_agent.intent.transformer import TransformerIntentPredictor

            self._transformer = TransformerIntentPredictor(transformer_dir)

    def __call__(self, text: str) -> ChatResult:
        query = self._model.encode([text], normalize_embeddings=True)[0]
        if self._transformer is not None:
            prediction = self._transformer.predict(text)
            intent_id = prediction.intent_id
            confidence = prediction.confidence
            score_type: Literal["semantic_similarity", "calibrated_probability"] = (
                "calibrated_probability"
            )
            classifier_version = prediction.model_version
        else:
            intent_scores = np.asarray(self._intent_embeddings @ query)
            intent_position = int(np.argmax(intent_scores))
            intent_id = self._intent_ids[intent_position]
            confidence = float(
                np.clip((intent_scores[intent_position] + 1.0) / 2.0, 0.0, 1.0)
            )
            score_type = "semantic_similarity"
            classifier_version = "minilm-semantic-demo-v1"
        candidates = [
            index for index, case in enumerate(self._cases) if case.intent_id == intent_id
        ]
        if confidence < self._intent_threshold or not candidates:
            return ChatResult(
                intent=intent_id,
                confidence=confidence,
                score_type=score_type,
                retrieval_similarity=None,
                response=(
                    "I’m not confident enough to answer that safely. "
                    "A support specialist should review it."
                ),
                action="fallback",
                reasons=("low_intent_confidence",),
                evidence_case_ids=(),
                classifier_version=classifier_version,
                retrieval_index_version="amazonhelp-direct-pairs-v1",
            )
        similarities = np.asarray(self._case_embeddings @ query)
        best = max(candidates, key=lambda index: float(similarities[index]))
        case = self._cases[best]
        reply = sanitize_historical_reply(case.reply_text)
        retrieval_similarity = float(similarities[best])
        if retrieval_similarity < 0.45 or not reply or _UNSAFE.search(reply):
            return ChatResult(
                intent=intent_id,
                confidence=confidence,
                score_type=score_type,
                retrieval_similarity=retrieval_similarity,
                response=(
                    "I found related cases, but not enough safe evidence for an automatic answer."
                ),
                action="fallback",
                reasons=("insufficient_retrieval_evidence",),
                evidence_case_ids=(),
                classifier_version=classifier_version,
                retrieval_index_version="amazonhelp-direct-pairs-v1",
            )
        return ChatResult(
            intent=intent_id,
            confidence=confidence,
            score_type=score_type,
            retrieval_similarity=retrieval_similarity,
            response=reply,
            action="respond",
            reasons=(),
            evidence_case_ids=(case.case_id,),
            classifier_version=classifier_version,
            retrieval_index_version="amazonhelp-direct-pairs-v1",
        )
