"""Compact local runtime backed by neural embeddings and real historical replies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    customer_text: str
    reply_text: str
    intent_id: str


def sanitize_historical_reply(text: str) -> str:
    """Remove user handles and links before showing a public historical reply."""
    cleaned = _HANDLE.sub("", text)
    cleaned = _URL.sub("", cleaned)
    cleaned = _DIRECT_NAME.sub("", cleaned)
    cleaned = _AGENT_SIGNATURE.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


class DemoSupportEngine:
    """Classify and retrieve using a pretrained neural sentence encoder."""

    def __init__(self, cases_path: Path) -> None:
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

    def __call__(self, text: str) -> ChatResult:
        query = self._model.encode([text], normalize_embeddings=True)[0]
        intent_scores = np.asarray(self._intent_embeddings @ query)
        intent_position = int(np.argmax(intent_scores))
        intent_id = self._intent_ids[intent_position]
        confidence = float(np.clip((intent_scores[intent_position] + 1.0) / 2.0, 0.0, 1.0))
        candidates = [
            index for index, case in enumerate(self._cases) if case.intent_id == intent_id
        ]
        if confidence < 0.62 or not candidates:
            return ChatResult(
                intent=intent_id,
                confidence=confidence,
                response=(
                    "I’m not confident enough to answer that safely. "
                    "A support specialist should review it."
                ),
                action="fallback",
                reasons=("low_intent_confidence",),
                evidence_case_ids=(),
                classifier_version="minilm-semantic-demo-v1",
                retrieval_index_version="amazonhelp-direct-pairs-v1",
            )
        similarities = np.asarray(self._case_embeddings @ query)
        best = max(candidates, key=lambda index: float(similarities[index]))
        case = self._cases[best]
        reply = sanitize_historical_reply(case.reply_text)
        if float(similarities[best]) < 0.30 or not reply or _UNSAFE.search(reply):
            return ChatResult(
                intent=intent_id,
                confidence=confidence,
                response=(
                    "I found related cases, but not enough safe evidence for an automatic answer."
                ),
                action="fallback",
                reasons=("insufficient_retrieval_evidence",),
                evidence_case_ids=(),
                classifier_version="minilm-semantic-demo-v1",
                retrieval_index_version="amazonhelp-direct-pairs-v1",
            )
        return ChatResult(
            intent=intent_id,
            confidence=confidence,
            response=reply,
            action="respond",
            reasons=(),
            evidence_case_ids=(case.case_id,),
            classifier_version="minilm-semantic-demo-v1",
            retrieval_index_version="amazonhelp-direct-pairs-v1",
        )
