"""Compact local runtime backed by neural embeddings and real historical replies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from support_agent.api.app import ChatResult
from support_agent.hashing import verify_jsonl_sha256

DEMO_EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

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
_UNSAFE = re.compile(
    r"\b(password|passcode|pin|cvv|card number|social security|verification code|"
    r"one.time (?:password|code)|otp|2fa)\b",
    re.I,
)
_CONTACT = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b(?:\d[ ()+.-]*){7,}\d\b", re.I)
_PERSONAL_NAME = re.compile(
    r"\b(?:Hi|Hello|Hey)[, ]+[A-Z][a-z]{1,20}\b|"
    r"\b(?:account|order|help|you|that|this)[, ]+[A-Z][a-z]{1,20}(?=[!?,.])"
)
_BILLING_RISK = re.compile(
    r"\b(charged twice|double charged|duplicate charge|unauthori[sz]ed|fraud|chargeback)\b", re.I
)
_CONTEXT_BOUND = re.compile(
    r"\b(?:previous|earlier|above|below|following) (?:link|message|reply)\b|"
    r"\b(?:don't|do not|didn't|did not) want\b",
    re.I,
)
_REPLY_INTENT_CUES = {
    "delivery_tracking": r"\b(track|deliver|carrier|package|parcel|shipment)\w*\b",
    "return_refund": r"\b(return|refund|exchange)\w*\b",
    "damaged_or_wrong_item": r"\b(replac|damag|defect|broken|wrong item)\w*\b",
    "account_or_payment": r"\b(account|payment|charg|billing|sign in|log in)\w*\b",
    "order_change_or_cancel": r"\b(cancel|modif|address|chang)\w*\b",
    "product_or_stock": r"\b(stock|availab|product information|product details)\w*\b",
}
_ACCOUNT_CLAIM = re.compile(
    r"\b(?:we|i)(?:['’]ve| have| already| just| can| will|'ll|’ll)?\s+"
    r"(?:have\s+)?(?:refunded|refund|cancelled|canceled|cancel|credited|unlocked|"
    r"replaced|processed|checked|accessed|issued)\b|"
    r"\b(?:your|the) (?:refund|replacement|order|account) (?:has been|was) "
    r"(?:processed|issued|cancelled|canceled|updated|checked|refunded)\b",
    re.I,
)
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
    source_thread_id: str
    split: str


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
    cleaned = _CONTACT.sub("[private details removed]", cleaned)
    cleaned = _DIRECT_NAME.sub("", cleaned)
    cleaned = _AGENT_SIGNATURE.sub("", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"\b(here|at):\s+(?=[A-Z])", r"\1. ", cleaned)
    cleaned = re.sub(r"\s+here(?=[.!?])", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*([:,-])\s*$", "", cleaned)
    cleaned = re.sub(r"\s+here$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def is_safe_historical_reply(text: str) -> bool:
    """Reject credentials, private contact data, and claims of completed account actions."""
    return not bool(
        _UNSAFE.search(text)
        or _CONTACT.search(text)
        or _ACCOUNT_CLAIM.search(text)
        or _PERSONAL_NAME.search(text)
        or _CONTEXT_BOUND.search(text)
    )


def is_relevant_historical_reply(
    text: str, intent_id: str, customer_text: str | None = None
) -> bool:
    """Require a concrete same-intent cue instead of a generic social/support reply."""
    pattern = _REPLY_INTENT_CUES.get(intent_id)
    if pattern is None or re.search(pattern, text, re.I) is None:
        return False
    if intent_id == "return_refund" and customer_text is not None:
        for operation in ("return", "refund"):
            if re.search(rf"\b{operation}\w*\b", customer_text, re.I) and not re.search(
                rf"\b{operation}\w*\b", text, re.I
            ):
                return False
    return True


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
            for line in cases_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        if not self._cases:
            raise ValueError("the local demo corpus is empty")
        if any(case.split != "train" or not case.source_thread_id for case in self._cases):
            raise ValueError("the demo corpus must contain only traceable train cases")
        manifest = json.loads(
            cases_path.with_suffix(".manifest.json").read_text(encoding="utf-8-sig")
        )
        corpus_hash = verify_jsonl_sha256(cases_path, manifest["corpus_sha256"])
        partitions = manifest["thread_partitions"]
        if any(partitions.get(case.source_thread_id) != "train" for case in self._cases):
            raise ValueError("demo evidence includes a non-train source thread")
        self._index_version = f"amazonhelp-{corpus_hash[:12]}-{DEMO_EMBEDDING_REVISION[:8]}"
        self._model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            revision=DEMO_EMBEDDING_REVISION,
            device="cpu",
            local_files_only=True,
        )
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
        private_request = bool(_UNSAFE.search(text) or _CONTACT.search(text))
        if private_request or _BILLING_RISK.search(text):
            return ChatResult(
                intent="account_or_payment",
                confidence=0.0,
                score_type="semantic_similarity",
                retrieval_similarity=None,
                response=(
                    "Please keep passwords, verification codes, and private details out of chat. "
                    "Use Amazon’s official app or website to contact support securely. "
                    "This demo cannot access or change your account."
                )
                if private_request
                else (
                    "A billing specialist should review duplicate or unauthorized charges. "
                    "Contact support through Amazon’s official app or website. "
                    "This demo cannot check charges or issue refunds."
                ),
                action="fallback",
                reasons=(
                    "privacy_or_credential_risk" if private_request else "money_or_billing_risk",
                ),
                evidence_case_ids=(),
                classifier_version="safety-rules-v1",
                retrieval_index_version=self._index_version,
            )
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
            confidence = float(np.clip((intent_scores[intent_position] + 1.0) / 2.0, 0.0, 1.0))
            score_type = "semantic_similarity"
            classifier_version = f"minilm-semantic-demo-{DEMO_EMBEDDING_REVISION[:12]}"
        candidates = [
            index
            for index, case in enumerate(self._cases)
            if case.intent_id == intent_id
            and is_safe_historical_reply(case.reply_text)
            and is_relevant_historical_reply(case.reply_text, intent_id, text)
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
                retrieval_index_version=self._index_version,
            )
        similarities = np.asarray(self._case_embeddings @ query)
        best = max(candidates, key=lambda index: float(similarities[index]))
        case = self._cases[best]
        reply = sanitize_historical_reply(case.reply_text)
        retrieval_similarity = float(np.clip(similarities[best], -1.0, 1.0))
        if (
            retrieval_similarity < 0.45
            or not reply
            or not is_safe_historical_reply(case.reply_text)
        ):
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
                retrieval_index_version=self._index_version,
            )
        return ChatResult(
            intent=intent_id,
            confidence=confidence,
            score_type=score_type,
            retrieval_similarity=retrieval_similarity,
            response=(
                f"A similar historical case suggested: “{reply}” "
                "This is general guidance; this demo cannot access your account or take action."
            ),
            action="respond",
            reasons=(),
            evidence_case_ids=(case.case_id,),
            classifier_version=classifier_version,
            retrieval_index_version=self._index_version,
        )
