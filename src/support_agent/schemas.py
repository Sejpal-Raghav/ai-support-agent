"""Strict data contracts shared by every later pipeline phase."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
CosineScore = Annotated[float, Field(ge=-1.0, le=1.0)]


class StrictModel(BaseModel):
    """Base model that makes schema drift an explicit implementation change."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RawMessage(StrictModel):
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: datetime
    text: str = Field(min_length=1)
    response_tweet_ids: tuple[str, ...]
    in_response_to_tweet_id: str | None

    @model_validator(mode="after")
    def datetime_is_timezone_aware(self) -> RawMessage:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class ReconstructedThread(StrictModel):
    thread_id: str
    brand_id: str
    root_tweet_ids: tuple[str, ...]
    messages: tuple[RawMessage, ...]
    started_at: datetime
    ended_at: datetime
    customer_turn_count: int = Field(ge=1)
    brand_turn_count: int = Field(ge=1)
    is_multi_turn: bool
    has_cycle: bool
    dangling_reference_count: int = Field(ge=0)
    reconstruction_version: str


class IntentDefinition(StrictModel):
    intent_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    description: str
    inclusion_rules: tuple[str, ...]
    exclusion_rules: tuple[str, ...]
    positive_examples: tuple[str, ...]
    counter_examples: tuple[str, ...]
    source_cluster_ids: tuple[int, ...]
    training_support: int = Field(ge=1)


class IntentTaxonomy(StrictModel):
    version: str
    brand_id: str
    created_from_split: Literal["train"]
    intents: tuple[IntentDefinition, ...] = Field(min_length=6, max_length=10)
    other_intent_id: Literal["other"] = "other"
    cluster_review_sha256: str
    content_sha256: str

    @model_validator(mode="after")
    def validates_intent_ids(self) -> IntentTaxonomy:
        ids = tuple(intent.intent_id for intent in self.intents)
        if len(ids) != len(set(ids)):
            raise ValueError("taxonomy intent IDs must be unique")
        if self.other_intent_id not in ids:
            raise ValueError("taxonomy must include the other intent")
        return self


def make_intent_enum(taxonomy: IntentTaxonomy) -> type[Enum]:
    """Construct a runtime enum from the reviewed taxonomy, never source code labels."""
    values = {intent.intent_id.upper(): intent.intent_id for intent in taxonomy.intents}
    return cast(type[Enum], Enum("IntentLabel", values, type=str))


class EscalationReason(str, Enum):
    AUTHENTICATION_REQUIRED = "authentication_required"
    ACCOUNT_ACTION_REQUIRED = "account_action_required"
    MONEY_OR_BILLING_RISK = "money_or_billing_risk"
    PRIVACY_OR_CREDENTIAL_RISK = "privacy_or_credential_risk"
    LEGAL_OR_SAFETY_RISK = "legal_or_safety_risk"
    LIVE_STATUS_REQUIRED = "live_status_required"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNSUPPORTED_INTENT = "unsupported_intent"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REPEATED_UNRESOLVED = "repeated_unresolved"
    AMBIGUOUS_REQUEST = "ambiguous_request"
    DRAFT_VALIDATION_FAILED = "draft_validation_failed"


class ResolutionProxy(str, Enum):
    POSITIVE_ACK = "positive_ack"
    TERMINAL_ACTION_PROXY = "terminal_action_proxy"
    UNRESOLVED_OR_UNKNOWN = "unresolved_or_unknown"


class GoldenExample(StrictModel):
    example_id: str
    thread_id: str
    incoming_tweet_id: str
    brand_id: str
    context_messages: tuple[RawMessage, ...]
    intent_label: str
    intent_ambiguous: bool
    requires_escalation: bool
    escalation_reasons: tuple[EscalationReason, ...]
    evidence_adequate: bool | None
    required_reply_elements: tuple[str, ...]
    forbidden_reply_elements: tuple[str, ...]
    annotation_notes: str
    sampling_stratum: str
    stage_one_probability: Probability
    stage_two_probability: Probability
    inclusion_probability: Probability
    sampling_weight: float = Field(gt=0.0)
    annotation_partition: Literal[
        "validation_model_selection",
        "validation_calibration",
        "validation_policy",
        "golden_test",
    ]
    annotator_id: str
    annotation_guideline_version: str
    annotated_at: datetime

    @model_validator(mode="after")
    def validates_context_and_probabilities(self) -> GoldenExample:
        if (
            not self.context_messages
            or self.context_messages[-1].tweet_id != self.incoming_tweet_id
        ):
            raise ValueError("context_messages must end at incoming_tweet_id")
        expected_probability = self.stage_one_probability * self.stage_two_probability
        if abs(self.inclusion_probability - expected_probability) > 1e-12:
            raise ValueError("inclusion_probability must equal stage one × stage two")
        return self


class ResolutionCase(StrictModel):
    case_id: str
    brand_id: str
    source_thread_id: str
    customer_tweet_id: str
    brand_reply_tweet_id: str
    context_text: str
    customer_text: str
    brand_reply_text: str
    intent_id: str
    resolution_proxy: ResolutionProxy
    safe_for_auto_handle: bool
    filter_reasons: tuple[str, ...]
    source_uri: str
    case_builder_version: str


class RetrievedCase(StrictModel):
    case_id: str
    source_thread_id: str
    customer_tweet_id: str
    brand_reply_tweet_id: str
    customer_text: str
    brand_reply_text: str
    intent_id: str
    similarity: CosineScore
    mmr_score: float
    rank: int = Field(ge=1, le=3)
    source_uri: str
    index_version: str


class RiskSignals(StrictModel):
    intent_max_probability: Probability
    intent_entropy_normalized: Probability
    intent_risk: Probability
    top_retrieval_similarity: CosineScore | None
    retrieval_adequacy_probability: Probability
    retrieval_risk: Probability
    retrieval_gap_risk: Probability
    multi_turn_risk: Probability
    severe_negative_risk: Probability
    hard_scope_categories: tuple[EscalationReason, ...]
    draft_validation_passed: bool
    raw_risk_score: Probability
    calibrated_safe_probability: Probability


class PredictionRecord(StrictModel):
    example_id: str
    system_id: str
    thread_id: str
    incoming_tweet_id: str
    predicted_intent: str
    intent_probabilities: dict[str, Probability]
    calibrated_intent_confidence: Probability
    retrieved_cases: tuple[RetrievedCase, ...]
    draft_reply: str
    draftable: bool
    cited_case_ids: tuple[str, ...]
    unsupported_information_needed: tuple[str, ...]
    draft_validation_failures: tuple[str, ...]
    risk_signals: RiskSignals
    action: Literal["auto_handle", "escalate"]
    decision_reasons: tuple[EscalationReason, ...]
    classifier_version: str
    embedding_model_id: str
    embedding_model_revision: str
    retrieval_index_version: str
    prompt_version: str
    prompt_sha256: str
    generator_model_snapshot: str | None
    policy_version: str
    cache_key: str | None
    created_at: datetime


class JudgeScores(StrictModel):
    relevance: int = Field(ge=1, le=4)
    grounding: int = Field(ge=1, le=4)
    usefulness: int = Field(ge=1, le=4)
    tone: int = Field(ge=1, le=4)
    safety: int = Field(ge=1, le=4)
    critical_violation: bool
    critical_violation_category: str | None
    rationale: str
    accepted: bool
    overall_ordinal: int = Field(ge=1, le=4)


class ExampleEvaluationRecord(StrictModel):
    example_id: str
    thread_id: str
    system_id: str
    sampling_stratum: str
    sampling_weight: float = Field(gt=0.0)
    gold_intent: str
    predicted_intent: str
    intent_correct: bool
    gold_requires_escalation: bool
    predicted_action: Literal["auto_handle", "escalate"]
    judge_scores: JudgeScores | None
    primary_human_scores: JudgeScores | None
    independent_human_scores: JudgeScores | None
    composite_safe_to_auto_handle: bool
    auto_handle_error: bool | None


class MetricEstimate(StrictModel):
    metric_name: str
    system_id: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_method: str
    sample_size: int = Field(ge=0)
    weighted: bool


class EvaluationResult(StrictModel):
    run_id: str
    source_manifest_sha256: str
    taxonomy_version: str
    policy_version: str
    metrics: tuple[MetricEstimate, ...]
    per_class_f1: dict[str, float]
    judge_binary_kappa: float
    judge_quadratic_weighted_kappa: float
    trust_decision: Literal["passes_prototype_trust_gate", "not_ready_for_auto_handle"]
    trust_failures: tuple[str, ...]
    example_results: tuple[ExampleEvaluationRecord, ...]
