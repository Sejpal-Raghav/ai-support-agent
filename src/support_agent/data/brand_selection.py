"""Predeclared EDA criteria and deterministic selected-brand decision."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from support_agent.hashing import canonical_sha256
from support_agent.schemas import RawMessage, ReconstructedThread, ResolutionProxy
from support_agent.text import detect_language, normalize_customer_text

MINIMUM_DIRECT_REPLY_PAIRS = 8_000
PILOT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PILOT_EMBEDDING_REVISION = "fa97f6e"


class BrandSelectionSettings(BaseModel):
    """All predeclared criteria used to select a brand."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0)
    pilot_messages_per_brand: int = Field(gt=0)
    top_brands_for_pilot: int = Field(gt=0)
    minimum_direct_reply_pairs: int = Field(gt=0)
    minimum_threads: int = Field(gt=0)
    minimum_resolution_proxies: int = Field(gt=0)
    minimum_inbound_messages: int = Field(gt=0)
    minimum_multi_turn_rate: float = Field(ge=0.0, le=1.0)
    minimum_english_share: float = Field(ge=0.0, le=1.0)
    minimum_pilot_clusters: int = Field(gt=0)
    maximum_pilot_clusters: int = Field(gt=0)
    minimum_pilot_cluster_coverage: float = Field(ge=0.0, le=1.0)
    maximum_largest_cluster_share: float = Field(ge=0.0, le=1.0)
    score_weights: dict[str, float]

    @model_validator(mode="after")
    def validates_weights(self) -> BrandSelectionSettings:
        expected = {
            "resolution_proxy",
            "inbound_messages",
            "multi_turn_rate",
            "pilot_cluster_coverage",
            "pilot_cluster_concentration",
        }
        if set(self.score_weights) != expected:
            raise ValueError("score_weights must contain the five predeclared terms")
        if abs(sum(self.score_weights.values()) - 1.0) > 1e-9:
            raise ValueError("score_weights must sum to 1.0")
        return self


class RiskRules(BaseModel):
    """Regex rules required only for Phase-1 resolution proxies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    hard_scope_categories: dict[str, tuple[str, ...]]
    positive_ack: tuple[str, ...]
    negative_followup: tuple[str, ...]
    severe_negative: tuple[str, ...]
    actionable_cue: tuple[str, ...]
    generic_only: tuple[str, ...]


class PilotMetrics(BaseModel):
    """Train-independent coherence indicators computed per candidate brand."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_size: int = Field(ge=0)
    english_share: float = Field(ge=0.0, le=1.0)
    non_noise_cluster_count: int = Field(ge=0)
    non_noise_cluster_coverage: float = Field(ge=0.0, le=1.0)
    largest_cluster_share: float = Field(ge=0.0, le=1.0)


class BrandStatistics(BaseModel):
    """Complete brand-level EDA row used by selection and report artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    brand_id: str
    thread_count: int = Field(ge=0)
    inbound_message_count: int = Field(ge=0)
    direct_reply_pair_count: int = Field(ge=0)
    resolution_proxy_count: int = Field(ge=0)
    positive_ack_count: int = Field(ge=0)
    terminal_action_proxy_count: int = Field(ge=0)
    multi_turn_rate: float = Field(ge=0.0, le=1.0)
    median_reply_minutes: float | None
    p90_reply_minutes: float | None
    pilot: PilotMetrics | None = None
    eligibility_failures: tuple[str, ...] = ()
    score: float | None = None


class BrandSelection(BaseModel):
    """Immutable selected-brand decision with all candidate evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_version: str = "1"
    settings_sha256: str
    selected_brand_id: str
    selected_score: float
    candidates: tuple[BrandStatistics, ...]


class NoEligibleBrandError(RuntimeError):
    """Raised instead of weakening the approved selection criteria."""


def load_brand_selection_settings(path: Path) -> BrandSelectionSettings:
    """Load fixed Phase-1 selection settings from YAML."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("brand selection configuration must be a mapping")
    return BrandSelectionSettings.model_validate(raw)


def load_risk_rules(path: Path) -> RiskRules:
    """Load versioned regex rules from YAML."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("risk rules configuration must be a mapping")
    return RiskRules.model_validate(raw)


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) is not None for pattern in patterns)


def _direct_children(thread: ReconstructedThread) -> dict[str, list[RawMessage]]:
    by_id = {message.tweet_id: message for message in thread.messages}
    children: dict[str, list[RawMessage]] = defaultdict(list)
    for message in thread.messages:
        parent = message.in_response_to_tweet_id
        if parent is not None and parent in by_id:
            children[parent].append(message)
    for parent_id, children_for_parent in children.items():
        children[parent_id] = sorted(
            children_for_parent, key=lambda item: (item.created_at, item.tweet_id)
        )
    return children


def _customer_descendants_after_reply(
    reply: RawMessage, children: dict[str, list[RawMessage]]
) -> list[RawMessage]:
    result: list[RawMessage] = []
    queue: deque[RawMessage] = deque(children.get(reply.tweet_id, []))
    cutoff = reply.created_at + timedelta(hours=48)
    while queue:
        candidate = queue.popleft()
        if candidate.created_at > cutoff:
            continue
        if candidate.inbound:
            result.append(candidate)
        queue.extend(children.get(candidate.tweet_id, []))
    return sorted(result, key=lambda item: (item.created_at, item.tweet_id))


def classify_resolution_proxy(
    thread: ReconstructedThread,
    customer_tweet_id: str,
    reply_tweet_id: str,
    rules: RiskRules,
) -> ResolutionProxy:
    """Classify one direct pair without calling terminal silence confirmed resolution."""
    by_id = {message.tweet_id: message for message in thread.messages}
    customer = by_id[customer_tweet_id]
    reply = by_id[reply_tweet_id]
    if not customer.inbound or reply.inbound:
        raise ValueError("resolution proxies require a customer message and a brand reply")
    if (
        reply.created_at < customer.created_at
        or reply.created_at - customer.created_at > timedelta(hours=72)
    ):
        return ResolutionProxy.UNRESOLVED_OR_UNKNOWN
    children = _direct_children(thread)
    descendants = _customer_descendants_after_reply(reply, children)
    positive = any(_matches_any(item.text, rules.positive_ack) for item in descendants)
    negative = any(_matches_any(item.text, rules.negative_followup) for item in descendants)
    if positive and not negative:
        return ResolutionProxy.POSITIVE_ACK
    has_later_message_within_window = any(
        message.created_at > reply.created_at
        and message.created_at <= reply.created_at + timedelta(hours=48)
        for message in thread.messages
    )
    generic_only = _matches_any(reply.text.strip(), rules.generic_only)
    account_action_claim = _matches_any(
        reply.text,
        (
            r"\b(we|i) (have |'ve |has )?"
            r"(refunded|cancelled|canceled|unlocked|credited|replaced|shipped)",
            r"\b(account|order) (has been|was) (changed|updated|checked|cancelled|canceled)",
        ),
    )
    if (
        not has_later_message_within_window
        and _matches_any(reply.text, rules.actionable_cue)
        and not generic_only
        and not account_action_claim
    ):
        return ResolutionProxy.TERMINAL_ACTION_PROXY
    return ResolutionProxy.UNRESOLVED_OR_UNKNOWN


def _direct_reply_pairs(
    thread: ReconstructedThread,
) -> list[tuple[RawMessage, RawMessage]]:
    children = _direct_children(thread)
    pairs: list[tuple[RawMessage, RawMessage]] = []
    for customer in thread.messages:
        if not customer.inbound:
            continue
        for response in children.get(customer.tweet_id, []):
            if not response.inbound:
                pairs.append((customer, response))
    return pairs


def _sample_for_pilot(messages: Sequence[RawMessage], size: int, seed: int) -> list[RawMessage]:
    return sorted(
        messages,
        key=lambda message: canonical_sha256(f"pilot:{seed}:{message.tweet_id}"),
    )[:size]


def compute_pilot_metrics(
    messages: Sequence[RawMessage], settings: BrandSelectionSettings
) -> PilotMetrics:
    """Embed and cluster the deterministic 5,000-message brand pilot."""
    sample = _sample_for_pilot(messages, settings.pilot_messages_per_brand, settings.seed)
    if not sample:
        return PilotMetrics(
            sample_size=0,
            english_share=0.0,
            non_noise_cluster_count=0,
            non_noise_cluster_coverage=0.0,
            largest_cluster_share=0.0,
        )
    language_results = [detect_language(message.text) for message in sample]
    english_share = sum(
        result.language == "en" and result.probability >= 0.80 for result in language_results
    ) / len(sample)
    english_texts = [
        normalize_customer_text(message.text)
        for message, result in zip(sample, language_results, strict=True)
        if result.language == "en" and result.probability >= 0.80
    ]
    if len(english_texts) < 50:
        return PilotMetrics(
            sample_size=len(sample),
            english_share=english_share,
            non_noise_cluster_count=0,
            non_noise_cluster_coverage=0.0,
            largest_cluster_share=0.0,
        )
    from hdbscan import HDBSCAN  # type: ignore[import-untyped]
    from sentence_transformers import SentenceTransformer
    from umap import UMAP  # type: ignore[import-untyped]

    embedder = SentenceTransformer(
        PILOT_EMBEDDING_MODEL,
        revision=PILOT_EMBEDDING_REVISION,
        device="cpu",
    )
    embeddings = embedder.encode(
        english_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    reduced = UMAP(
        n_neighbors=30,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=settings.seed,
        transform_seed=settings.seed,
    ).fit_transform(embeddings)
    labels = HDBSCAN(
        min_cluster_size=max(50, int(np.ceil(0.01 * len(english_texts)))),
        min_samples=15,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        prediction_data=True,
    ).fit_predict(reduced)
    non_noise = labels[labels >= 0]
    if len(non_noise) == 0:
        return PilotMetrics(
            sample_size=len(sample),
            english_share=english_share,
            non_noise_cluster_count=0,
            non_noise_cluster_coverage=0.0,
            largest_cluster_share=0.0,
        )
    _, counts = np.unique(non_noise, return_counts=True)
    return PilotMetrics(
        sample_size=len(sample),
        english_share=english_share,
        non_noise_cluster_count=int(len(counts)),
        non_noise_cluster_coverage=float(len(non_noise) / len(english_texts)),
        largest_cluster_share=float(max(counts) / len(english_texts)),
    )


def _base_statistics(
    brand_id: str, threads: Sequence[ReconstructedThread], rules: RiskRules
) -> BrandStatistics:
    pairs = [pair for thread in threads for pair in _direct_reply_pairs(thread)]
    proxies: list[ResolutionProxy] = []
    reply_minutes: list[float] = []
    for thread in threads:
        for customer, reply in _direct_reply_pairs(thread):
            proxies.append(
                classify_resolution_proxy(thread, customer.tweet_id, reply.tweet_id, rules)
            )
            if reply.created_at >= customer.created_at:
                reply_minutes.append((reply.created_at - customer.created_at).total_seconds() / 60)
    positive_ack_count = sum(proxy is ResolutionProxy.POSITIVE_ACK for proxy in proxies)
    terminal_count = sum(proxy is ResolutionProxy.TERMINAL_ACTION_PROXY for proxy in proxies)
    return BrandStatistics(
        brand_id=brand_id,
        thread_count=len(threads),
        inbound_message_count=sum(
            message.inbound for thread in threads for message in thread.messages
        ),
        direct_reply_pair_count=len(pairs),
        resolution_proxy_count=positive_ack_count + terminal_count,
        positive_ack_count=positive_ack_count,
        terminal_action_proxy_count=terminal_count,
        multi_turn_rate=sum(thread.is_multi_turn for thread in threads) / len(threads),
        median_reply_minutes=float(np.median(reply_minutes)) if reply_minutes else None,
        p90_reply_minutes=float(np.quantile(reply_minutes, 0.90)) if reply_minutes else None,
    )


def _eligibility_failures(
    stats: BrandStatistics, settings: BrandSelectionSettings
) -> tuple[str, ...]:
    pilot = stats.pilot
    failures: list[str] = []
    if stats.direct_reply_pair_count < settings.minimum_direct_reply_pairs:
        failures.append("minimum_direct_reply_pairs")
    if stats.thread_count < settings.minimum_threads:
        failures.append("minimum_threads")
    if stats.resolution_proxy_count < settings.minimum_resolution_proxies:
        failures.append("minimum_resolution_proxies")
    if stats.inbound_message_count < settings.minimum_inbound_messages:
        failures.append("minimum_inbound_messages")
    if stats.multi_turn_rate < settings.minimum_multi_turn_rate:
        failures.append("minimum_multi_turn_rate")
    if pilot is None:
        failures.append("pilot_not_run")
    else:
        if pilot.english_share < settings.minimum_english_share:
            failures.append("minimum_english_share")
        if not (
            settings.minimum_pilot_clusters
            <= pilot.non_noise_cluster_count
            <= settings.maximum_pilot_clusters
        ):
            failures.append("pilot_cluster_count")
        if pilot.non_noise_cluster_coverage < settings.minimum_pilot_cluster_coverage:
            failures.append("minimum_pilot_cluster_coverage")
        if pilot.largest_cluster_share > settings.maximum_largest_cluster_share:
            failures.append("maximum_largest_cluster_share")
    return tuple(failures)


def _score(stats: BrandStatistics, settings: BrandSelectionSettings) -> float:
    if stats.pilot is None:
        raise ValueError("cannot score a brand before pilot metrics are available")
    weights = settings.score_weights
    return (
        weights["resolution_proxy"] * min(stats.resolution_proxy_count / 20_000, 1.0)
        + weights["inbound_messages"] * min(stats.inbound_message_count / 50_000, 1.0)
        + weights["multi_turn_rate"] * min(stats.multi_turn_rate / 0.25, 1.0)
        + weights["pilot_cluster_coverage"] * stats.pilot.non_noise_cluster_coverage
        + weights["pilot_cluster_concentration"] * (1.0 - stats.pilot.largest_cluster_share)
    )


def compute_brand_statistics(
    threads: Sequence[ReconstructedThread], settings: BrandSelectionSettings, rules: RiskRules
) -> list[BrandStatistics]:
    """Compute all EDA metrics and pilots before any selected-brand choice."""
    by_brand: dict[str, list[ReconstructedThread]] = defaultdict(list)
    for thread in threads:
        by_brand[thread.brand_id].append(thread)
    base = {
        brand_id: _base_statistics(brand_id, brand_threads, rules)
        for brand_id, brand_threads in by_brand.items()
    }
    pilot_candidates = sorted(
        (
            stats
            for stats in base.values()
            if stats.direct_reply_pair_count >= settings.minimum_direct_reply_pairs
        ),
        key=lambda item: (-item.direct_reply_pair_count, item.brand_id),
    )[: settings.top_brands_for_pilot]
    pilot_by_brand: dict[str, PilotMetrics] = {}
    for stats in pilot_candidates:
        inbound = [
            message
            for thread in by_brand[stats.brand_id]
            for message in thread.messages
            if message.inbound
        ]
        pilot_by_brand[stats.brand_id] = compute_pilot_metrics(inbound, settings)
    results: list[BrandStatistics] = []
    for brand_id, stats in base.items():
        provisional = stats.model_copy(update={"pilot": pilot_by_brand.get(brand_id)})
        failures = _eligibility_failures(provisional, settings)
        score = _score(provisional, settings) if not failures else None
        results.append(
            provisional.model_copy(update={"eligibility_failures": failures, "score": score})
        )
    return sorted(results, key=lambda item: item.brand_id)


def select_brand(
    stats: Sequence[BrandStatistics], settings: BrandSelectionSettings
) -> BrandSelection:
    """Select the highest predeclared score or fail without relaxing criteria."""
    eligible = [stat for stat in stats if not stat.eligibility_failures and stat.score is not None]
    if not eligible:
        raise NoEligibleBrandError("No brand meets all predeclared Phase-1 criteria")
    winner = sorted(
        eligible,
        key=lambda item: (-(item.score if item.score is not None else float("inf")), item.brand_id),
    )[0]
    if winner.score is None:
        raise AssertionError("eligible winner must have a score")
    settings_hash = canonical_sha256(settings.model_dump(mode="json"))
    return BrandSelection(
        settings_sha256=settings_hash,
        selected_brand_id=winner.brand_id,
        selected_score=winner.score,
        candidates=tuple(sorted(stats, key=lambda item: item.brand_id)),
    )
