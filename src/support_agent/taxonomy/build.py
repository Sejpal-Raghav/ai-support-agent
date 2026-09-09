"""Human-reviewed conversion of clusters into a compact runtime taxonomy."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from support_agent.hashing import canonical_sha256, file_sha256
from support_agent.schemas import IntentDefinition, IntentTaxonomy
from support_agent.taxonomy.cluster import ClusterResult


class ClusterReview(BaseModel):
    """Completed human decision for one HDBSCAN cluster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_id: int = Field(ge=0)
    coherent: bool
    candidate_intent_id: str | None
    display_name: str | None
    requested_resolution: str | None
    required_agent_capability: str | None
    inclusion_rule_1: str | None
    inclusion_rule_2: str | None
    exclusion_rule_1: str | None
    exclusion_rule_2: str | None
    same_safe_resolution_yes_no: bool | None
    different_routing_value_yes_no: bool | None
    merge_target_cluster_id: int | None
    review_notes: str

    @model_validator(mode="after")
    def validates_completed_review(self) -> ClusterReview:
        if not self.coherent:
            return self
        required = (
            self.candidate_intent_id,
            self.display_name,
            self.requested_resolution,
            self.inclusion_rule_1,
            self.inclusion_rule_2,
            self.exclusion_rule_1,
            self.exclusion_rule_2,
        )
        if any(value is None or not value.strip() for value in required):
            raise ValueError("coherent cluster review must define taxonomy fields")
        if self.merge_target_cluster_id is not None:
            if self.same_safe_resolution_yes_no is not True:
                raise ValueError("a merge requires same_safe_resolution_yes_no=true")
            if self.different_routing_value_yes_no is not False:
                raise ValueError("a merge requires different_routing_value_yes_no=false")
        return self


def _optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_int(value: str | None) -> int | None:
    normalized = _optional_string(value)
    return int(normalized) if normalized is not None else None


def _optional_bool(value: str | None) -> bool | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered == "yes":
        return True
    if lowered == "no":
        return False
    raise ValueError(f"Expected yes/no value, got {value!r}")


def load_cluster_reviews(path: Path) -> list[ClusterReview]:
    """Load a completed CSV review packet and reject incomplete/coercive values."""
    reviews: list[ClusterReview] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            coherent = _optional_bool(row.get("coherent"))
            if coherent is None:
                raise ValueError(f"Cluster {row.get('cluster_id')} has no coherent decision")
            reviews.append(
                ClusterReview(
                    cluster_id=int(row["cluster_id"]),
                    coherent=coherent,
                    candidate_intent_id=_optional_string(row.get("candidate_intent_id")),
                    display_name=_optional_string(row.get("display_name")),
                    requested_resolution=_optional_string(row.get("requested_resolution")),
                    required_agent_capability=_optional_string(
                        row.get("required_agent_capability")
                    ),
                    inclusion_rule_1=_optional_string(row.get("inclusion_rule_1")),
                    inclusion_rule_2=_optional_string(row.get("inclusion_rule_2")),
                    exclusion_rule_1=_optional_string(row.get("exclusion_rule_1")),
                    exclusion_rule_2=_optional_string(row.get("exclusion_rule_2")),
                    same_safe_resolution_yes_no=_optional_bool(
                        row.get("same_safe_resolution_yes_no")
                    ),
                    different_routing_value_yes_no=_optional_bool(
                        row.get("different_routing_value_yes_no")
                    ),
                    merge_target_cluster_id=_optional_int(row.get("merge_target_cluster_id")),
                    review_notes=row.get("review_notes", "").strip(),
                )
            )
    cluster_ids = [review.cluster_id for review in reviews]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("cluster review CSV contains duplicate cluster IDs")
    return reviews


def _resolve_merge_target(cluster_id: int, reviews: dict[int, ClusterReview]) -> int:
    seen: set[int] = set()
    current = cluster_id
    while reviews[current].merge_target_cluster_id is not None:
        if current in seen:
            raise ValueError("cluster review merge targets form a cycle")
        seen.add(current)
        target = reviews[current].merge_target_cluster_id
        if target not in reviews:
            raise ValueError(f"cluster {current} merges into unknown cluster {target}")
        current = target
    return current


def build_taxonomy(
    cluster_result: ClusterResult,
    reviews: Sequence[ClusterReview],
    brand_id: str,
    cluster_review_sha256: str,
) -> IntentTaxonomy:
    """Apply documented review rules and reject taxonomies outside the fixed bounds."""
    review_map = {review.cluster_id: review for review in reviews}
    observed_ids = {
        assignment.cluster_id
        for assignment in cluster_result.assignments
        if assignment.cluster_id >= 0
    }
    if observed_ids != set(review_map):
        raise ValueError("review IDs must exactly match non-noise cluster IDs")
    members_by_cluster: dict[int, list[str]] = defaultdict(list)
    for assignment in cluster_result.assignments:
        members_by_cluster[assignment.cluster_id].append(assignment.normalized_text)
    grouped_clusters: dict[int, list[int]] = defaultdict(list)
    for cluster_id, review in review_map.items():
        if review.coherent:
            grouped_clusters[_resolve_merge_target(cluster_id, review_map)].append(cluster_id)
    total_assignments = len(cluster_result.assignments)
    definitions: list[IntentDefinition] = []
    for root_id, cluster_ids in sorted(grouped_clusters.items()):
        root = review_map[root_id]
        support = sum(len(members_by_cluster[cluster_id]) for cluster_id in cluster_ids)
        if support / total_assignments < 0.03:
            continue
        if root.candidate_intent_id is None or root.display_name is None:
            raise AssertionError("coherent merge root must be complete")
        definitions.append(
            IntentDefinition(
                intent_id=root.candidate_intent_id,
                display_name=root.display_name,
                description=root.requested_resolution or root.display_name,
                inclusion_rules=(root.inclusion_rule_1 or "", root.inclusion_rule_2 or ""),
                exclusion_rules=(root.exclusion_rule_1 or "", root.exclusion_rule_2 or ""),
                positive_examples=tuple(members_by_cluster[root_id][:3]),
                counter_examples=tuple(
                    text
                    for cluster_id, texts in sorted(members_by_cluster.items())
                    if cluster_id != root_id
                    for text in texts[:1]
                )[:3],
                source_cluster_ids=tuple(sorted(cluster_ids)),
                training_support=support,
            )
        )
    definitions.sort(key=lambda item: (-item.training_support, item.intent_id))
    definitions = definitions[:9]
    other_support = total_assignments - sum(item.training_support for item in definitions)
    definitions.append(
        IntentDefinition(
            intent_id="other",
            display_name="Other / unsupported",
            description="A request that does not fit a reviewed substantive intent.",
            inclusion_rules=(
                "No substantive definition applies.",
                "Cluster noise or rare merged intent.",
            ),
            exclusion_rules=(
                "A reviewed substantive intent applies.",
                "A known safe resolution pattern exists.",
            ),
            positive_examples=tuple(members_by_cluster.get(-1, [])[:3]),
            counter_examples=tuple(
                item.positive_examples[0] for item in definitions if item.positive_examples
            )[:3],
            source_cluster_ids=(-1,),
            training_support=max(other_support, 1),
        )
    )
    ids = [definition.intent_id for definition in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError("review generated duplicate intent IDs")
    if not 6 <= len(definitions) <= 10:
        raise ValueError(
            "taxonomy must contain 6–10 classes including other; review/split another cluster first"
        )
    provisional = {
        "brand_id": brand_id,
        "created_from_split": "train",
        "intents": [definition.model_dump(mode="json") for definition in definitions],
        "other_intent_id": "other",
        "cluster_review_sha256": cluster_review_sha256,
    }
    content_sha256 = canonical_sha256(provisional)
    return IntentTaxonomy(
        version=f"taxonomy-{content_sha256[:12]}",
        brand_id=brand_id,
        created_from_split="train",
        intents=tuple(definitions),
        other_intent_id="other",
        cluster_review_sha256=cluster_review_sha256,
        content_sha256=content_sha256,
    )


def write_taxonomy(taxonomy: IntentTaxonomy, output_path: Path) -> None:
    """Write final taxonomy as stable YAML only after all review constraints pass."""
    import yaml

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(taxonomy.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )


def review_file_sha256(path: Path) -> str:
    """Return the review-file hash recorded in taxonomy provenance."""
    return file_sha256(path)
