from support_agent.taxonomy.build import ClusterReview, build_taxonomy
from support_agent.taxonomy.cluster import ClusterAssignment, ClusterResult


def _review(cluster_id: int) -> ClusterReview:
    return ClusterReview(
        cluster_id=cluster_id,
        coherent=True,
        candidate_intent_id=f"intent_{cluster_id}",
        display_name=f"Intent {cluster_id}",
        requested_resolution=f"Resolve issue {cluster_id}",
        required_agent_capability="public guidance",
        inclusion_rule_1="matches example one",
        inclusion_rule_2="matches example two",
        exclusion_rule_1="not another intent",
        exclusion_rule_2="not ambiguous",
        same_safe_resolution_yes_no=None,
        different_routing_value_yes_no=None,
        merge_target_cluster_id=None,
        review_notes="fixture",
    )


def test_taxonomy_has_six_labels_including_other() -> None:
    assignments = tuple(
        ClusterAssignment(
            tweet_id=f"{cluster_id}-{index}",
            normalized_text=f"message for cluster {cluster_id} {index}",
            cluster_id=cluster_id,
            membership_probability=1.0,
        )
        for cluster_id in range(5)
        for index in range(20)
    )
    result = ClusterResult(
        assignments=assignments,
        embedding_path="unused.npy",
        reduced_embedding_path="unused_reduced.npy",
        settings_sha256="a" * 64,
    )
    taxonomy = build_taxonomy(
        result,
        [_review(cluster_id) for cluster_id in range(5)],
        brand_id="BrandHelp",
        cluster_review_sha256="b" * 64,
    )
    assert len(taxonomy.intents) == 6
    assert taxonomy.other_intent_id == "other"


def test_taxonomy_rejects_fewer_than_six_labels() -> None:
    assignments = tuple(
        ClusterAssignment(
            tweet_id=f"{cluster_id}-{index}",
            normalized_text=f"message {cluster_id} {index}",
            cluster_id=cluster_id,
            membership_probability=1.0,
        )
        for cluster_id in range(4)
        for index in range(25)
    )
    result = ClusterResult(
        assignments=assignments,
        embedding_path="unused.npy",
        reduced_embedding_path="unused_reduced.npy",
        settings_sha256="a" * 64,
    )
    try:
        build_taxonomy(
            result,
            [_review(cluster_id) for cluster_id in range(4)],
            brand_id="BrandHelp",
            cluster_review_sha256="b" * 64,
        )
    except ValueError as error:
        assert "6–10" in str(error)
    else:
        raise AssertionError("expected taxonomy size validation to fail")
