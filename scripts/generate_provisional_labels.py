"""Generate train-centroid provisional intents for annotation stratification only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

from support_agent.evaluation.annotations import build_decision_points
from support_agent.schemas import IntentTaxonomy, ReconstructedThread
from support_agent.taxonomy.cluster import TaxonomySettings, embed_for_taxonomy

app = typer.Typer(add_completion=False)


def _load_threads(path: Path) -> dict[str, ReconstructedThread]:
    return {
        thread.thread_id: thread
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for thread in [ReconstructedThread.model_validate_json(line)]
    }


def _load_taxonomy(path: Path) -> IntentTaxonomy:
    return IntentTaxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@app.command()
def main(
    partition: str = typer.Option(..., help="validation or test"),
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    taxonomy_path: Path = typer.Option(Path("configs/taxonomy.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/provisional_intents.jsonl")),
) -> None:
    """Assign non-train customer turns to nearest reviewed train intent centroid."""
    if partition not in {"validation", "test"}:
        raise typer.BadParameter("partition must be validation or test")
    manifest = json.loads((artifacts_dir / "split_manifest.json").read_text(encoding="utf-8"))
    threads = _load_threads(artifacts_dir / "cohort_threads.jsonl")
    taxonomy = _load_taxonomy(taxonomy_path)
    assignments = pd.read_parquet(artifacts_dir / "taxonomy_clusters.parquet")
    embeddings = np.load(artifacts_dir / "embeddings" / "train_messages.npy")
    if len(assignments) != len(embeddings):
        raise ValueError("taxonomy assignment and train embedding row counts differ")
    cluster_to_intent = {
        cluster_id: intent.intent_id
        for intent in taxonomy.intents
        for cluster_id in intent.source_cluster_ids
    }
    labels = [
        cluster_to_intent.get(int(cluster), taxonomy.other_intent_id)
        for cluster in assignments["cluster_id"]
    ]
    centroids: dict[str, np.ndarray] = {}
    for intent_id in (intent.intent_id for intent in taxonomy.intents):
        indices = [index for index, label in enumerate(labels) if label == intent_id]
        if not indices:
            continue
        centroid = embeddings[indices].mean(axis=0)
        centroids[intent_id] = centroid / np.linalg.norm(centroid)
    if set(centroids) != {intent.intent_id for intent in taxonomy.intents}:
        raise ValueError("each final intent must have at least one train cluster assignment")
    target_threads = [
        threads[str(identifier)] for identifier in manifest[f"{partition}_thread_ids"]
    ]
    raw_points = build_decision_points(target_threads, {})
    texts = [point.context_messages[-1].text for point in raw_points]
    settings = TaxonomySettings()
    vectors = embed_for_taxonomy(texts, settings)
    intent_ids = tuple(sorted(centroids))
    centroid_matrix = np.vstack([centroids[intent_id] for intent_id in intent_ids])
    similarities = vectors @ centroid_matrix.T
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for point, scores in zip(raw_points, similarities, strict=True):
            best = int(np.argmax(scores))
            handle.write(
                json.dumps(
                    {
                        "tweet_id": point.incoming_tweet_id,
                        "intent_id": intent_ids[best],
                        "centroid_similarity": float(scores[best]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    typer.echo(f"Wrote {len(raw_points)} provisional {partition} intent labels to {output}")


if __name__ == "__main__":
    app()
