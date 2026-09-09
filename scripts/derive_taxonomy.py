"""Build train-only clusters, export review, and finalize taxonomy after review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from numpy.typing import NDArray

from support_agent.config import load_settings
from support_agent.data.deduplicate import audit_cross_split_near_duplicates
from support_agent.data.split import ThreadSplits
from support_agent.schemas import ReconstructedThread
from support_agent.taxonomy.build import (
    build_taxonomy,
    load_cluster_reviews,
    review_file_sha256,
    write_taxonomy,
)
from support_agent.taxonomy.cluster import TaxonomySettings, cluster_messages, export_cluster_review

app = typer.Typer(add_completion=False)


def _load_threads(path: Path) -> dict[str, ReconstructedThread]:
    with path.open("r", encoding="utf-8") as handle:
        threads = [ReconstructedThread.model_validate_json(line) for line in handle if line.strip()]
    return {thread.thread_id: thread for thread in threads}


def _split_threads(
    threads: dict[str, ReconstructedThread], manifest: dict[str, object]
) -> ThreadSplits:
    def resolve(name: str) -> tuple[ReconstructedThread, ...]:
        ids = manifest[f"{name}_thread_ids"]
        if not isinstance(ids, list):
            raise ValueError(f"split manifest {name}_thread_ids must be a list")
        return tuple(threads[str(thread_id)] for thread_id in ids)

    return ThreadSplits(
        train=resolve("train"),
        validation=resolve("validation"),
        test=resolve("test"),
    )


class _MiniLMEmbedder:
    def __init__(self, settings: TaxonomySettings) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            settings.embedding_model_id,
            revision=settings.embedding_revision,
            device="cpu",
        )
        self._batch_size = settings.embedding_batch_size

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray(
            self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            ),
            dtype=np.float32,
        )


@app.command()
def main(
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    config: Path = typer.Option(Path("configs/pipeline.yaml"), exists=True),
    review_csv: Path | None = typer.Option(None),
) -> None:
    """Export review by default; pass --review-csv only after human review is complete."""
    pipeline = load_settings(config)
    manifest = json.loads((artifacts_dir / "split_manifest.json").read_text(encoding="utf-8"))
    all_threads = _load_threads(artifacts_dir / "cohort_threads.jsonl")
    splits = _split_threads(all_threads, manifest)
    taxonomy_settings = TaxonomySettings(
        seed=pipeline.seeds.global_,
        embedding_model_id=pipeline.models.embedding.model_id,
        embedding_revision=pipeline.models.embedding.revision,
        embedding_dimension=pipeline.models.embedding.dimension,
        embedding_batch_size=pipeline.models.embedding.batch_size,
    )
    train_messages = [message for thread in splits.train for message in thread.messages]
    result = cluster_messages(train_messages, artifacts_dir / "embeddings", taxonomy_settings)
    assignments = pd.DataFrame([item.model_dump() for item in result.assignments])
    assignments.to_parquet(artifacts_dir / "taxonomy_clusters.parquet", index=False)
    review_path = Path("data/annotations/cluster_review.csv")
    export_cluster_review(result, review_path, artifacts_dir / "taxonomy_cluster_report.parquet")
    near_duplicate_audit = audit_cross_split_near_duplicates(
        splits, _MiniLMEmbedder(taxonomy_settings), cosine_threshold=0.95
    )
    (artifacts_dir / "near_duplicate_audit.json").write_text(
        near_duplicate_audit.model_dump_json(indent=2), encoding="utf-8"
    )
    if review_csv is None:
        typer.echo(
            f"Cluster review exported to {review_path}; complete it before finalizing taxonomy."
        )
        return
    reviews = load_cluster_reviews(review_csv)
    taxonomy = build_taxonomy(
        result,
        reviews,
        brand_id=str(manifest["brand_id"]),
        cluster_review_sha256=review_file_sha256(review_csv),
    )
    write_taxonomy(taxonomy, Path("configs/taxonomy.yaml"))
    typer.echo(f"Taxonomy finalized: {taxonomy.version}")


if __name__ == "__main__":
    app()
