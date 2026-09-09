"""Build the Phase 4 train-only historical-resolution retrieval artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
import yaml

from support_agent.retrieval.cases import build_resolution_cases
from support_agent.retrieval.index import build_retrieval_index, save_retrieval_index
from support_agent.schemas import IntentTaxonomy, ReconstructedThread
from support_agent.taxonomy.cluster import ClusterAssignment, ClusterResult, TaxonomySettings

app = typer.Typer(add_completion=False)


def _load_threads(path: Path) -> dict[str, ReconstructedThread]:
    return {
        thread.thread_id: thread
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for thread in [ReconstructedThread.model_validate_json(line)]
    }


@app.command()
def main(
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    taxonomy_path: Path = typer.Option(Path("configs/taxonomy.yaml"), exists=True),
) -> None:
    """Materialize cases from train IDs alone; validation/test never enter the index."""
    taxonomy = IntentTaxonomy.model_validate(
        yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    )
    manifest = json.loads((artifacts_dir / "split_manifest.json").read_text(encoding="utf-8"))
    threads = _load_threads(artifacts_dir / "cohort_threads.jsonl")
    train_threads = [threads[str(identifier)] for identifier in manifest["train_thread_ids"]]
    assignments = pd.read_parquet(artifacts_dir / "taxonomy_clusters.parquet")
    cluster_result = ClusterResult(
        assignments=tuple(
            ClusterAssignment.model_validate(row) for row in assignments.to_dict(orient="records")
        ),
        embedding_path=(artifacts_dir / "embeddings" / "train_messages.npy").as_posix(),
        reduced_embedding_path=(
            artifacts_dir / "embeddings" / "train_messages_reduced.npy"
        ).as_posix(),
        settings_sha256="loaded_from_phase_2_artifact",
    )
    cases = build_resolution_cases(train_threads, taxonomy, cluster_result)
    index = build_retrieval_index(cases, TaxonomySettings())
    output_dir = artifacts_dir / "retrieval"
    save_retrieval_index(index, output_dir)
    typer.echo(f"Built {len(index.cases)} safe train-only cases: {index.index_version}")


if __name__ == "__main__":
    app()
