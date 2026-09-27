"""Run Phase-1 ingestion, reconstruction, EDA, and deterministic brand selection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import typer
from pydantic import BaseModel

from support_agent.data.brand_selection import (
    compute_brand_statistics,
    load_brand_selection_settings,
    load_risk_rules,
    select_brand,
)
from support_agent.data.ingest import ingest_csv, iter_ingested_messages
from support_agent.data.threads import reconstruct_threads
from support_agent.hashing import file_sha256
from support_agent.provenance import ArtifactRef, build_artifact_manifest

app = typer.Typer(add_completion=False)


def _write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")


@app.command()
def main(
    raw_csv: Path = typer.Option(Path("data/raw/twcs.csv"), exists=True, readable=True),
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    interim_dir: Path = typer.Option(Path("data/interim")),
    brand_config: Path = typer.Option(Path("configs/brand_selection.yaml"), exists=True),
    risk_config: Path = typer.Option(Path("configs/risk_keywords.yaml"), exists=True),
    max_rows: int | None = typer.Option(
        750_000,
        help="Deterministic leading source rows; use 0 only through code, never the CLI.",
    ),
) -> None:
    """Materialize all Phase-1 artifacts and print the selected brand."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    ingested_path = interim_dir / "messages.parquet"
    threads_path = artifacts_dir / "reconstructed_threads.jsonl"
    stats_path = artifacts_dir / "brand_stats.parquet"
    selection_path = artifacts_dir / "brand_selection.json"
    manifest_path = artifacts_dir / "phase1_manifest.json"

    ingest_stats = ingest_csv(raw_csv, ingested_path, max_rows=max_rows)
    threads = reconstruct_threads(list(iter_ingested_messages(ingested_path)))
    _write_jsonl(threads_path, threads)
    settings = load_brand_selection_settings(brand_config)
    rules = load_risk_rules(risk_config)
    stats = compute_brand_statistics(threads, settings, rules)
    pd.DataFrame([item.model_dump(mode="json") for item in stats]).to_parquet(
        stats_path, index=False
    )
    selection = select_brand(stats, settings)
    selection_path.write_text(
        json.dumps(
            {
                "source_csv_sha256": file_sha256(raw_csv),
                "ingest": ingest_stats.model_dump(mode="json"),
                "selection": selection.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = build_artifact_manifest(
        inputs=(ArtifactRef(path=raw_csv.as_posix(), sha256=file_sha256(raw_csv)),),
        outputs=(ingested_path, threads_path, stats_path, selection_path),
        command=("uv", "run", "python", "scripts/select_brand.py"),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Selected brand: {selection.selected_brand_id} ({selection.selected_score:.6f})")


if __name__ == "__main__":
    app()
