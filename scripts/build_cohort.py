"""Freeze Phase-2 deduplication, cohort membership, and chronological splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from support_agent.config import load_settings
from support_agent.data.deduplicate import remove_exact_duplicate_threads
from support_agent.data.split import assert_no_split_leakage, chronological_split, select_cohort
from support_agent.schemas import ReconstructedThread

app = typer.Typer(add_completion=False)


def _load_threads(path: Path) -> list[ReconstructedThread]:
    with path.open("r", encoding="utf-8") as handle:
        return [ReconstructedThread.model_validate_json(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, threads: list[ReconstructedThread]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for thread in threads:
            handle.write(thread.model_dump_json() + "\n")


@app.command()
def main(
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    config: Path = typer.Option(Path("configs/pipeline.yaml"), exists=True),
) -> None:
    """Create the fixed 12,000-thread selected-brand cohort and split manifest."""
    selection = json.loads((artifacts_dir / "brand_selection.json").read_text(encoding="utf-8"))
    selected_brand = selection["selection"]["selected_brand_id"]
    all_threads = _load_threads(artifacts_dir / "reconstructed_threads.jsonl")
    brand_threads = [thread for thread in all_threads if thread.brand_id == selected_brand]
    retained, duplicate_audit = remove_exact_duplicate_threads(brand_threads)
    settings = load_settings(config)
    cohort = select_cohort(retained, settings.cohort.threads, settings.seeds.global_)
    splits = chronological_split(cohort, settings.cohort.split_ratios)
    leakage = assert_no_split_leakage(splits)

    cohort_jsonl = artifacts_dir / "cohort_threads.jsonl"
    _write_jsonl(cohort_jsonl, cohort)
    pd.DataFrame(
        [
            {
                "thread_id": thread.thread_id,
                "brand_id": thread.brand_id,
                "started_at": thread.started_at,
                "ended_at": thread.ended_at,
                "message_count": len(thread.messages),
                "customer_turn_count": thread.customer_turn_count,
                "brand_turn_count": thread.brand_turn_count,
            }
            for thread in cohort
        ]
    ).to_parquet(artifacts_dir / "cohort_threads.parquet", index=False)
    (artifacts_dir / "duplicate_audit.json").write_text(
        duplicate_audit.model_dump_json(indent=2), encoding="utf-8"
    )
    split_manifest = {
        "split_version": "1",
        "brand_id": selected_brand,
        "train_thread_ids": [thread.thread_id for thread in splits.train],
        "validation_thread_ids": [thread.thread_id for thread in splits.validation],
        "test_thread_ids": [thread.thread_id for thread in splits.test],
        "leakage_audit": leakage.model_dump(mode="json"),
    }
    (artifacts_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    typer.echo(
        f"Frozen cohort for {selected_brand}: "
        f"{len(splits.train)}/{len(splits.validation)}/{len(splits.test)}"
    )


if __name__ == "__main__":
    app()
