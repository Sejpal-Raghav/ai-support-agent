"""Export blind, stratified human-annotation packets from one held-out split."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from support_agent.evaluation.annotations import (
    build_decision_points,
    select_golden_annotation_set,
    select_validation_annotation_set,
    write_annotation_tasks,
)
from support_agent.schemas import ReconstructedThread

app = typer.Typer(add_completion=False)


def _load_threads(path: Path) -> dict[str, ReconstructedThread]:
    threads: dict[str, ReconstructedThread] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                thread = ReconstructedThread.model_validate_json(line)
                threads[thread.thread_id] = thread
    return threads


def _load_provisional_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tweet_id, intent_id = str(row["tweet_id"]), str(row["intent_id"])
            if tweet_id in labels:
                raise ValueError(f"duplicate provisional label for {tweet_id}")
            labels[tweet_id] = intent_id
    return labels


@app.command()
def main(
    kind: str = typer.Option(..., help="validation or golden"),
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    provisional_labels: Path = typer.Option(
        Path("artifacts/provisional_intents.jsonl"), exists=True
    ),
    output_dir: Path = typer.Option(Path("data/annotations")),
) -> None:
    """Write the exact Phase 3 packets; never mix validation and test inputs."""
    if kind not in {"validation", "golden"}:
        raise typer.BadParameter("kind must be validation or golden")
    manifest = json.loads((artifacts_dir / "split_manifest.json").read_text(encoding="utf-8"))
    threads = _load_threads(artifacts_dir / "cohort_threads.jsonl")
    split = "validation" if kind == "validation" else "test"
    selected = [threads[str(identifier)] for identifier in manifest[f"{split}_thread_ids"]]
    points = build_decision_points(selected, _load_provisional_labels(provisional_labels))
    if kind == "validation":
        samples = select_validation_annotation_set(points)
        names = ("validation_model_selection", "validation_calibration", "validation_policy")
        for name, sample in zip(names, samples, strict=True):
            write_annotation_tasks(sample, output_dir / f"{name}.csv")
    else:
        write_annotation_tasks(select_golden_annotation_set(points), output_dir / "golden_test.csv")
    typer.echo(f"Exported {kind} annotation packet(s) to {output_dir}")


if __name__ == "__main__":
    app()
