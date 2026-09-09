"""Train Phase 3 intent systems strictly from train clusters and validation labels."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import typer
import yaml

from support_agent.evaluation.annotations import (
    load_completed_annotations,
    validate_annotation_consistency,
)
from support_agent.intent.calibrate import fit_temperature_scaler
from support_agent.intent.predict import classifier_logits
from support_agent.intent.train import (
    build_pseudo_labeled_training_set,
    train_embedding_logistic,
    train_majority_classifier,
    train_tfidf_logistic,
)
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


def _load_taxonomy(path: Path) -> IntentTaxonomy:
    return IntentTaxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _load_cluster_result(artifacts_dir: Path) -> ClusterResult:
    assignments = pd.read_parquet(artifacts_dir / "taxonomy_clusters.parquet")
    return ClusterResult(
        assignments=tuple(
            ClusterAssignment.model_validate(row)
            for row in assignments.to_dict(orient="records")
        ),
        embedding_path=(artifacts_dir / "embeddings" / "train_messages.npy").as_posix(),
        reduced_embedding_path=(
            artifacts_dir / "embeddings" / "train_messages_reduced.npy"
        ).as_posix(),
        settings_sha256="loaded_from_phase_2_artifact",
    )


def _validated_records(path: Path, taxonomy: IntentTaxonomy, partition: str, count: int):
    records = load_completed_annotations(path)
    audit = validate_annotation_consistency(records, taxonomy)
    if (
        audit.record_count != count
        or audit.partitions != {partition: count}
        or audit.invalid_intent_ids
        or audit.invalid_escalation_examples
        or audit.duplicate_example_ids
    ):
        raise ValueError(f"invalid completed {partition} annotation packet: {audit}")
    return records


@app.command()
def main(
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    annotation_dir: Path = typer.Option(Path("data/annotations")),
    taxonomy_path: Path = typer.Option(Path("configs/taxonomy.yaml"), exists=True),
) -> None:
    """Fit all intent models and fit the main model temperature on calibration only."""
    taxonomy = _load_taxonomy(taxonomy_path)
    manifest = json.loads((artifacts_dir / "split_manifest.json").read_text(encoding="utf-8"))
    threads = _load_threads(artifacts_dir / "cohort_threads.jsonl")
    train_threads = [threads[str(identifier)] for identifier in manifest["train_thread_ids"]]
    selection = _validated_records(
        annotation_dir / "validation_model_selection.csv",
        taxonomy,
        "validation_model_selection",
        100,
    )
    calibration = _validated_records(
        annotation_dir / "validation_calibration.csv",
        taxonomy,
        "validation_calibration",
        100,
    )
    records = build_pseudo_labeled_training_set(
        _load_cluster_result(artifacts_dir), taxonomy, train_threads
    )
    if len({record.intent_id for record in records}) < 2:
        raise ValueError("intent training requires pseudo-labels for at least two classes")
    settings = TaxonomySettings()
    main_model = train_embedding_logistic(records, selection, taxonomy, settings)
    tfidf_model = train_tfidf_logistic(records, selection, taxonomy)
    majority_model = train_majority_classifier(records, taxonomy)
    label_to_index = {label: index for index, label in enumerate(main_model.class_labels)}
    if any(record.intent_label not in label_to_index for record in calibration):
        raise ValueError("calibration labels include a class absent from the main model")
    logits = classifier_logits(
        [record.context_messages[-1].text for record in calibration], main_model, settings
    )
    import numpy as np

    calibrator = fit_temperature_scaler(
        logits,
        np.asarray(
            [label_to_index[record.intent_label] for record in calibration], dtype=np.int64
        ),
    )
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(main_model, model_dir / "intent_main.joblib")
    joblib.dump(tfidf_model, model_dir / "intent_tfidf_baseline.joblib")
    joblib.dump(majority_model, model_dir / "intent_majority_baseline.joblib")
    (model_dir / "intent_temperature.json").write_text(
        json.dumps(calibrator.__dict__, indent=2, sort_keys=True), encoding="utf-8"
    )
    typer.echo(
        f"Trained main and baselines; main C={main_model.selected_c}, T={calibrator.temperature}"
    )


if __name__ == "__main__":
    app()
