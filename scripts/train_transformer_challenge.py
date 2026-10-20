"""Train and select BERTweet/DeBERTa using validation labels only."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import typer
import yaml
from sklearn.metrics import f1_score

from support_agent.evaluation.annotations import (
    load_completed_annotations,
    validate_annotation_consistency,
)
from support_agent.intent.calibrate import fit_temperature_scaler, softmax
from support_agent.intent.selection import (
    ModelCandidate,
    select_confidence_threshold,
    select_model_candidate,
)
from support_agent.intent.train import build_pseudo_labeled_training_set
from support_agent.intent.transformer import (
    TransformerIntentPredictor,
    TransformerTrainingConfig,
    train_transformer,
)
from support_agent.schemas import GoldenExample, IntentTaxonomy, ReconstructedThread
from support_agent.taxonomy.cluster import ClusterAssignment, ClusterResult
from support_agent.text import normalize_customer_text

app = typer.Typer(add_completion=False)
MODEL_IDS = ("vinai/bertweet-base", "microsoft/deberta-v3-base")


def _load_taxonomy(path: Path) -> IntentTaxonomy:
    return IntentTaxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _load_threads(path: Path) -> dict[str, ReconstructedThread]:
    return {
        thread.thread_id: thread
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for thread in [ReconstructedThread.model_validate_json(line)]
    }


def _load_clusters(artifacts_dir: Path) -> ClusterResult:
    frame = pd.read_parquet(artifacts_dir / "taxonomy_clusters.parquet")
    return ClusterResult(
        assignments=tuple(
            ClusterAssignment.model_validate(row) for row in frame.to_dict(orient="records")
        ),
        embedding_path=(artifacts_dir / "embeddings" / "train_messages.npy").as_posix(),
        reduced_embedding_path=(
            artifacts_dir / "embeddings" / "train_messages_reduced.npy"
        ).as_posix(),
        settings_sha256="loaded_from_reviewed_artifact",
    )


def _validated_packet(
    path: Path, taxonomy: IntentTaxonomy, partition: str, count: int
) -> list[GoldenExample]:
    records = load_completed_annotations(path)
    audit = validate_annotation_consistency(records, taxonomy)
    if (
        audit.record_count != count
        or audit.partitions != {partition: count}
        or audit.invalid_intent_ids
        or audit.invalid_escalation_examples
        or audit.duplicate_example_ids
    ):
        raise ValueError(f"invalid completed {partition} packet: {audit}")
    return records


def _texts(records: list[GoldenExample]) -> list[str]:
    return [normalize_customer_text(record.context_messages[-1].text) for record in records]


def _warm_p95(predictor: TransformerIntentPredictor, texts: list[str]) -> float:
    samples = (texts * ((21 // len(texts)) + 1))[:21]
    predictor.predict(samples[0])
    timings: list[float] = []
    for text in samples[1:]:
        started = perf_counter()
        predictor.predict(text)
        timings.append((perf_counter() - started) * 1000)
    return float(np.quantile(timings, 0.95))


@app.command()
def main(
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    annotation_dir: Path = typer.Option(Path("data/annotations")),
    taxonomy_path: Path = typer.Option(Path("configs/taxonomy.yaml"), exists=True),
) -> None:
    """Run the fixed-budget challenger and write a validation-only selection manifest."""
    taxonomy = _load_taxonomy(taxonomy_path)
    split_manifest = json.loads((artifacts_dir / "split_manifest.json").read_text(encoding="utf-8"))
    all_threads = _load_threads(artifacts_dir / "cohort_threads.jsonl")
    train_threads = [all_threads[str(value)] for value in split_manifest["train_thread_ids"]]
    selection = _validated_packet(
        annotation_dir / "validation_model_selection.csv",
        taxonomy,
        "validation_model_selection",
        100,
    )
    calibration = _validated_packet(
        annotation_dir / "validation_calibration.csv",
        taxonomy,
        "validation_calibration",
        100,
    )
    training_records = build_pseudo_labeled_training_set(
        _load_clusters(artifacts_dir), taxonomy, train_threads
    )
    validation_ids = set(split_manifest["validation_thread_ids"])
    if any(record.thread_id not in validation_ids for record in [*selection, *calibration]):
        raise ValueError("selection and calibration packets must belong to validation threads")
    if {record.incoming_tweet_id for record in selection} & {
        record.incoming_tweet_id for record in calibration
    }:
        raise ValueError("selection and calibration packets must contain distinct requests")

    candidates: list[ModelCandidate] = []
    per_intent_scores: dict[str, dict[str, float]] = {}
    transformer_root = artifacts_dir / "models" / "transformers"
    for model_id in MODEL_IDS:
        model_dir = transformer_root / model_id.rsplit("/", 1)[-1]
        train_transformer(
            training_records,
            selection,
            model_dir,
            TransformerTrainingConfig(model_id=model_id),
        )
        predictor = TransformerIntentPredictor(model_dir)
        predictions = [predictor.predict(text).intent_id for text in _texts(selection)]
        macro_f1 = float(
            f1_score(
                [record.intent_label for record in selection],
                predictions,
                labels=[intent.intent_id for intent in taxonomy.intents],
                average="macro",
                zero_division=0,
            )
        )
        per_intent_scores[model_id] = dict(
            zip(
                [intent.intent_id for intent in taxonomy.intents],
                f1_score(
                    [record.intent_label for record in selection],
                    predictions,
                    labels=[intent.intent_id for intent in taxonomy.intents],
                    average=None,
                    zero_division=0,
                ).tolist(),
                strict=True,
            )
        )
        candidates.append(
            ModelCandidate(
                model_id=model_id,
                artifact_dir=model_dir.as_posix(),
                validation_macro_f1=macro_f1,
                warm_p95_latency_ms=_warm_p95(predictor, _texts(selection)),
            )
        )

    winner = select_model_candidate(candidates)
    winner_path = Path(winner.artifact_dir)
    predictor = TransformerIntentPredictor(winner_path)
    calibration_logits = predictor.predict_logits(_texts(calibration))
    label_to_index = {label: index for index, label in enumerate(predictor.labels)}
    gold_indices = np.asarray(
        [label_to_index[record.intent_label] for record in calibration], dtype=np.int64
    )
    scaler = fit_temperature_scaler(calibration_logits, gold_indices)
    probabilities = softmax(calibration_logits, scaler.temperature)
    predicted_indices = probabilities.argmax(axis=1)
    threshold = select_confidence_threshold(
        probabilities.max(axis=1).tolist(),
        (predicted_indices == gold_indices).tolist(),
    )
    (winner_path / "intent_temperature.json").write_text(
        json.dumps(scaler.__dict__, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "selection_partition": "validation_model_selection",
        "calibration_partition": "validation_calibration",
        "selection_rule": "macro_f1_then_warm_p95_within_0.005",
        "target_accepted_precision": 0.90,
        "minimum_accepted": 20,
        "confidence_threshold": threshold,
        "selected": winner.__dict__,
        "candidates": [candidate.__dict__ for candidate in candidates],
        "validation_per_intent_f1": per_intent_scores,
        "quality_gate_passed": (
            winner.validation_macro_f1 >= 0.80
            and all(
                score >= 0.65
                for intent, score in per_intent_scores[winner.model_id].items()
                if intent != taxonomy.other_intent_id
            )
        ),
    }
    output_path = transformer_root / "selection.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"Selected {winner.model_id}; manifest written to {output_path}")


if __name__ == "__main__":
    app()
