"""Fine-tuned transformer intent classification for the voice chatbot."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from support_agent.hashing import canonical_sha256
from support_agent.intent.train import IntentTrainingRecord
from support_agent.schemas import GoldenExample
from support_agent.text import normalize_customer_text


@dataclass(frozen=True)
class TransformerTrainingConfig:
    model_id: str = "vinai/bertweet-base"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    early_stopping_patience: int = 1
    seed: int = 1729


@dataclass(frozen=True)
class TransformerPrediction:
    intent_id: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


def train_transformer(
    records: Sequence[IntentTrainingRecord],
    validation_records: Sequence[GoldenExample],
    output_dir: Path,
    config: TransformerTrainingConfig,
) -> str:
    """Fine-tune on train-side labels and select epochs only on model-selection labels."""
    if not records:
        raise ValueError("transformer training requires records")
    labels = tuple(sorted({record.intent_id for record in records}))
    if len(labels) < 2:
        raise ValueError("transformer training requires at least two intent labels")
    if not validation_records:
        raise ValueError("transformer training requires validation model-selection labels")
    if any(record.intent_label not in labels for record in validation_records):
        raise ValueError("validation labels include a class absent from transformer training")
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError("Install the ml dependency group to train the transformer.") from error

    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False)
    class TextDataset(Dataset[dict[str, torch.Tensor]]):
        def __init__(self, texts: Sequence[str], target_labels: Sequence[str]) -> None:
            self._labels = target_labels
            self._encoded = tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=config.max_length,
            )

        def __len__(self) -> int:
            return len(self._labels)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            item = {key: torch.tensor(value[index]) for key, value in self._encoded.items()}
            item["labels"] = torch.tensor(label_to_id[self._labels[index]])
            return item

    train_dataset = TextDataset(
        [record.normalized_text for record in records], [record.intent_id for record in records]
    )
    validation_dataset = TextDataset(
        [
            normalize_customer_text(record.context_messages[-1].text)
            for record in validation_records
        ],
        [record.intent_label for record in validation_records],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        seed=config.seed,
        data_seed=config.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
    )

    def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
        logits = np.asarray(evaluation_prediction.predictions)
        target_ids = np.asarray(evaluation_prediction.label_ids)
        predicted_ids = np.argmax(logits, axis=1)
        return {
            "macro_f1": float(
                f1_score(
                    target_ids,
                    predicted_ids,
                    labels=list(range(len(labels))),
                    average="macro",
                    zero_division=0,
                )
            )
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
    )
    trainer.train()
    evaluation = trainer.evaluate()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    version = canonical_sha256(
        {
            "config": asdict(config),
            "labels": labels,
            "records": [(r.tweet_id, r.intent_id) for r in records],
        }
    )
    (output_dir / "voice_transformer_manifest.json").write_text(
        json.dumps(
            {
                "model_version": version,
                "labels": labels,
                "config": asdict(config),
                "validation_macro_f1": float(evaluation["eval_macro_f1"]),
                "validation_partition": "validation_model_selection",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return version


class TransformerIntentPredictor:
    """Loaded inference wrapper that returns calibrated-ready probability distributions."""

    def __init__(self, model_dir: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the ml dependency group to use the transformer.") from error
        manifest = json.loads(
            (model_dir / "voice_transformer_manifest.json").read_text(encoding="utf-8")
        )
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()
        self._version = str(manifest["model_version"])
        calibration_path = model_dir / "intent_temperature.json"
        self._temperature = (
            float(json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"])
            if calibration_path.exists()
            else 1.0
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(
            str(self._model.config.id2label[index])
            for index in range(len(self._model.config.id2label))
        )

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if not texts:
            raise ValueError("prediction requires at least one text")
        tokens = self._tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        with self._torch.no_grad():
            return np.asarray(self._model(**tokens).logits.cpu().numpy(), dtype=np.float64)

    def predict(self, text: str) -> TransformerPrediction:
        logits = self.predict_logits([text])[0]
        shifted = logits / self._temperature - np.max(logits / self._temperature)
        probabilities = (np.exp(shifted) / np.exp(shifted).sum()).tolist()
        labels = self._model.config.id2label
        distribution = {
            str(labels[index]): float(value) for index, value in enumerate(probabilities)
        }
        intent_id = max(distribution, key=lambda label: distribution[label])
        return TransformerPrediction(
            intent_id, distribution[intent_id], distribution, self._version
        )
