"""Fine-tuned transformer intent classification for the voice chatbot."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from support_agent.hashing import canonical_sha256
from support_agent.intent.train import IntentTrainingRecord


@dataclass(frozen=True)
class TransformerTrainingConfig:
    model_id: str = "vinai/bertweet-base"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    seed: int = 1729


@dataclass(frozen=True)
class TransformerPrediction:
    intent_id: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


def train_transformer(
    records: Sequence[IntentTrainingRecord], output_dir: Path, config: TransformerTrainingConfig
) -> str:
    """Fine-tune a sequence classifier only from train-side reviewed/pseudo labels."""
    if not records:
        raise ValueError("transformer training requires records")
    labels = tuple(sorted({record.intent_id for record in records}))
    if len(labels) < 2:
        raise ValueError("transformer training requires at least two intent labels")
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError("Install the ml dependency group to train the transformer.") from error

    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False)
    encoded = tokenizer(
        [record.normalized_text for record in records],
        padding=True,
        truncation=True,
        max_length=config.max_length,
    )

    class TrainingDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            item = {key: torch.tensor(value[index]) for key, value in encoded.items()}
            item["labels"] = torch.tensor(label_to_id[records[index].intent_id])
            return item

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
        save_strategy="no",
        report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=TrainingDataset()).train()
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
            {"model_version": version, "labels": labels, "config": asdict(config)}, indent=2
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

    def predict(self, text: str) -> TransformerPrediction:
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with self._torch.no_grad():
            probabilities = self._torch.softmax(self._model(**tokens).logits[0], dim=-1).tolist()
        labels = self._model.config.id2label
        distribution = {
            str(labels[index]): float(value) for index, value in enumerate(probabilities)
        }
        intent_id = max(distribution, key=distribution.get)
        return TransformerPrediction(
            intent_id, distribution[intent_id], distribution, self._version
        )
