"""Compare the compact encoder against cached PyTorch MiniLM on fixed smoke probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from support_agent.demo_runtime import INTENT_SEEDS
from support_agent.embeddings import EMBEDDING_REVISION, OnnxSentenceEncoder

PROBES = (
    "Where is my delayed package?", "I want to return an item", "My item arrived damaged",
    "I received the wrong product", "Please cancel my order", "Is this product in stock?",
    "My payment was declined", "I cannot sign in to my account", "My refund hasn't arrived",
    "The carrier says delivered but nothing arrived", "Can I change my delivery address?",
    "When will this item be available?", "Please help me track my order",
    "The replacement is defective", "I need information about a product",
    "CAN I RETURN MY ORDER?", "I'd like my money back", "My package is missing",
    "My order arrived broken", "How can I modify an order?", "I can't log in",
    "I did not get the refund", "How do I cancel this?", "The item is unavailable",
    "Please check the order. " * 40,
)


def main() -> None:
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/minilm-onnx"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/compact-embedding-parity.json")
    )
    args = parser.parse_args()
    reference = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", revision=EMBEDDING_REVISION,
        device="cpu", local_files_only=True,
    )
    compact = OnnxSentenceEncoder(args.model)
    probes = [text for group in INTENT_SEEDS.values() for text in group] + list(PROBES)
    seeds = [". ".join(group) for group in INTENT_SEEDS.values()]
    expected = reference.encode(probes, normalize_embeddings=True, show_progress_bar=False)
    actual = compact.encode(probes, batch_size=4)
    cosine = np.sum(expected * actual, axis=1)
    reference_seeds = reference.encode(seeds, normalize_embeddings=True)
    compact_seeds = compact.encode(seeds, batch_size=4)
    reference_intents = np.argmax(expected @ reference_seeds.T, axis=1)
    compact_intents = np.argmax(actual @ compact_seeds.T, axis=1)
    agreement = float(np.mean(reference_intents == compact_intents))
    result = {
        "probe_count": len(probes), "revision": EMBEDDING_REVISION,
        "min_cosine_to_pytorch": float(cosine.min()),
        "mean_cosine_to_pytorch": float(cosine.mean()),
        "intent_agreement": agreement,
        "passed": bool(cosine.min() >= 0.98 and agreement >= 0.95),
        "scope": "Fixed engineering smoke probes; not held-out task accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
