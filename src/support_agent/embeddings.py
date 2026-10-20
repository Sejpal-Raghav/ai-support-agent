"""Bounded, offline MiniLM inference without importing the training framework."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MAX_SEQUENCE_LENGTH = 256


class SentenceEncoder(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 1,
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]: ...


def mean_pool(
    hidden: NDArray[np.float32], attention_mask: NDArray[np.int64]
) -> NDArray[np.float32]:
    """Match MiniLM's attention-masked mean pooling and L2 normalization."""
    mask = attention_mask[..., None].astype(np.float32)
    pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
    normalized = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return cast(NDArray[np.float32], normalized.astype(np.float32))


class OnnxSentenceEncoder:
    """Run the pinned 8-bit MiniLM graph in small batches on one CPU thread."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # Do not retain a growing arena after long input or a startup embedding batch.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self._session = ort.InferenceSession(
            str(model_path / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 1,
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]:
        if not normalize_embeddings:
            raise ValueError("The retrieval encoder requires normalized embeddings.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = []
        # A single-text batch also keeps dynamic quantization independent of its neighbors.
        for text in texts:
            encoded = self._tokenizer.encode_batch([text])
            tokens = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in encoded], dtype=np.int64
                ),
                "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
            }
            inputs = {key: value for key, value in tokens.items() if key in self._input_names}
            hidden = self._session.run(None, inputs)[0]
            vectors.append(mean_pool(hidden, tokens["attention_mask"]))
        return cast(NDArray[np.float32], np.concatenate(vectors))


def load_sentence_encoder(model_path: Path | None, backend: str) -> SentenceEncoder:
    if backend == "onnx":
        if model_path is None:
            raise ValueError("The ONNX encoder requires a prepared local model directory.")
        return OnnxSentenceEncoder(model_path)
    if backend != "pytorch":
        raise ValueError(f"Unsupported embedding backend: {backend}")
    from sentence_transformers import SentenceTransformer

    return cast(
        SentenceEncoder,
        SentenceTransformer(
            str(model_path) if model_path is not None else "sentence-transformers/all-MiniLM-L6-v2",
            revision=None if model_path is not None else EMBEDDING_REVISION,
            device="cpu",
            local_files_only=True,
        ),
    )
