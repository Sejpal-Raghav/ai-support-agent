"""Pooling parity, offline backend selection, and bounded inference allocations."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from support_agent.embeddings import load_sentence_encoder, mean_pool


def test_mean_pool_excludes_padding_and_normalizes_vectors():
    hidden = np.array([[[3, 0], [0, 4], [100, 100]]], dtype=np.float32)
    pooled = mean_pool(hidden, np.array([[1, 1, 0]], dtype=np.int64))
    np.testing.assert_allclose(pooled, [[0.6, 0.8]], atol=1e-6)
    assert pooled.dtype == np.float32


def test_empty_mask_does_not_produce_nan():
    pooled = mean_pool(
        np.zeros((1, 2, 3), dtype=np.float32), np.zeros((1, 2), dtype=np.int64)
    )
    assert np.isfinite(pooled).all()


def test_onnx_backend_bounds_batches_without_importing_pytorch(monkeypatch):
    calls = []

    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()

        def enable_truncation(self, **kwargs):
            assert kwargs == {"max_length": 256}

        def enable_padding(self, **kwargs):
            pass

        def encode_batch(self, texts):
            assert len(texts) == 1
            return [
                SimpleNamespace(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 0])
                for _ in texts
            ]

    class Session:
        def __init__(self, path, options, providers):
            assert options.intra_op_num_threads == 1
            assert options.enable_cpu_mem_arena is False
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name=key) for key in ("input_ids", "attention_mask")]

        def run(self, outputs, inputs):
            assert all(value.dtype == np.int64 for value in inputs.values())
            assert inputs["input_ids"].shape[0] == 1
            assert "token_type_ids" not in inputs
            calls.append(inputs)
            return [
                np.tile([[[1, 0], [1, 0]]], (inputs["input_ids"].shape[0], 1, 1)).astype(np.float32)
            ]

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))
    monkeypatch.setitem(
        sys.modules, "onnxruntime",
        SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session),
    )
    model = load_sentence_encoder(Path("fixture"), "onnx")
    result = model.encode(["one"] * 10, batch_size=32)
    np.testing.assert_array_equal(result, [[1, 0]] * 10)
    assert [call["input_ids"].shape[0] for call in calls] == [1] * 10


def test_onnx_backend_requires_local_models():
    with pytest.raises(ValueError, match="local model"):
        load_sentence_encoder(None, "onnx")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        load_sentence_encoder(None, "download-a-model")
