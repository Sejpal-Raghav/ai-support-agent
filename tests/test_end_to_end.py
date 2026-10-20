"""Exercise audio to grounded response with deterministic model doubles."""

from __future__ import annotations

import json
import sys
import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from support_agent.api.app import create_app
from support_agent.demo_runtime import DemoSupportEngine
from support_agent.hashing import file_sha256
from support_agent.voice.transcription import transcribe_pcm_wav


class FixedEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return np.tile([1.0, 0.0], (len(texts), 1))


@pytest.fixture
def demo_engine(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FixedEncoder)
    )
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "AmazonHelp:reply-1",
                "customer_text": "Where is my package?",
                "reply_text": "Please check the carrier tracking in your order history.",
                "intent_id": "delivery_tracking",
                "source_thread_id": "thread-1",
                "split": "train",
            }
        ),
        encoding="utf-8",
    )
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "corpus_sha256": file_sha256(path),
                "thread_partitions": {"thread-1": "train"},
            }
        ),
        encoding="utf-8",
    )
    return DemoSupportEngine(path)


def test_audio_to_evidence_backed_response(demo_engine) -> None:
    class Recognizer:
        def AcceptWaveform(self, data):  # noqa: N802
            return False

        def FinalResult(self):  # noqa: N802
            return json.dumps({"text": "where is my package"})

    recording = BytesIO()
    with wave.open(recording, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes((1000).to_bytes(2, "little", signed=True) * 16000)

    def transcriber(payload, content_type):
        return transcribe_pcm_wav(payload, recognizer=Recognizer(), asr_model_version="fixture")

    client = TestClient(create_app(responder=demo_engine, transcriber=transcriber))
    transcript = client.post(
        "/api/transcribe", files={"audio": ("clip.wav", recording.getvalue(), "audio/wav")}
    )
    assert transcript.status_code == 200
    response = client.post("/api/respond", json={"text": transcript.json()["transcript"]})
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "respond"
    assert body["evidence_case_ids"] == ["AmazonHelp:reply-1"]
    assert body["retrieval_index_version"].startswith("amazonhelp-")
    assert "carrier tracking" in body["response"]


@pytest.mark.parametrize(
    "text",
    [
        "My password is secret, where is my package?",
        "Use verification code 123456 to check my order",
        "Contact me at customer@example.com about my package",
    ],
)
def test_credential_requests_fall_back_without_evidence(demo_engine, text) -> None:
    result = demo_engine(text)
    assert result.action == "fallback"
    assert result.reasons == ("privacy_or_credential_risk",)
    assert result.evidence_case_ids == ()
    assert "secret" not in result.response


def test_duplicate_charge_requires_billing_review(demo_engine) -> None:
    result = demo_engine("I was charged twice for my package")
    assert result.action == "fallback"
    assert result.reasons == ("money_or_billing_risk",)
    assert result.evidence_case_ids == ()


def test_runtime_rejects_tampered_evidence(tmp_path: Path, demo_engine) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("carrier", "fraud"), encoding="utf-8")
    with pytest.raises(ValueError, match="content does not match"):
        DemoSupportEngine(path)


def test_runtime_accepts_secret_file_final_newline_and_bom(tmp_path: Path, demo_engine) -> None:
    path = tmp_path / "cases.jsonl"
    original_version = demo_engine("where is my package").retrieval_index_version
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes() + b"\n")
    loaded = DemoSupportEngine(path)
    assert loaded("where is my package").retrieval_index_version == original_version


def test_runtime_rejects_held_out_provenance(tmp_path: Path, demo_engine) -> None:
    path = tmp_path / "cases.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["thread_partitions"]["thread-1"] = "test"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="non-train"):
        DemoSupportEngine(tmp_path / "cases.jsonl")
