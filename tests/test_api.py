from __future__ import annotations

from fastapi.testclient import TestClient

from support_agent.api.app import ChatResult, HealthPayload, create_app
from support_agent.voice.transcription import TranscriptionResult


def test_health_and_response_contracts() -> None:
    app = create_app(
        responder=lambda _: ChatResult(
            intent="delivery_issue",
            confidence=0.91,
            response="Please check your delivery tracking.",
            action="respond",
            reasons=(),
            evidence_case_ids=("case-1",),
            classifier_version="bertweet-test",
            retrieval_index_version="index-test",
        ),
        readiness=lambda: HealthPayload(
            status="ready", asr_ready=True, classifier_ready=True, retrieval_ready=True
        ),
    )
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ready"
    response = client.post(
        "/api/respond", json={"text": "where is my delivery", "session_id": "s1"}
    )
    assert response.status_code == 200
    assert response.json()["evidence_case_ids"] == ["case-1"]
    assert response.json()["session_id"] == "s1"


def test_transcription_contract_and_typed_error() -> None:
    def transcriber(_: bytes, __: str | None) -> TranscriptionResult:
        return TranscriptionResult("hello", 1000, "vosk-test", (), 12)

    client = TestClient(create_app(transcriber=transcriber))
    response = client.post(
        "/api/transcribe", files={"audio": ("clip.webm", b"audio", "audio/webm")}
    )
    assert response.status_code == 200
    assert response.json()["transcript"] == "hello"
    unavailable = TestClient(create_app()).post(
        "/api/transcribe", files={"audio": ("clip.webm", b"audio", "audio/webm")}
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "asr_unavailable"
