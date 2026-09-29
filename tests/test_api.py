from __future__ import annotations

from fastapi.testclient import TestClient

from support_agent.api.app import ChatResult, HealthPayload, create_app
from support_agent.voice.transcription import TranscriptionResult


def test_health_and_response_contracts() -> None:
    app = create_app(
        responder=lambda _: ChatResult(
            intent="delivery_issue",
            confidence=0.91,
            score_type="calibrated_probability",
            retrieval_similarity=0.83,
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
    assert response.json()["score_type"] == "calibrated_probability"
    assert response.json()["retrieval_similarity"] == 0.83


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


def test_response_failure_is_a_typed_service_error() -> None:
    def failing_responder(_: str) -> ChatResult:
        raise RuntimeError("model artifact missing")

    response = TestClient(create_app(responder=failing_responder)).post(
        "/api/respond", json={"text": "where is my parcel", "session_id": "s1"}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "response_unavailable",
        "message": "Response service unavailable.",
    }


def test_static_frontend_is_served_after_api_routes(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<h1>Listening desk</h1>", encoding="utf-8")
    app = create_app(static_dir=tmp_path)
    client = TestClient(app)
    assert client.get("/").text == "<h1>Listening desk</h1>"
    assert client.get("/api/health").status_code == 200
