"""Small typed FastAPI surface for transcription and evidence-backed responses."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from support_agent.voice.transcription import (
    AudioValidationError,
    TranscriptionResult,
    transcribe_webm_audio,
)


class RespondRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)


class RespondPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    text: str
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    response: str
    action: str
    reasons: tuple[str, ...]
    evidence_case_ids: tuple[str, ...]
    classifier_version: str
    retrieval_index_version: str
    latency_ms: int = Field(ge=0)


class HealthPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    asr_ready: bool
    classifier_ready: bool
    retrieval_ready: bool


@dataclass(frozen=True)
class ChatResult:
    intent: str
    confidence: float
    response: str
    action: str
    reasons: tuple[str, ...]
    evidence_case_ids: tuple[str, ...]
    classifier_version: str
    retrieval_index_version: str


ChatResponder = Callable[[str], ChatResult]
Transcriber = Callable[[bytes, str | None], TranscriptionResult]


def create_app(
    *,
    responder: ChatResponder | None = None,
    transcriber: Transcriber | None = None,
    readiness: Callable[[], HealthPayload] | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Create an injectable app; production wiring is deliberately artifact-dependent."""
    app = FastAPI(title="Voice Support Chatbot", version="0.1.0")

    def unavailable_responder(_: str) -> ChatResult:
        return ChatResult(
            intent="unavailable",
            confidence=0.0,
            response="The chatbot model is still loading. Please try again shortly.",
            action="fallback",
            reasons=("classifier_unavailable",),
            evidence_case_ids=(),
            classifier_version="unavailable",
            retrieval_index_version="unavailable",
        )

    def unavailable_transcriber(_: bytes, __: str | None) -> TranscriptionResult:
        raise RuntimeError("Speech recognition is still loading.")

    active_responder = responder or unavailable_responder
    active_transcriber = transcriber or unavailable_transcriber

    @app.get("/api/health", response_model=HealthPayload)
    def health() -> HealthPayload:
        return readiness() if readiness else HealthPayload(
            status="degraded", asr_ready=False, classifier_ready=False, retrieval_ready=False
        )

    @app.post("/api/transcribe")
    async def transcribe(audio: UploadFile = File(...)) -> dict[str, object]:
        try:
            result = active_transcriber(await audio.read(), audio.content_type)
        except AudioValidationError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": error.code, "message": str(error)},
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "asr_unavailable", "message": str(error)},
            ) from error
        return asdict(result)

    @app.post("/api/respond", response_model=RespondPayload)
    def respond(request: RespondRequest) -> RespondPayload:
        started = perf_counter()
        result = active_responder(request.text.strip())
        return RespondPayload(
            session_id=request.session_id or str(uuid4()),
            text=request.text.strip(),
            intent=result.intent,
            confidence=result.confidence,
            response=result.response,
            action=result.action,
            reasons=result.reasons,
            evidence_case_ids=result.evidence_case_ids,
            classifier_version=result.classifier_version,
            retrieval_index_version=result.retrieval_index_version,
            latency_ms=round((perf_counter() - started) * 1000),
        )

    if static_dir and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


def build_vosk_transcriber(model_path: Path) -> Transcriber:
    """Bind production configuration without loading a model at import time."""

    def transcriber(audio: bytes, content_type: str | None) -> TranscriptionResult:
        return transcribe_webm_audio(audio, content_type=content_type, model_path=model_path)

    return transcriber


app = create_app(static_dir=Path(os.environ["STATIC_DIR"]) if "STATIC_DIR" in os.environ else None)
