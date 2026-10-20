"""Small typed FastAPI surface for transcription and evidence-backed responses."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from support_agent.voice.transcription import (
    MAX_AUDIO_BYTES,
    AudioValidationError,
    TranscriptionResult,
    load_vosk_model,
    transcribe_webm_audio,
)


class RespondRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must contain a non-whitespace character")
        return value


class RespondPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    text: str
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    score_type: Literal["semantic_similarity", "calibrated_probability"]
    retrieval_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
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
    message: str | None = None


@dataclass(frozen=True)
class ChatResult:
    intent: str
    confidence: float
    score_type: Literal["semantic_similarity", "calibrated_probability"]
    retrieval_similarity: float | None
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
            score_type="semantic_similarity",
            retrieval_similarity=None,
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
    inference_lock = Lock()

    @contextmanager
    def inference_slot() -> Iterator[None]:
        # Concurrent recognizers/encoders can exhaust a small container. Fail briefly
        # instead of retaining a queue of audio uploads or allocating another decoder.
        if not inference_lock.acquire(blocking=False):
            raise RuntimeError("The service is busy. Please try again shortly.")
        try:
            yield
        finally:
            inference_lock.release()

    def bounded_transcriber(audio: bytes, content_type: str | None) -> TranscriptionResult:
        with inference_slot():
            return active_transcriber(audio, content_type)

    @app.get("/api/health", response_model=HealthPayload)
    def health() -> HealthPayload:
        return (
            readiness()
            if readiness
            else HealthPayload(
                status="unconfigured",
                asr_ready=False,
                classifier_ready=False,
                retrieval_ready=False,
                message="The server has not configured the speech and response models.",
            )
        )

    @app.get("/api/ready", response_model=HealthPayload)
    def ready() -> HealthPayload:
        payload = health()
        if not (payload.asr_ready and payload.classifier_ready and payload.retrieval_ready):
            raise HTTPException(status_code=503, detail=payload.model_dump())
        return payload

    @app.post("/api/transcribe")
    async def transcribe(audio: UploadFile = File(...)) -> dict[str, object]:
        try:
            payload = await audio.read(MAX_AUDIO_BYTES + 1)
            if len(payload) > MAX_AUDIO_BYTES:
                raise AudioValidationError("audio_too_large", "Audio must be at most 10 MB.")
            result = await run_in_threadpool(bounded_transcriber, payload, audio.content_type)
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
        finally:
            await audio.close()
        return asdict(result)

    @app.post("/api/respond", response_model=RespondPayload)
    def respond(request: RespondRequest) -> RespondPayload:
        started = perf_counter()
        try:
            with inference_slot():
                result = active_responder(request.text.strip())
        except RuntimeError as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "response_unavailable", "message": "Response service unavailable."},
            ) from error
        return RespondPayload(
            session_id=request.session_id or str(uuid4()),
            text=request.text.strip(),
            intent=result.intent,
            confidence=result.confidence,
            score_type=result.score_type,
            retrieval_similarity=result.retrieval_similarity,
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
    """Load and bind the configured Vosk model before readiness can report true."""
    load_vosk_model(str(model_path))

    def transcriber(audio: bytes, content_type: str | None) -> TranscriptionResult:
        return transcribe_webm_audio(
            audio,
            content_type=content_type,
            model_path=model_path,
            asr_model_version=model_path.name,
        )

    return transcriber


def build_application() -> FastAPI:
    """Wire optional local artifacts without making imports download or train models."""
    static_dir = Path(os.environ["STATIC_DIR"]) if "STATIC_DIR" in os.environ else None
    cases_path = os.environ.get("DEMO_CASES_PATH")
    transformer_selection_path = os.environ.get("TRANSFORMER_SELECTION_PATH")
    vosk_path = os.environ.get("VOSK_MODEL_PATH")
    responder: ChatResponder | None = None
    transcriber: Transcriber | None = None
    if cases_path:
        from support_agent.demo_runtime import DemoSupportEngine

        transformer_dir: Path | None = None
        intent_threshold = 0.62
        if transformer_selection_path:
            selection = json.loads(Path(transformer_selection_path).read_text(encoding="utf-8"))
            if selection.get("quality_gate_passed") is not True:
                raise RuntimeError(
                    "The selected transformer has not passed validation quality gates."
                )
            transformer_dir = Path(selection["selected"]["artifact_dir"])
            intent_threshold = float(selection["confidence_threshold"])
        responder = DemoSupportEngine(
            Path(cases_path),
            transformer_dir=transformer_dir,
            intent_threshold=intent_threshold,
            model_path=Path(os.environ["MINILM_MODEL_PATH"])
            if os.environ.get("MINILM_MODEL_PATH")
            else None,
            embedding_backend=os.environ.get("EMBEDDING_BACKEND", "pytorch"),
        )
    if vosk_path:
        transcriber = build_vosk_transcriber(Path(vosk_path))

    def readiness() -> HealthPayload:
        return HealthPayload(
            status="ready"
            if responder and transcriber
            else "degraded"
            if responder
            else "unconfigured",
            asr_ready=transcriber is not None,
            classifier_ready=responder is not None,
            retrieval_ready=responder is not None,
            message=(
                None
                if responder and transcriber
                else "Speech recognition is unavailable. Typed questions are available."
                if responder
                else "The server has not configured the response model and evidence corpus."
            ),
        )

    return create_app(
        responder=responder,
        transcriber=transcriber,
        readiness=readiness,
        static_dir=static_dir,
    )


app = build_application()
