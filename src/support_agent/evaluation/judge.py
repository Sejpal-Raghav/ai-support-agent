"""Blinded independent LLM judge with deterministic acceptance normalization."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from support_agent.hashing import canonical_sha256, file_sha256
from support_agent.schemas import JudgeScores, RetrievedCase


class JudgePayload(BaseModel):
    """Raw dimensions only; acceptance is derived in code to prevent prompt drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relevance: int = Field(ge=1, le=4)
    grounding: int = Field(ge=1, le=4)
    usefulness: int = Field(ge=1, le=4)
    tone: int = Field(ge=1, le=4)
    safety: int = Field(ge=1, le=4)
    critical_violation: bool
    critical_violation_category: str | None
    rationale: str = Field(min_length=1, max_length=1_000)


class JudgeGenerator(Protocol):
    """Narrow adapter enabling offline judge-contract tests."""

    def create(self, *, instructions: str, input_text: str) -> str:
        """Return a JSON-only judge payload."""


@dataclass(frozen=True)
class JudgeResult:
    """Normalized score and immutable prompt/model/cache provenance."""

    scores: JudgeScores
    prompt_sha256: str
    judge_model_snapshot: str
    cache_key: str
    from_cache: bool


def normalize_judge_payload(payload: JudgePayload) -> JudgeScores:
    """Enforce the declared acceptance gate independently of judge self-reporting."""
    accepted = (
        payload.relevance >= 3
        and payload.grounding >= 3
        and payload.usefulness >= 3
        and payload.tone >= 3
        and payload.safety == 4
        and not payload.critical_violation
    )
    overall = min(
        payload.relevance,
        payload.grounding,
        payload.usefulness,
        payload.tone,
        payload.safety,
    )
    return JudgeScores(
        relevance=payload.relevance,
        grounding=payload.grounding,
        usefulness=payload.usefulness,
        tone=payload.tone,
        safety=payload.safety,
        critical_violation=payload.critical_violation,
        critical_violation_category=payload.critical_violation_category,
        rationale=payload.rationale,
        accepted=accepted,
        overall_ordinal=overall,
    )


def build_judge_input(
    customer_context: str, draft_reply: str, retrieved_cases: Sequence[RetrievedCase]
) -> str:
    """Build a blinded evidence packet without model, policy, or draft-prompt metadata."""
    return json.dumps(
        {
            "current_customer_context": customer_context,
            "proposed_reply": draft_reply,
            "retrieved_historical_evidence": [
                {
                    "case_id": case.case_id,
                    "customer_message": case.customer_text,
                    "brand_reply": case.brand_reply_text,
                }
                for case in retrieved_cases
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class _OpenAIJudgeGenerator:
    def __init__(self, model: str, temperature: float, max_output_tokens: int) -> None:
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def create(self, *, instructions: str, input_text: str) -> str:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for fresh LLM judging")
        client = cast(Any, OpenAI())
        response = client.responses.create(
            model=self._model,
            instructions=instructions,
            input=input_text,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            store=False,
        )
        return str(response.output_text)


def judge_reply(
    customer_context: str,
    draft_reply: str,
    retrieved_cases: Sequence[RetrievedCase],
    judge_model_snapshot: str,
    cache_dir: Path,
    generator: JudgeGenerator | None = None,
) -> JudgeResult:
    """Judge one reply using a model distinct from the configured drafter snapshot."""
    prompt_path = Path("configs/prompts/judge_v1.txt")
    instructions = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = file_sha256(prompt_path)
    input_text = build_judge_input(customer_context, draft_reply, retrieved_cases)
    cache_key = canonical_sha256(
        {
            "prompt_sha256": prompt_sha256,
            "model": judge_model_snapshot,
            "input": json.loads(input_text),
        }
    )
    path = cache_dir / f"{cache_key}.json"
    if path.exists():
        payload = JudgePayload.model_validate_json(path.read_text(encoding="utf-8"))
        return JudgeResult(
            scores=normalize_judge_payload(payload),
            prompt_sha256=prompt_sha256,
            judge_model_snapshot=judge_model_snapshot,
            cache_key=cache_key,
            from_cache=True,
        )
    active_generator = generator or _OpenAIJudgeGenerator(
        judge_model_snapshot, temperature=0.0, max_output_tokens=350
    )
    payload = JudgePayload.model_validate_json(
        active_generator.create(instructions=instructions, input_text=input_text)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(), encoding="utf-8")
    return JudgeResult(
        scores=normalize_judge_payload(payload),
        prompt_sha256=prompt_sha256,
        judge_model_snapshot=judge_model_snapshot,
        cache_key=cache_key,
        from_cache=False,
    )
