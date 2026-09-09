"""Grounded reply drafting with local immutable cache keys and evidence citations."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from support_agent.hashing import canonical_sha256, file_sha256
from support_agent.schemas import RetrievedCase


class DraftPayload(BaseModel):
    """The minimal structured output accepted from a reply generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    draft_reply: str = Field(max_length=280)
    cited_case_ids: tuple[str, ...] = Field(max_length=3)


class ResponseGenerator(Protocol):
    """Small interface that permits deterministic fake generators in tests."""

    def create(self, *, instructions: str, input_text: str) -> str:
        """Return the model's JSON-only text response."""


@dataclass(frozen=True)
class DraftResult:
    """Generated proposal and immutable provenance needed by downstream policy/evaluation."""

    draft_reply: str
    cited_case_ids: tuple[str, ...]
    prompt_sha256: str
    generator_model_snapshot: str
    cache_key: str
    from_cache: bool


def _prompt_path() -> Path:
    return Path("configs/prompts/drafter_v1.txt")


def _render_prompt(brand_id: str) -> tuple[str, str]:
    path = _prompt_path()
    template = path.read_text(encoding="utf-8")
    prompt = template.replace("{{brand_id}}", brand_id)
    return prompt, file_sha256(path)


def build_drafter_input(
    brand_id: str,
    context_text: str,
    predicted_intent: str,
    retrieved_cases: Sequence[RetrievedCase],
) -> str:
    """Serialize only current context and explicitly auditable train-case evidence."""
    if not 1 <= len(retrieved_cases) <= 3:
        raise ValueError("a grounded draft requires one to three retrieved cases")
    if any(case.intent_id != predicted_intent for case in retrieved_cases):
        raise ValueError("drafter evidence must all match the predicted intent")
    payload = {
        "brand_id": brand_id,
        "current_customer_context": context_text,
        "predicted_intent": predicted_intent,
        "historical_evidence": [
            {
                "case_id": case.case_id,
                "historical_customer_message": case.customer_text,
                "historical_brand_reply": case.brand_reply_text,
                "source_uri": case.source_uri,
            }
            for case in retrieved_cases
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class _OpenAIResponseGenerator:
    """Thin Responses API adapter; network use occurs only when explicitly invoked."""

    def __init__(self, model: str, temperature: float, max_output_tokens: int) -> None:
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def create(self, *, instructions: str, input_text: str) -> str:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for fresh draft generation")
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


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def generate_grounded_draft(
    brand_id: str,
    context_text: str,
    predicted_intent: str,
    retrieved_cases: Sequence[RetrievedCase],
    generator_model_snapshot: str,
    cache_dir: Path,
    generator: ResponseGenerator | None = None,
) -> DraftResult:
    """Generate/cache a JSON proposal whose citations are restricted to retrieved evidence."""
    prompt, prompt_sha256 = _render_prompt(brand_id)
    input_text = build_drafter_input(brand_id, context_text, predicted_intent, retrieved_cases)
    cache_key = canonical_sha256(
        {
            "prompt_sha256": prompt_sha256,
            "model": generator_model_snapshot,
            "input": json.loads(input_text),
        }
    )
    path = _cache_path(cache_dir, cache_key)
    if path.exists():
        payload = DraftPayload.model_validate_json(path.read_text(encoding="utf-8"))
        return DraftResult(
            draft_reply=payload.draft_reply,
            cited_case_ids=payload.cited_case_ids,
            prompt_sha256=prompt_sha256,
            generator_model_snapshot=generator_model_snapshot,
            cache_key=cache_key,
            from_cache=True,
        )
    active_generator = generator or _OpenAIResponseGenerator(
        generator_model_snapshot, temperature=0.0, max_output_tokens=180
    )
    payload = DraftPayload.model_validate_json(
        active_generator.create(instructions=prompt, input_text=input_text)
    )
    evidence_ids = {case.case_id for case in retrieved_cases}
    if not payload.draft_reply:
        if payload.cited_case_ids:
            raise ValueError("an empty draft cannot cite evidence")
    elif not payload.cited_case_ids:
        raise ValueError("a non-empty draft must cite one or more retrieved cases")
    if not set(payload.cited_case_ids).issubset(evidence_ids):
        raise ValueError("drafter cited a case outside the retrieved evidence")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(), encoding="utf-8")
    return DraftResult(
        draft_reply=payload.draft_reply,
        cited_case_ids=payload.cited_case_ids,
        prompt_sha256=prompt_sha256,
        generator_model_snapshot=generator_model_snapshot,
        cache_key=cache_key,
        from_cache=False,
    )
