"""Canonical hashing utilities used by manifests and LLM caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible content."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, chunk_bytes: int = 1_048_576) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_jsonl_sha256(path: Path, expected_hash: str) -> str:
    """Accept only exact bytes or equivalent line-ending/BOM/final-newline transport changes."""
    if file_sha256(path) == expected_hash:
        return expected_hash
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    for newline in ("\n", "\r\n"):
        body = normalized.replace("\n", newline)
        for suffix in ("", newline):
            candidate = (body + suffix).encode("utf-8")
            if hashlib.sha256(candidate).hexdigest() == expected_hash:
                return expected_hash
    raise ValueError(
        "The demo corpus content does not match its manifest after line-ending normalization. "
        "Replace both Render Secret Files with the matching demo_cases.jsonl and "
        "demo_cases.manifest.json from the same local export; do not edit their contents."
    )
