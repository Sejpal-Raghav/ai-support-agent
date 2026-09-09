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
