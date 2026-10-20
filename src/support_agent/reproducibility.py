"""Freeze and verify compact reviewer-facing artifacts without rerunning APIs or models."""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from support_agent.hashing import file_sha256


class FrozenArtifactManifest(BaseModel):
    """Content-addressed inventory of all files required for cached headline reproduction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    source_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    files: dict[str, str] = Field(min_length=1)


def _relative_path(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("frozen artifact must be inside its frozen directory") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("frozen artifact path is unsafe")
    return relative.as_posix()


def freeze_artifacts(
    source_files: Sequence[Path],
    frozen_dir: Path,
    source_manifest_sha256: str,
) -> FrozenArtifactManifest:
    """Copy explicit result files into the frozen directory and record exact SHA-256 hashes."""
    if re.fullmatch(r"[a-f0-9]{64}", source_manifest_sha256) is None:
        raise ValueError("source manifest SHA-256 must be 64 lowercase hex characters")
    if not source_files:
        raise ValueError("at least one result file must be frozen")
    names = [source.name for source in source_files]
    if len(names) != len(set(names)) or "manifest.json" in names:
        raise ValueError("frozen source basenames must be unique and cannot be manifest.json")
    for source in source_files:
        if not source.is_file():
            raise ValueError(f"frozen source file does not exist: {source}")
        if (frozen_dir / source.name).resolve() == source.resolve():
            raise ValueError("cannot freeze a file already inside frozen_dir")
        _relative_path(frozen_dir, frozen_dir / source.name)
    frozen_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for source in source_files:
        if not source.is_file():
            raise ValueError(f"frozen source file does not exist: {source}")
        destination = frozen_dir / source.name
        if destination.resolve() == source.resolve():
            raise ValueError("cannot freeze a file already inside frozen_dir")
        shutil.copy2(source, destination)
        hashes[_relative_path(frozen_dir, destination)] = file_sha256(destination)
    if not hashes:
        raise ValueError("at least one result file must be frozen")
    manifest = FrozenArtifactManifest(
        source_manifest_sha256=source_manifest_sha256,
        files=dict(sorted(hashes.items())),
    )
    (frozen_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_frozen_manifest(path: Path) -> FrozenArtifactManifest:
    """Read a strict frozen manifest, rejecting arbitrary extra fields."""
    return FrozenArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))


def verify_frozen_artifacts(frozen_dir: Path, manifest: FrozenArtifactManifest) -> None:
    """Raise if any frozen result is missing or hash-mismatched."""
    for relative, expected_hash in manifest.files.items():
        candidate = frozen_dir / relative
        _relative_path(frozen_dir, candidate)
        if not candidate.is_file():
            raise ValueError(f"frozen artifact is missing: {relative}")
        if file_sha256(candidate) != expected_hash:
            raise ValueError(f"frozen artifact hash mismatch: {relative}")


def required_headline_files(manifest: FrozenArtifactManifest) -> Mapping[str, str]:
    """Ensure cached reproduction contains the result table and per-example evidence audit."""
    required = {"headline_metrics.json", "example_evaluations.jsonl", "evaluation_result.json"}
    missing = required.difference(manifest.files)
    if missing:
        raise ValueError(
            f"frozen manifest is missing required cached result files: {sorted(missing)}"
        )
    return manifest.files
