"""Artifact-manifest contracts for reproducible pipeline phases."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from support_agent.hashing import file_sha256


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    command: tuple[str, ...]
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactRef, ...]
    created_at: datetime


class ManifestVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    mismatched_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]


def build_artifact_manifest(
    inputs: Sequence[ArtifactRef], outputs: Sequence[Path], command: tuple[str, ...]
) -> ArtifactManifest:
    """Build an immutable manifest after a phase has materialized its outputs."""
    output_refs = tuple(
        ArtifactRef(path=output.as_posix(), sha256=file_sha256(output)) for output in outputs
    )
    return ArtifactManifest(
        command=command,
        inputs=tuple(inputs),
        outputs=output_refs,
        created_at=datetime.now(UTC),
    )


def verify_manifest(manifest: ArtifactManifest, root: Path) -> ManifestVerification:
    """Verify all manifest artifacts below root without changing any artifact."""
    mismatched: list[str] = []
    missing: list[str] = []
    for artifact in (*manifest.inputs, *manifest.outputs):
        path = root / artifact.path
        if not path.is_file():
            missing.append(artifact.path)
        elif file_sha256(path) != artifact.sha256:
            mismatched.append(artifact.path)
    return ManifestVerification(
        valid=not mismatched and not missing,
        mismatched_paths=tuple(mismatched),
        missing_paths=tuple(missing),
    )
