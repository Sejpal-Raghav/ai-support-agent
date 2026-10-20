from __future__ import annotations

from pathlib import Path

import pytest

from support_agent.reproducibility import (
    FrozenArtifactManifest,
    freeze_artifacts,
    load_frozen_manifest,
    required_headline_files,
    verify_frozen_artifacts,
)


def test_frozen_artifacts_are_content_addressed_and_verified(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    files = []
    for name in ("headline_metrics.json", "example_evaluations.jsonl", "evaluation_result.json"):
        path = source_dir / name
        path.write_text("fixture", encoding="utf-8")
        files.append(path)
    frozen_dir = tmp_path / "frozen"
    frozen = freeze_artifacts(files, frozen_dir, "a" * 64)
    required_headline_files(frozen)
    verify_frozen_artifacts(frozen_dir, load_frozen_manifest(frozen_dir / "manifest.json"))
    (frozen_dir / "headline_metrics.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_artifacts(frozen_dir, frozen)


def test_frozen_manifest_cannot_read_outside_its_directory(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (tmp_path / "outside.json").write_text("private", encoding="utf-8")
    manifest = FrozenArtifactManifest(
        source_manifest_sha256="a" * 64, files={"../outside.json": "b" * 64}
    )
    with pytest.raises(ValueError, match="inside"):
        verify_frozen_artifacts(frozen, manifest)


def test_duplicate_frozen_basenames_fail_before_copying(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    sources = [first / "result.json", second / "result.json"]
    for path in sources:
        path.write_text("fixture", encoding="utf-8")
    frozen = tmp_path / "frozen"
    with pytest.raises(ValueError, match="unique"):
        freeze_artifacts(sources, frozen, "a" * 64)
    assert not frozen.exists()
