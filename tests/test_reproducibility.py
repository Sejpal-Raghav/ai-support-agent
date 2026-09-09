from __future__ import annotations

from pathlib import Path

import pytest

from support_agent.reproducibility import (
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
