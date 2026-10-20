"""Deployment entrypoint must never silently start without its inference assets."""

import os
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

configure_environment = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_render.py")
)["configure_environment"]


def test_render_startup_explains_missing_secret_files(tmp_path: Path) -> None:
    with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match="Secret Files"):
        configure_environment(tmp_path, tmp_path / "secrets")


def test_render_startup_uses_secret_files_and_port_independent_paths(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "demo_cases.jsonl").write_text("fixture", encoding="utf-8")
    (secrets / "demo_cases.manifest.json").write_text("{}", encoding="utf-8")
    frontend = tmp_path / "frontend/dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("fixture", encoding="utf-8")
    (tmp_path / "artifacts/models/vosk-model-en-us-0.22-lgraph").mkdir(parents=True)
    with patch.dict(os.environ, {}, clear=True):
        configure_environment(tmp_path, secrets)
        assert os.environ["DEMO_CASES_PATH"] == str(secrets / "demo_cases.jsonl")
        assert os.environ["STATIC_DIR"] == str(frontend)
        assert os.environ["HF_HOME"] == str(tmp_path / "artifacts/huggingface")
        assert os.environ["HF_HUB_OFFLINE"] == "1"
