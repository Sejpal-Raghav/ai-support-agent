"""Deployment entrypoint must never silently start without its inference assets."""

import json
import os
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

from support_agent.deployment import BUNDLE_NAME, package_evidence
from support_agent.hashing import file_sha256

configure_environment = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_render.py")
)["configure_environment"]


def test_render_startup_explains_missing_secret_files(tmp_path: Path) -> None:
    with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match="Secret Files"):
        configure_environment(tmp_path, tmp_path / "secrets")


@pytest.fixture
def deployment_assets(tmp_path: Path) -> tuple[Path, Path]:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    corpus = secrets / "demo_cases.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "case_id": "one",
                "source_thread_id": "thread",
                "split": "train",
            }
        ),
        encoding="utf-8",
    )
    (secrets / "demo_cases.manifest.json").write_text(
        json.dumps(
            {
                "corpus_sha256": file_sha256(corpus),
                "thread_partitions": {"thread": "train"},
            }
        ),
        encoding="utf-8",
    )
    frontend = tmp_path / "frontend/dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("fixture", encoding="utf-8")
    (tmp_path / "artifacts/models/vosk-model-en-us-0.22-lgraph").mkdir(parents=True)
    model = tmp_path / "artifacts/models/minilm"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "modules.json").write_text("[]", encoding="utf-8")
    return tmp_path, secrets


def test_render_startup_uses_secret_files_and_port_independent_paths(deployment_assets) -> None:
    tmp_path, secrets = deployment_assets
    frontend = tmp_path / "frontend/dist"
    with patch.dict(os.environ, {}, clear=True):
        configure_environment(tmp_path, secrets)
        assert os.environ["DEMO_CASES_PATH"] == str(secrets / "demo_cases.jsonl")
        assert os.environ["STATIC_DIR"] == str(frontend)
        assert os.environ["HF_HOME"] == str(tmp_path / "artifacts/huggingface")
        assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_render_startup_accepts_one_bundle_without_the_legacy_secret_pair(deployment_assets):
    root, secrets = deployment_assets
    corpus = secrets / "demo_cases.jsonl"
    package_evidence(corpus, secrets / BUNDLE_NAME)
    corpus.unlink()
    corpus.with_suffix(".manifest.json").unlink()
    with patch.dict(os.environ, {}, clear=True):
        configure_environment(root, secrets)
        assert Path(os.environ["DEMO_CASES_PATH"]).is_relative_to(
            root / "artifacts/deployed_evidence"
        )


def test_render_startup_reports_missing_prepared_model(deployment_assets):
    root, secrets = deployment_assets
    (root / "artifacts/models/minilm/modules.json").unlink()
    with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match="MiniLM"):
        configure_environment(root, secrets)


@pytest.mark.parametrize("bundle_exists", [False, True])
def test_render_startup_respects_explicit_evidence_path_over_bundle(
    deployment_assets, bundle_exists
):
    root, secrets = deployment_assets
    bundle = secrets / BUNDLE_NAME
    if bundle_exists:
        bundle.write_text("bad bundle", encoding="ascii")
    environment = {
        "DEMO_CASES_PATH": str(secrets / "demo_cases.jsonl"),
        "INFERENCE_BUNDLE_PATH": str(bundle),
    }
    with patch.dict(os.environ, environment, clear=True):
        configure_environment(root, secrets)
        assert os.environ["DEMO_CASES_PATH"] == str(secrets / "demo_cases.jsonl")


def test_render_startup_explains_missing_explicit_bundle(deployment_assets):
    root, secrets = deployment_assets
    environment = {"INFERENCE_BUNDLE_PATH": str(secrets / "missing.b64")}
    with (
        patch.dict(os.environ, environment, clear=True),
        pytest.raises(RuntimeError, match="INFERENCE_BUNDLE_PATH does not exist"),
    ):
        configure_environment(root, secrets)
