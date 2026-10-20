"""Build-time inference downloads must match their immutable checksums."""

import hashlib
import runpy
from io import BytesIO
from pathlib import Path

import pytest

prepare = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/prepare_render_models.py")
)


def test_model_download_is_installed_only_after_hash_verification(tmp_path, monkeypatch):
    content = b"verified model fixture"
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: BytesIO(content)
    )
    target = tmp_path / "model.onnx"
    prepare["download_verified"](
        "https://fixture.invalid/model", target, hashlib.sha256(content).hexdigest()
    )
    assert target.read_bytes() == content


def test_bad_model_download_does_not_replace_an_existing_model(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: BytesIO(b"corrupt"))
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous verified model")
    with pytest.raises(ValueError, match="checksum"):
        prepare["download_verified"]("https://fixture.invalid/model", target, "0" * 64)
    assert target.read_bytes() == b"previous verified model"
    assert not target.with_suffix(".onnx.part").exists()
