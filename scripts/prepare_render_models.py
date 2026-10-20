"""Cache the same pinned inference models during a native Render build."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
ONNX_FILES = {
    "onnx/model_quint8_avx2.onnx": (
        "model.onnx", "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"
    ),
    "tokenizer.json": (
        "tokenizer.json", "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
    ),
}


def download_verified(url: str, output: Path, expected_hash: str) -> None:
    """Only install model bytes matching the immutable source checksums."""
    partial = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as target:
        shutil.copyfileobj(response, target)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != expected_hash:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Model checksum mismatch: {output.name}")
    partial.replace(output)


def prepare_onnx(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    model_dir = root / "artifacts/models/minilm"
    model_dir.mkdir(parents=True, exist_ok=True)
    for remote, (local, expected_hash) in ONNX_FILES.items():
        print(f"Preparing pinned MiniLM file: {remote}", flush=True)
        downloaded = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename=remote,
            revision=REVISION,
            cache_dir=str(Path(os.environ.get("MODEL_DOWNLOAD_CACHE", ".cache/models")) / "minilm"),
        )
        download_verified(Path(downloaded).resolve().as_uri(), model_dir / local, expected_hash)
        print(f"Verified {local}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("HF_HOME", str(root / "artifacts/huggingface"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    if os.environ.get("EMBEDDING_BACKEND", "pytorch") == "onnx":
        prepare_onnx(root)
        vosk_name = "vosk-model-small-en-us-0.15"
    else:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", revision=REVISION, device="cpu"
        ).save(str(root / "artifacts/models/minilm"))
        vosk_name = "vosk-model-en-us-0.22-lgraph"
    if os.environ.get("MODEL_DOWNLOAD_CACHE"):
        model_cache = Path(os.environ["MODEL_DOWNLOAD_CACHE"]) / "vosk"
        subprocess.run(
            [sys.executable, "scripts/download_vosk_model.py", "--model-name", vosk_name,
             "--output-dir", str(model_cache)], check=True
        )
        shutil.copytree(
            model_cache / vosk_name, root / "artifacts/models" / vosk_name, dirs_exist_ok=True
        )
    else:
        subprocess.run(
            [sys.executable, "scripts/download_vosk_model.py", "--model-name", vosk_name],
            check=True,
        )
    import imageio_ffmpeg

    binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not binary.is_file():
        raise RuntimeError("The bundled FFmpeg executable is missing.")
    subprocess.run([str(binary), "-version"], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
