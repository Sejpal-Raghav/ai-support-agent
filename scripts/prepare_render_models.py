"""Cache the same pinned inference models during a native Render build."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("HF_HOME", str(root / "artifacts/huggingface"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        device="cpu",
    ).save(str(root / "artifacts/models/minilm"))
    subprocess.run([sys.executable, "scripts/download_vosk_model.py"], check=True)
    import imageio_ffmpeg

    binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not binary.is_file():
        raise RuntimeError("The bundled FFmpeg executable is missing.")
    subprocess.run([str(binary), "-version"], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
