"""Download and verify the pinned Vosk model used by the local voice demo."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

MODEL_NAME = "vosk-model-en-us-0.22-lgraph"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MODEL_SHA256 = "d9838b4aaa82a75c4a17f5aca300eaca129aaab2a7cbf951bafbB500eb9c4334".lower()


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a trusted-checksum ZIP while still rejecting path traversal."""
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(destination)


@app.command()
def main(output_dir: Path = typer.Option(Path("artifacts/models"))) -> None:
    """Materialize the immutable local ASR model under the ignored artifacts tree."""
    model_dir = output_dir / MODEL_NAME
    if model_dir.is_dir():
        typer.echo(f"Vosk model already exists at {model_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{MODEL_NAME}.zip"
    partial_path = output_dir / f"{MODEL_NAME}.zip.part"
    if not archive_path.is_file() or file_sha256(archive_path) != MODEL_SHA256:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response, partial_path.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        if file_sha256(partial_path) != MODEL_SHA256:
            partial_path.unlink(missing_ok=True)
            raise ValueError("downloaded Vosk model failed SHA-256 verification")
        partial_path.replace(archive_path)

    safe_extract(archive_path, output_dir)
    if not model_dir.is_dir():
        raise RuntimeError("Vosk archive did not contain the expected model directory")
    typer.echo(f"Vosk model ready at {model_dir}")


if __name__ == "__main__":
    app()
