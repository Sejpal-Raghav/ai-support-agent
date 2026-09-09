"""Freeze compact, content-addressed cached evaluation artifacts for reviewers."""

from __future__ import annotations

from pathlib import Path

import typer

from support_agent.reproducibility import freeze_artifacts

app = typer.Typer(add_completion=False)


@app.command()
def main(
    source_manifest_sha256: str = typer.Option(
        ..., help="SHA-256 from artifacts/split_manifest.json"
    ),
    artifact: list[Path] = typer.Option(
        ..., exists=True, help="Repeat for each result file to freeze"
    ),
    frozen_dir: Path = typer.Option(Path("data/frozen")),
) -> None:
    """Copy final cached results, then write data/frozen/manifest.json with their hashes."""
    manifest = freeze_artifacts(artifact, frozen_dir, source_manifest_sha256)
    typer.echo(f"Froze {len(manifest.files)} result file(s) in {frozen_dir}")


if __name__ == "__main__":
    app()
