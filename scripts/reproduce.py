"""Reviewer-facing cached reproduction entrypoint; never downloads data or calls an LLM."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from support_agent.reproducibility import (
    load_frozen_manifest,
    required_headline_files,
    verify_frozen_artifacts,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/pipeline.yaml"), exists=True),
    mode: str = typer.Option("cached"),
    verify_hashes: bool = typer.Option(False),
    frozen_dir: Path = typer.Option(Path("data/frozen")),
) -> None:
    """Print frozen headline metrics after content verification in less than 15 minutes."""
    del config  # The config path is accepted for a stable reviewer-facing command contract.
    if mode != "cached":
        raise typer.BadParameter("only --mode cached is supported for reviewer reproduction")
    manifest_path = frozen_dir / "manifest.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(
            "frozen result manifest is absent; run the full pipeline and freeze final artifacts "
            "first"
        )
    manifest = load_frozen_manifest(manifest_path)
    required_headline_files(manifest)
    del verify_hashes  # Retained for CLI compatibility; verification is always required.
    verify_frozen_artifacts(frozen_dir, manifest)
    headline_path = frozen_dir / "headline_metrics.json"
    headline = json.loads(headline_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
