"""Download the Kaggle Customer Support on Twitter dataset."""

from pathlib import Path

import typer

from support_agent.data.download import DEFAULT_DESTINATION, download_dataset

app = typer.Typer(add_completion=False)


@app.command()
def main(
    destination: Path = typer.Option(
        DEFAULT_DESTINATION,
        "--destination",
        help="Where to place twcs.csv for the ingestion pipeline.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing destination file.",
    ),
) -> None:
    """Download from Kaggle and place the validated source CSV."""
    installed_path = download_dataset(destination, overwrite=overwrite)
    typer.echo(f"Dataset ready: {installed_path}")


if __name__ == "__main__":
    app()
