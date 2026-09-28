"""Create the deterministic 750k-row source subset used by the localhost demo."""

from pathlib import Path

import typer

from support_agent.data.ingest import ingest_csv

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_csv: Path = typer.Option(Path("data/raw/twcs.csv"), exists=True, readable=True),
    output_path: Path = typer.Option(Path("data/interim/messages.parquet")),
    max_rows: int = typer.Option(750_000, min=10_000),
) -> None:
    stats = ingest_csv(raw_csv, output_path, max_rows=max_rows)
    typer.echo(
        f"Wrote {stats.rows_written:,} validated rows to {output_path} "
        f"({stats.rows_rejected:,} rejected)"
    )


if __name__ == "__main__":
    app()
