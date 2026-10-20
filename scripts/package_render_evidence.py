"""Export one private ASCII inference bundle; generated output stays outside Git."""

from pathlib import Path

import typer

from support_agent.deployment import BUNDLE_NAME, package_evidence

app = typer.Typer(add_completion=False)


@app.command()
def main(
    cases_path: Path = typer.Option(Path("artifacts/demo_cases.jsonl"), exists=True),
    output_path: Path = typer.Option(Path("artifacts/render") / BUNDLE_NAME),
) -> None:
    count = package_evidence(cases_path, output_path)
    typer.echo(f"Exported {count} verified train cases to {output_path}")
    typer.echo(f"Render Secret File name: {BUNDLE_NAME} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    app()
