"""Validate completed annotation packets before any tuning or test reporting."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from support_agent.evaluation.annotations import (
    load_completed_annotations,
    validate_annotation_consistency,
)
from support_agent.schemas import IntentTaxonomy

app = typer.Typer(add_completion=False)


def _load_taxonomy(path: Path) -> IntentTaxonomy:
    return IntentTaxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@app.command()
def main(
    packet: Path = typer.Option(..., exists=True),
    taxonomy_path: Path = typer.Option(Path("configs/taxonomy.yaml"), exists=True),
    expected_partition: str = typer.Option(...),
    expected_count: int = typer.Option(..., min=1),
    audit_output: Path = typer.Option(Path("artifacts/annotation_audit.json")),
) -> None:
    """Fail closed unless all required annotation labels are complete and consistent."""
    records = load_completed_annotations(packet)
    audit = validate_annotation_consistency(records, _load_taxonomy(taxonomy_path))
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
    failures: list[str] = []
    if audit.record_count != expected_count:
        failures.append(f"expected {expected_count} records, found {audit.record_count}")
    if audit.partitions != {expected_partition: expected_count}:
        failures.append(f"partition counts are {audit.partitions}, not {expected_partition!r}")
    if audit.invalid_intent_ids:
        failures.append(f"invalid intent labels: {audit.invalid_intent_ids}")
    if audit.invalid_escalation_examples:
        failures.append("escalation reason consistency failed")
    if audit.duplicate_example_ids:
        failures.append("duplicate annotation example IDs")
    if failures:
        raise typer.Exit(code=1)
    typer.echo(f"Validated {audit.record_count} completed labels in {packet}")


if __name__ == "__main__":
    app()
