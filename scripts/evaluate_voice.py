"""Evaluate real held-out clips locally or reproduce a cached voice result without models."""

from __future__ import annotations

import csv
import json
import mimetypes
from pathlib import Path

import typer

from support_agent.evaluation.voice import VoiceEvaluationRecord, evaluate_voice
from support_agent.hashing import file_sha256
from support_agent.voice.transcription import AudioValidationError, transcribe_webm_audio

app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest: Path | None = typer.Option(None, exists=True),
    predictions: Path | None = typer.Option(None, exists=True),
    model_path: Path = typer.Option(Path("artifacts/models/vosk-model-en-us-0.22-lgraph")),
    output_dir: Path = typer.Option(Path("artifacts/voice_evaluation")),
) -> None:
    """Use --manifest for real audio, or --predictions for cached metric reproduction."""
    if (manifest is None) == (predictions is None):
        raise typer.BadParameter("supply exactly one of --manifest or --predictions")
    records: list[VoiceEvaluationRecord] = []
    if predictions is not None:
        records = [
            VoiceEvaluationRecord.model_validate_json(line)
            for line in predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source_hash = file_sha256(predictions)
    else:
        assert manifest is not None
        source_hash = file_sha256(manifest)
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise typer.BadParameter("voice manifest is empty")
        for row in rows:
            if row["consent"] != "yes" or row["split"] != "held_out":
                raise typer.BadParameter("every clip needs consent=yes and split=held_out")
            audio_path = manifest.parent / row["audio_path"]
            content_type = mimetypes.guess_type(audio_path.name)[0]
            if audio_path.suffix.lower() == ".webm":
                content_type = "audio/webm"
            try:
                result = transcribe_webm_audio(
                    audio_path.read_bytes(),
                    content_type=content_type,
                    model_path=model_path,
                    allow_empty_transcript=True,
                )
            except AudioValidationError as error:
                raise typer.BadParameter(
                    f"{row['recording_id']}: {error.code}; repair the audio format and rerun "
                    "without dropping held-out clips"
                ) from error
            records.append(
                VoiceEvaluationRecord(
                    recording_id=row["recording_id"],
                    speaker_id=row["speaker_id"],
                    noise=row["noise"],
                    split=row["split"],
                    reference=row["reference"],
                    transcript=result.transcript,
                    asr_model_version=result.asr_model_version,
                    duration_ms=result.duration_ms,
                    latency_ms=result.latency_ms,
                )
            )
    report = evaluate_voice(records)
    report["input_sha256"] = source_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = output_dir / "voice_predictions.jsonl"
    if predictions is None or predictions.resolve() != cached.resolve():
        cached.write_text(
            "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
        )
    (output_dir / "voice_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
