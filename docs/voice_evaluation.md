# Held-out voice evaluation

Keep participant audio and transcripts under ignored `data/raw/voice/`. Obtain consent and use
five speaker IDs rather than names. Each speaker supplies ten previously held-out English support
requests: five quiet, three moderate-noise, two noisy. This produces 25/15/10 clips overall. Do not
select only successful recognitions or tune ASR, thresholds, or intent models on these clips.

Create `data/raw/voice/manifest.csv` with these columns:

```csv
recording_id,speaker_id,noise,split,consent,audio_path,reference
```

Use unique recording IDs, `quiet`/`moderate`/`noisy`, `held_out`, `yes`, an audio path relative to
the manifest, and a human-written verbatim reference. Quote CSV text containing commas. WAV,
WebM, and OGG inputs follow the same 10-MB and 30-second limits as the app. References must be
written and frozen before inspecting ASR output. Synthetic speech is suitable for engineering
smoke tests only and does not meet this held-out participant requirement.

Evaluate real audio entirely locally with the existing Vosk model:

```powershell
uv run --offline --extra api python scripts/evaluate_voice.py --manifest data/raw/voice/manifest.csv
```

Outputs are `artifacts/voice_evaluation/voice_predictions.jsonl` and `voice_metrics.json`. WER is
the total word-edit distance divided by total reference words, not the mean of clip WERs.
Normalization lowercases English words, ignores punctuation, and preserves apostrophes. Empty
recognition is retained as deletions. Invalid formats or overlong clips stop evaluation rather
than silently dropping examples. Review and preserve the full recording roster when repairing
format issues. The report exposes sample-size, speaker-count, noise-distribution, and WER <= 30%
gates separately; a small smoke sample cannot pass the full acceptance check.

Recompute the same metric table from cached predictions without ASR or model downloads:

```powershell
uv run --offline python scripts/evaluate_voice.py --predictions artifacts/voice_evaluation/voice_predictions.jsonl
```

Cached predictions have strict fields: `recording_id`, `speaker_id`, `noise`, `split`, `reference`,
`transcript`, `asr_model_version`, `duration_ms`, and `latency_ms`. Preserve these per-record results
and their hashes alongside the final report. This command measures transcription; held-out
end-to-end task success and response appropriateness still require final human ratings.
