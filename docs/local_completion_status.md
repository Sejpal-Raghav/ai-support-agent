# Local Completion Status

## Implemented and verified

- Browser microphone recording with level meter, 30-second timer, cancel, cleanup, and typed fallback.
- In-memory FFmpeg conversion and Vosk transcription with typed errors for silence, format, size,
  duration, conversion, and model failures.
- Neural MiniLM intent/retrieval localhost demo over 900 sanitized AmazonHelp cases.
- Train-only evidence IDs, same-intent retrieval, safety sanitization, and fallback reasons.
- BERTweet/DeBERTa fixed-budget training, early stopping, validation-only comparison, temperature
  scaling, selective confidence thresholding, and automatic runtime loading of the winner.
- React states for startup, permission, recording, transcription, response, retry, and evidence.
- 64 Python tests, 3 frontend behavior tests, Ruff, strict mypy, production build, and npm audit.

## Human evidence still required

These tasks cannot be automated without invalidating the assignment evidence:

1. Review the cluster-derived 6–10 intent taxonomy.
2. Hand-label 100 model-selection, 100 calibration, 100 policy, and 200 untouched golden examples
   using `data/annotations/annotation_guidelines.md`.
3. Blindly re-rate the specified consistency subsample and independently rate at least 50 judged
   replies for Cohen's kappa.
4. Record 50 held-out clips from five consenting speakers: 25 quiet, 15 moderate-noise, 10 noisy.
5. Run the transformer challenger, freeze evaluation artifacts, and replace report placeholders
   only with measured macro-F1, WER, precision, coverage, task success, latency, and intervals.

## Local acceptance check

```powershell
uv run ruff check src scripts tests
uv run mypy
uv run --extra api python -m pytest -q
cd frontend
npm test
npm run build
npm audit --omit=dev
cd ..
uv run --extra ml --extra api python scripts/run_local.py
```

At `http://127.0.0.1:7860`, the header must show **Ready**. A typed support request must display the
request, answer or fallback, and response details. A microphone request must display recognized
speech before its answer. Public deployment is intentionally outside this local checkpoint.
