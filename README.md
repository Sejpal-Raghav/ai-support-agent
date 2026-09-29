# Voice-Enabled Support Chatbot

A localhost-ready voice support chatbot grounded in real AmazonHelp conversations from Kaggle's
Customer Support on Twitter dataset. It accepts microphone or typed input, displays the recognized
speech, predicts one of six support intents with a neural sentence encoder, retrieves a similar
historical resolution, and shows the evidence case ID behind its response.

## What works locally now

- Browser recording in Chrome/Edge, with server-side FFmpeg conversion and local Vosk transcription.
- Neural intent classification using `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieval over 900 sanitized, problem-solving customer-to-AmazonHelp reply pairs from the local
  Twitter subset.
- Safe fallback for low-confidence, weak-evidence, or credential-related requests.
- Resilient React/Vite interface with listening level, timer, typed fallback, retry, transcript,
  response, decision reason, and evidence-detail states.
- Docker definition for a Hugging Face Docker Space.

The repository also contains an executable BERTweet-versus-DeBERTa challenger. It trains both
models under the same fixed budget, selects on validation macro-F1 (warm p95 breaks differences
below `0.005`), fits temperature scaling on a separate validation packet, and selects a confidence
cutoff targeting at least 90% accepted precision. MiniLM powers the current demo until the required
human-reviewed labels are completed; the runtime automatically loads the selected transformer
artifact when `artifacts/models/transformers/selection.json` exists.

## Run the localhost demo

Requirements: Python 3.11, `uv`, Node.js 20+, and Chrome or Edge.

```powershell
uv sync --frozen --extra data --extra ml --extra api --extra dev
uv run --extra api python scripts/download_vosk_model.py
cd frontend
npm ci
npm run build
cd ..
uv run --extra ml --extra api python scripts/run_local.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The first launch downloads and caches the
MiniLM model. Allow microphone access, press **Talk**, speak for less than 30 seconds, and press
**Stop**. The browser recording is transcribed locally by Vosk; typed input remains available.

Startup can take roughly one minute on CPU while Vosk loads. The header changes from **Starting**
to **Ready** only after the ASR, classifier, and retrieval components are loaded.

### Build the local evidence corpus from scratch

Place Kaggle's `twcs.csv` at `data/raw/twcs.csv`, then run:

```powershell
uv run --extra data python scripts/ingest_local_demo.py
uv run --extra data python scripts/build_local_demo.py
```

The first command deterministically ingests the leading 750,000 source rows. The second extracts
up to 150 direct AmazonHelp customer/reply pairs for each supported intent. Raw data and generated
artifacts are intentionally ignored by Git.

To download the Kaggle dataset instead of copying it manually:

```powershell
uv run --extra data python scripts/download_dataset.py
```

### Train the reviewed transformer challenger

Complete and validate the model-selection and calibration packets first; the command fails closed
when labels are absent, incomplete, or in the wrong partition.

```powershell
uv run --extra data python scripts/export_annotations.py --kind validation
# Human: complete the generated CSV packets using data/annotations/annotation_guidelines.md
uv run --extra data python scripts/validate_annotations.py --packet data/annotations/validation_model_selection.csv
uv run --extra ml --extra data python scripts/train_transformer_challenge.py
```

Neither model choice nor thresholds read the golden test set.

## Supported intents

| Intent | Typical request |
|---|---|
| `delivery_tracking` | Late, missing, or incorrectly delivered package |
| `return_refund` | Return or refund request |
| `damaged_or_wrong_item` | Damaged, defective, or incorrect product |
| `account_or_payment` | Account access, charges, or payment problems |
| `order_change_or_cancel` | Cancel or modify an order/address |
| `product_or_stock` | Availability or product information |

## API

- `GET /api/health` reports ASR, classifier, and retrieval readiness.
- `POST /api/transcribe` accepts WebM/OGG/WAV audio up to 10 MB and 30 seconds. `run_local.py`
  automatically configures the pinned downloaded Vosk model; deployments may set `VOSK_MODEL_PATH`.
- `POST /api/respond` accepts `{"text":"...","session_id":"optional"}` and returns intent,
  confidence, action, response, reasons, evidence IDs, model versions, and latency.

## Verification

```powershell
uv run --extra api pytest -q
uv run ruff check src scripts tests
uv run mypy
cd frontend
npm test
npm run build
npm audit --omit=dev
```

Current local verification: 64 Python tests and 3 frontend behavior tests pass; Ruff and strict
mypy pass; the frontend production build succeeds and its dependency audit reports zero
vulnerabilities.

## Safety and privacy boundaries

- The chatbot never posts messages or performs account actions.
- Historical handles, direct customer names, URLs, and agent signatures are removed from replies.
- Praise-only and social exchanges are excluded, and historical replies must contain a support
  action before they can enter the evidence corpus.
- Credential-related historical replies are never returned automatically.
- Uploaded audio is converted in memory and is not retained or logged.
- Low-confidence or weak-evidence messages produce a visible fallback.

## Remaining human and deployment work

Code cannot truthfully manufacture the required evidence. A human must finish the reviewed
taxonomy/annotation packets and record the held-out five-speaker audio set before training and
freezing accuracy, WER, coverage, latency, and task-success results. See
[docs/local_completion_status.md](docs/local_completion_status.md) for exact gates.

## Deployment (deferred)

`Dockerfile` builds the React assets and serves them from FastAPI on port `7860`. The Hugging Face
Space metadata and deployment notes are in `deployment/huggingface/README.md`. A public deployment
still requires a Hugging Face account, the Vosk model directory, and generated demo/model artifacts.

See [docs/voice_assignment_requirements.md](docs/voice_assignment_requirements.md) for the
requirement-to-evidence matrix. Reported evaluation results must come from held-out human labels
and voice recordings; this README does not invent metrics that have not been measured.
