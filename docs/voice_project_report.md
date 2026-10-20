# Voice-enabled support chatbot: local project report

Author: Raghav Sejpal. Local checkpoint: 20 October 2026.

Deployment update, 21 October 2026: the default Render image now runs the same six-layer,
384-dimensional MiniLM neural encoder as a pinned 8-bit ONNX graph. A Rust tokenizer, masked
mean pooling, and L2 normalization reproduce the sentence embedding pipeline without loading
PyTorch. Intent seed similarity and same-intent retrieval remain unchanged. Speech uses
`vosk-model-small-en-us-0.15` and FFmpeg to fit the free 512 MiB instance. A shared inference
slot bounds simultaneous decoder/encoder allocations. The original PyTorch/large-Vosk local
profile and separate training tools remain available.

The compact encoder matched the original model's intent choices on all 43 fixed engineering
probes (mean embedding cosine 0.9921, minimum 0.9805). This is a quantization compatibility
check, not held-out task accuracy. The first Linux integration run passed at 512 MiB and 0.1 CPU,
with swap/networking disabled and approximately 272 MiB peak memory. The synthetic WAV was
transcribed correctly. See [the free-tier audit](render_free_audit_2026-10-21.md) for final evidence
and limitations. The live target is <https://ai-support-agent-5e43.onrender.com/> and source is
<https://github.com/Sejpal-Raghav/ai-support-agent>; a target URL alone does not prove a new deploy
has become healthy. The older checkpoint details below are retained as historical context.

## Dataset and scope

The project uses the locally available Kaggle Customer Support on Twitter corpus and AmazonHelp
customer-to-brand exchanges. The ingestion command reads the first 750,000 source rows. The demo
builder reconstructs connected conversation membership before chronological 70/15/15 partitioning,
excludes held-out exact request duplicates, and selects only train-side evidence. Heuristic intent,
support-action, reply-relevance, privacy, and account-claim filters produce **641** current cases:

| Intent | Train evidence cases |
|---|---:|
| Delivery tracking | 150 |
| Return/refund | 150 |
| Damaged or wrong item | 22 |
| Account/payment | 150 |
| Order change/cancel | 130 |
| Product/stock | 39 |

These are weakly labelled demonstration cases, not human-reviewed training or test labels.
Raw messages, models, audio, and generated artifacts remain local and ignored by Git.

## Architecture and methodology

The React interface records up to 30 seconds of English audio or accepts typed text. FastAPI
validates the upload; FFmpeg converts it to mono 16-kHz PCM and Vosk produces a transcript. The
transcript is displayed while the response request is running. Cached MiniLM embeddings classify
the demonstration request using intent seed similarity, then retrieve a same-intent train case.
Runtime verifies the corpus manifest and pins MiniLM revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` from the existing local cache.
Semantic similarity is labelled as similarity, rather than a calibrated probability.

Credential/private-data and risky billing requests receive a safe fallback. Retrieved replies
must pass safety and intent-relevance checks. Returned historical suggestions cite case IDs and
state that the demo cannot access accounts or perform actions. Weak confidence or weak evidence
also leads to fallback. This is a retrieval system; it does not call a hosted language model.

The separate BERTweet/DeBERTa challenger trains on train-side cluster labels, compares on the
model-selection packet, and calibrates on a distinct validation packet. Its required taxonomy,
human labels, trained weights, and measured acceptance results are not yet available. Runtime
refuses a selected transformer that lacks a passing validation quality gate.

## Measured local results

The engineering suite passes 93 Python tests and six frontend tests, and the React production
build succeeds. Local health reports all inference components ready using cached models. A real
FFmpeg/Vosk request recognized a synthetic 4.019-second fixture as
`where is my package my delivery is late`. This verifies integration only.

Validation macro/per-intent F1, held-out WER, coverage, response task success, safety/error rates,
agreement, confidence intervals, and final latency distributions are **not yet measured**. No
acceptance-level quality claims are made from unit tests or this synthetic smoke check.

## Limitations and completion

Historical tweets can be stale, conditional, generic, or inappropriate despite heuristic filters.
The six demo intents are not a reviewed taxonomy, the evidence distribution is imbalanced, and
semantic near-duplicates still need review. The application has no private account or live order
access, cannot contact support, and cannot complete transactions. Browser tests use mocked media
devices; actual microphone checks in both Chrome and Edge remain manual.

Complete reviewed labels, the transformer comparison, consenting-speaker audio evaluation,
independent reply ratings, and frozen held-out evaluation before final submission. See
`local_audit_2026-10-20.md` and `voice_evaluation.md` for the evidence checklist and commands.

Source location: local repository supplied by the user; GitHub submission deferred at the user's
request. Live URL: not deployed; deployment excluded from this checkpoint. Local application:
<http://127.0.0.1:7860>.

## Reproduction

With the existing environment, cached models, and generated corpus:

```powershell
uv run --offline --extra api pytest -q
uv run --offline ruff check src scripts tests
uv run --offline mypy
cd frontend
npm test
npm run build
cd ..
uv run --offline --extra ml --extra api python scripts/run_local.py
```

Cached final text reproduction requires a real frozen bundle first:
`uv run --offline python scripts/reproduce.py --mode cached --verify-hashes`.
It currently fails clearly because final evaluation artifacts are absent. Cached voice metric
reproduction is documented separately and does not load an ASR model.
