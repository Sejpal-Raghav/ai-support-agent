# Render Free compatibility audit — 21 October 2026

## Change and requirement coverage

The previous Docker image loaded PyTorch MiniLM and a large Vosk graph and exceeded Render's
free 512 MiB memory limit. The default image now contains a compact inference-only environment:

- **Neural intent classification:** the same pretrained six-layer, 384-dimensional MiniLM
  transformer, pinned to revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, as an official
  8-bit ONNX export. Rust tokenization, attention-masked mean pooling, and L2 normalization
  preserve the sentence embedding pipeline. No PyTorch import is required in the deployed app.
- **Responses:** the existing 641 verified train cases, seed-based intent matching, same-intent
  retrieval, privacy/billing filters, evidence integrity checks, and safe fallbacks are retained.
- **Speech:** actual server-side FFmpeg conversion and `vosk-model-small-en-us-0.15` replace the
  larger speech model. The API identifies the actual selected model instead of reporting the old
  default name. English speech, typed input, transcript, intent, and response remain available.
- **Bounded processing:** one shared inference slot prevents simultaneous speech/text calls from
  allocating extra decoders. Concurrent requests receive a retryable 503. Health checks remain
  available. A decoder error releases the slot.
- **Deployment:** the root Dockerfile and native build script install a separate pinned inference
  dependency set. The Docker model download cache does not enter the image. Immutable model
  checksums are verified before installation; local models work without runtime network access.

No paid service, API key, Ollama, or hosted LLM is introduced. The original PyTorch local profile
and separate model-training tools remain available in the repository. The compact image uses
the existing evidence secrets; no private evidence, weights, recordings, or generated results are
committed to Git.

## Measured engineering checks

The first complete Linux test constrained the service to **512 MiB RAM and 0.1 CPU**, disabled
swap and outbound networking, mounted only the verified private bundle read-only, and used a
Render-style port. It passed static assets, readiness, seven typed requests (including privacy and
billing fallbacks), real WAV-to-transcript processing, and transcript-to-response processing.

| Check | Measured result |
|---|---|
| Initial constrained cold startup | 190.73 seconds |
| Initial constrained memory peak | 285,413,376 bytes (272.2 MiB) |
| Synthetic WAV duration | 4.019 seconds |
| Recognized text | `where is my package my delivery is late` |
| ASR decoding latency at 0.1 CPU | 17.595 seconds |
| Docker readiness | Healthy; all three inference readiness flags true |
| Quantized versus original encoder | 43/43 intent choices agree on fixed smoke probes |
| Embedding cosine to original | Minimum 0.9805; mean 0.9921 |
| Python regression suite | 133 passed; one existing Starlette/AnyIO deprecation warning |
| Frontend | 9 tests passed; production build passed |
| Static checks | Ruff passed; Mypy passed across 61 source/script files |

The **final release image**,
`sha256:45abd0735bed2323e9c3d4de50996ccca23fb311af1fa85ae2ddfda00c281747`,
passed the same limits using a **28-second Opus/WebM** recording: ready after **179.36 seconds**,
peak **303,239,168 bytes (289.2 MiB)**, ASR decoding **34.201 seconds**, nonempty recognized
speech, and a successful follow-up chatbot response. Docker health was healthy and the actual
speech-model version was `vosk-model-small-en-us-0.15`.

The local browser preview also passed typed submission and displayed **Delivery tracking**
with the support response. Its container used 512 MiB, no swap, and the legacy Secret File pair
with the old built-in Vosk path override, verifying automatic migration. The local preview had
two CPUs for a responsive recording/demo; the isolated integration tests above used 0.1 CPU.
The runtime environment was checked: ONNX Runtime, tokenizers, and Vosk are installed; PyTorch,
Sentence Transformers, pandas, and scikit-learn are absent.

Single-text quantization is intentional: grouping unrelated texts into a larger batch changed
the minimum embedding compatibility score below the preselected 0.98 gate. That optimization
was rejected and the single-text implementation reverified. Comparison probes are engineering
checks, not held-out task accuracy, calibration, or speech word-error-rate measurements.

The client transcription deadline was extended from 45 seconds to a bounded three minutes.
Tests verify that a slower valid response can finish and that a hung request still times out.
The 30-second recording limit remains unchanged. Cold startup and longer speech requests are
slower at 0.1 CPU; short questions are preferable for a demo.

## Reproduction and artifacts

See [the current Render instructions](../deployment/render/README.md). The container verifier's
defaults are the actual free-tier limits: 512 MiB, 0.1 CPU, no swap, no external network, and a
ten-minute startup deadline. It supports WAV, WebM/Opus, and OGG upload MIME types.

Ignored local evidence includes `artifacts/render-free-build-v2.log`,
`artifacts/render-free-check-512m.json`, `artifacts/compact-embedding-parity.json`,
`artifacts/render-free-release-check-512m.json`, and `artifacts/render-free-browser-proof.png`.
Each container report records the tested image, real cgroup peak,
HTTP outcomes, transcript, and final Docker health state. Generated fixtures contain synthetic
speech, and are not participant recordings.

Both the new ASCII evidence bundle and the previous matching JSONL/manifest pair are supported.
Startup migrates the old built-in Vosk path when it is absent from the compact image. The local
preview exercises that migration with the previous pair.

## Limits of the evidence

The smaller Vosk model trades recognition quality for resource use. No new human-speech WER or
held-out response accuracy is claimed. Real microphone permissions remain browser/device
dependent; API audio integration cannot establish that a physical microphone has permission.
The resource measurements are Linux Docker measurements, not Render's own metrics.

## Public deployment verification

After publishing commit `51c8a18`, the public HTTPS service at
<https://ai-support-agent-5e43.onrender.com/> returned all three readiness flags true and a real
`delivery_tracking` response from `minilm-onnx-semantic-c9745ed1d9f2`.

A real multipart WebM upload was then sent to the public transcription endpoint. The service
recognized **“where is my package my delivery is late”**, reported
`vosk-model-small-en-us-0.15`, and returned a Delivery tracking response for that transcript.
The two HTTPS requests completed in **7.78 seconds**. The public browser also showed Ready,
an enabled recording control, and a working Send question action with visible intent and response.

Evidence is saved locally in `artifacts/render-public-free-verification.json`,
`artifacts/render-public-free-voice.json`, and `artifacts/render-public-free-proof.png`.
No Render plan or payment settings were changed. This confirms deployed audio/API integration;
physical microphone permission remains a browser/device check.

The full GitHub check also exposed five pre-existing typing/stub errors in training/evaluation
scripts. They were corrected with typed ndarray/record annotations, Pydantic dictionary validation
for CSV values, and the missing joblib stub configuration. The exact CI commands now pass locally:
`ruff check .`, `mypy src scripts` (61 files), and all 133 Python tests.
