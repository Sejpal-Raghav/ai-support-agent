# Deploy on Render Free (512 MiB)

The default Docker image uses **8-bit ONNX MiniLM and small English Vosk**. It keeps server-side
speech recognition, neural intent classification, safe train-only retrieval, and the same
transcript/intent/response UI. PyTorch and research packages are excluded from the deployed
environment. No API key, Ollama, paid compute plan, or paid external service is required.

See [the free-tier verification report](../../docs/render_free_audit_2026-10-21.md) for measured
limits. The previous PyTorch image's 1 GiB requirement is superseded for this compact image.

## Existing service: fastest path

1. **Keep the Free compute plan (512 MiB).** Deploy the latest repaired commit using the root
   Dockerfile, repository root directory, and default Docker CMD. Clear a custom Docker Command
   override if one was added. Docker services need no separate build/start command.
2. **Keep the existing matching evidence Secret Files.** The last supplied Render log already
   confirmed they passed validation. Both the original JSONL/manifest pair and the newer single
   bundle work with this image; changing to the bundle is optional.
3. Set Health Check Path to **`/api/ready`**. Render provides `PORT`; the server binds to
   `0.0.0.0:$PORT`.
4. Remove obsolete custom model paths and `TRANSFORMER_SELECTION_PATH` if present. The image
   selects ONNX and the small speech model automatically. An override pointing at the previous
   built-in large Vosk directory is migrated if that directory is absent.
5. Wait for cold startup. At 0.1 CPU the first measured run took approximately three minutes.
   Then `/api/ready` must return 200 with all three readiness flags true.
6. Over the public HTTPS URL, submit a typed question and check its response and intent. Then
   grant microphone permission, record a short English question, and check its transcript,
   response, and intent. Short questions give the quickest demo on the free CPU.

The client allows up to three minutes for longer recordings to finish transcription. The UI
still records at most 30 seconds. Inference uses one shared slot so simultaneous speech/text
requests cannot allocate multiple decoders; a concurrent request receives a retryable 503.
Health checks remain available while a request is processing.

## Private inference evidence

Preferred single-file transport:

```powershell
uv run --offline python scripts/package_render_evidence.py
```

Add a Render Secret File named **`support_agent_bundle.b64`** and paste the complete contents of
`artifacts/render/support_agent_bundle.b64`. The current 641-case bundle is 88,513 bytes. Keep
it out of Git. It contains the exact corpus and compact manifest, preserving the hash, record
count, unique IDs, and train-only provenance. Line wrapping and whitespace from pasting are
accepted; changed content, corruption, or mismatched evidence are rejected.

The previous pair remains supported: `demo_cases.jsonl` and `demo_cases.manifest.json`, mounted
under `/etc/secrets/`. Use the compact matching files from `artifacts/render/`. An explicit
`DEMO_CASES_PATH` takes priority over a bundle. The legacy loader permits equivalent UTF-8 BOM,
CRLF/LF, and final-newline changes, and rejects changes to the actual JSON content.

Render's combined Secret File limit is 1 MB. See
[Render Secret Files documentation](https://render.com/docs/configure-environment-variables#secret-files).
Never upload raw data, credentials, recordings, the virtual environment, or the whole artifacts
directory. Regenerate the bundle whenever the corpus changes.

## Build and verify locally

```powershell
docker build --progress=plain -t support-agent:render-free .
uv run --offline --extra api python scripts/verify_render_container.py
```

The verifier defaults to **512 MiB, 0.1 CPU, no swap, no outbound networking**, and a ten-minute
cold-start deadline. It mounts the private bundle read-only, checks readiness and static assets,
submits seven typed requests including safety fallbacks, and waits for Docker health to be healthy.
To exercise actual FFmpeg/Vosk as well, supply an existing synthetic or authorized audio fixture:

```powershell
uv run --offline --extra api python scripts/verify_render_container.py --audio-fixture artifacts/local_smoke.wav --expected-transcript "where is my package my delivery is late"
```

WAV, OGG, and WebM fixtures use their appropriate upload MIME type. Results/logs stay under ignored
`artifacts/`; the verifier removes only its own temporary container. Synthetic smoke audio proves
integration, not human speech accuracy or physical microphone permissions.

The build downloads the pinned ONNX model/tokenizer, verifies their SHA-256 values, verifies the
small Vosk archive, and executes bundled FFmpeg. Model downloads are cached between builds.
Runtime uses local model files and needs no model download.

## Optional native Python service

| Setting | Value |
|---|---|
| Runtime | Python 3.11 |
| Root directory | Repository root |
| Build Command | `bash scripts/build_render.sh` |
| Start Command | `python scripts/run_render.py` |
| Health Check Path | `/api/ready` |
| `PYTHON_VERSION` | `3.11.11` |

Node/npm must be available during build. The script installs the compact dependencies and project
without pulling in research dependencies. Startup detects the prepared ONNX model automatically.
The full constrained Linux integration benchmark uses Docker.

## Reading failures

- Missing deployment files: check the exact Secret File names and remove stale path overrides.
- Invalid bundle or hash mismatch: replace the bundle, or both legacy files, from one verified export.
- A log still demands 1 GiB: confirm the latest compact commit and Docker CMD are deployed.
- Readiness stays false on the public site: Render may still be serving the previous deployment.
  Confirm the deployed commit and latest deploy logs.
- Speech is slow: keep questions short; cold startup and CPU throttling are expected on Free.

The public deployment is complete only after the new commit becomes healthy and both typed and
microphone requests pass over HTTPS.
