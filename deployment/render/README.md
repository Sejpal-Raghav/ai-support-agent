# Deploy the verified application on Render

The Docker image was built and exercised locally on Linux, including real MiniLM and Vosk
inference with outbound networking disabled. See [the audit](../../docs/render_audit_2026-10-21.md)
for measurements and the limits of that verification. A local commit does not update Render.

## Existing Docker service: fastest path

1. Use an instance with **at least 1 GiB RAM; 2 GiB is recommended**. The no-swap Linux test
   peaked at approximately 776 MiB. A 512 MiB container was killed while loading the models;
   startup now explains the memory requirement before loading them. No plan was changed by
   the local repair.
2. Publish the repaired local commit to the repository branch connected to Render when ready.
   Keep the service runtime **Docker**, root directory at the repository root, and Dockerfile
   path `./Dockerfile`. Remove a custom **Docker Command** override so the image's CMD runs.
   Docker services do not need separate Build Command or Start Command settings.
3. Under **Environment -> Secret Files**, add **`support_agent_bundle.b64`**. Paste the complete
   contents of the already generated local file **`artifacts/render/support_agent_bundle.b64`**.
   This is one ASCII file containing both verified inference files, currently 88,513 bytes.
   Keep it out of Git. No model weights or raw dataset need to be uploaded as secrets.
4. Remove stale `DEMO_CASES_PATH`, `MINILM_MODEL_PATH`, `VOSK_MODEL_PATH`, `STATIC_DIR`,
   `HF_HOME`, or `INFERENCE_BUNDLE_PATH` overrides unless intentionally using custom paths.
   An explicit `DEMO_CASES_PATH` takes priority over the new bundle. In particular, a stale
   Windows path or `/app/artifacts/demo_cases.jsonl` would bypass automatic bundle selection.
5. Set **Health Check Path** to **`/api/ready`**. Save the settings and rebuild the repaired commit.
   Render supplies `PORT`; the application binds to `0.0.0.0:$PORT`. No API key or Ollama is used.
6. Verify the public HTTPS URL: `/api/ready` must return HTTP 200 with `asr_ready`,
   `classifier_ready`, and `retrieval_ready` all `true`. Submit a typed delivery question and
   check its response and intent. Then record a short English question, grant microphone
   permission, and check the transcript, intent, and response.

The build installs CPU inference dependencies, saves the pinned MiniLM model to an explicit local
directory, downloads and verifies Vosk, and runs the bundled FFmpeg executable. Runtime does not
download models. Increased download timeouts and cached dependency/model layers address the
package-download failures reproduced during the audit.

Render limits the **combined** size of a service's secret files to 1 MB. The new bundle preserves
the exact corpus bytes, manifest hash, unique IDs, and train-only provenance. Whitespace changes
from pasting the ASCII bundle do not change its contents. Corruption or mismatched evidence still
fails verification. See [Render Secret Files documentation](https://render.com/docs/configure-environment-variables#secret-files).

## Regenerate and verify locally

Run from the repository root after building the local inference corpus:

```powershell
uv run --offline python scripts/package_render_evidence.py
docker build --progress=plain -t support-agent:render-verified .
uv run --offline --extra api python scripts/verify_render_container.py --memory 2g
```

The container check mounts only the bundle at `/etc/secrets/support_agent_bundle.b64`, sets a
Render-style port, disables outbound networking and swap, checks readiness and static assets,
submits seven typed requests including safety fallbacks, and waits for Docker health to be healthy.
Its JSON result and logs stay under ignored `artifacts/`. To exercise FFmpeg and Vosk as well, pass
an existing short WAV file; synthetic smoke audio is not a speech-accuracy evaluation:

```powershell
uv run --offline --extra api python scripts/verify_render_container.py --memory 2g --audio-fixture artifacts/local_smoke.wav --expected-transcript "where is my package my delivery is late"
```

The verifier removes only the temporary container it creates. The private bundle and generated
models remain local and ignored. Regenerate the bundle whenever the corpus changes; never edit
its encoded contents manually.

## Existing two-file configuration

The previous matching pair remains supported: `demo_cases.jsonl` and
`demo_cases.manifest.json` under `/etc/secrets/`. Use the compact files from `artifacts/render/`,
exported from the same corpus build. A present bundle is preferred unless `DEMO_CASES_PATH` is
explicitly set. The pair was also tested against the repaired Docker image.

The legacy loader permits equivalent UTF-8 BOM, CRLF/LF, and final-newline changes. Changes to
JSON values or missing records still fail the manifest hash. The single bundle avoids this
two-file copying problem entirely.

## Optional native Python service

Keep the Docker service for the repair above. If choosing a separate native service, use these
settings, provide the same bundle, and meet the same memory requirement:

| Setting | Value |
|---|---|
| Runtime | Python 3.11 |
| Root directory | Repository root |
| Build Command | `bash scripts/build_render.sh` |
| Start Command | `python scripts/run_render.py` |
| Health Check Path | `/api/ready` |
| `PYTHON_VERSION` | `3.11.11` |

Node/npm must be available to build the frontend. The build stops if frontend building, model
preparation, or the FFmpeg check fails. The native entrypoint is covered by unit tests; the full
Linux integration benchmark used Docker.

## Reading failures

- Package download timeout during build: retry the repaired build; its HTTP timeout is longer.
- Missing deployment files: check the exact Secret File name and remove stale path overrides.
- Invalid inference bundle: paste the complete exported `.b64` file again.
- Corpus hash mismatch with the legacy pair: replace both files from the same verified export.
- Memory limit below 1 GiB: select adequate instance memory before retrying.
- Public health still shows all readiness flags false: the public service may still be serving
  an earlier successful deployment. Confirm the deployed commit and inspect the latest deploy
  logs. Render keeps an existing healthy deployment when a replacement fails; see
  [Render deployment behavior](https://render.com/docs/deploys).

The local repair does not establish that a new deployment is live. Public inference and a real
browser microphone recording still need to pass after the repaired commit and settings are deployed.
