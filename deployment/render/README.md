# Repair the deployment on Render

The original Dockerfile built the frontend without installing ML dependencies, providing models,
or configuring evidence paths. It could serve a working page with all inference readiness flags false.
Changing browser microphone permissions cannot fix that server state.

## Existing Docker service

Keep the existing service if it uses Docker. The updated Dockerfile installs CPU inference
dependencies, caches MiniLM/Vosk at build time, and starts with the same validated entrypoint as
the native service. Add the two Secret Files described below, remove any custom Docker Command
override so the updated Dockerfile CMD is used, and rebuild the latest source. Render's `PORT`
overrides the default port. No separate build/start command is needed for Docker.

## Native Python service

Use a **Python web service**, Python **3.11**, with repository root as the service root:

| Setting | Value |
|---|---|
| Build Command | `bash scripts/build_render.sh` |
| Start Command | `python scripts/run_render.py` |
| Health Check Path | `/api/ready` |
| `PYTHON_VERSION` environment variable | `3.11.11` |

The root Dockerfile is not used by a native Python service. Ensure Node/npm are available in the build environment;
the build intentionally stops if either frontend building or model preparation fails.

## Required inference files (both runtimes)

In Environment → Secret Files, add these **two existing local inference files** using their exact
filenames (select/paste their contents; do not commit them to Git):

- Prepared local `artifacts/render/demo_cases.jsonl` → `demo_cases.jsonl`
- Prepared local `artifacts/render/demo_cases.manifest.json` → `demo_cases.manifest.json`

The prepared manifest contains only the train memberships needed for inference and retains the
original manifest's hash. The original matching `artifacts/demo_cases.*` files also work if these
compact deployment copies have not been prepared.

They are available to the running service under `/etc/secrets/`. The matching manifest is mandatory
and must come from the same corpus build. Do not include raw Twitter data, participant recordings,
credentials, the local virtual environment, or the full artifacts folder.

The build caches pinned MiniLM weights and downloads/checksums Vosk. Startup configures model,
evidence, and static paths automatically and binds to `0.0.0.0:$PORT`. No API key or Ollama is needed.
If you have old path variables pointing to `/app` or a Windows directory, remove those overrides
or set them to the actual native-service paths; explicit variables take precedence.

After rebuilding, `/api/health` must report ASR, classifier and retrieval readiness as true.
`/api/ready` returns HTTP 503 until all three are available. Test both typed input and a microphone
recording over the public HTTPS URL. Monitor startup peak RAM; the models were tested locally,
not against Render's smallest instance memory limit.
The existing Windows process peaked at approximately 1 GB working memory; this is an indication
that a 512 MB instance may be insufficient, not a Linux memory benchmark. Start with sufficient
headroom (for example 2 GB) and verify actual service usage.

The scripts and UI can be checked locally, but this deployment is not repaired until the Render
settings/secret files are supplied and the new build passes public inference checks.
