# Render deployment audit and repair — 21 October 2026

**Historical PyTorch image audit.** The default deployment was subsequently replaced with a
compact ONNX + small Vosk runtime for Render Free. See
[the superseding free-tier audit](render_free_audit_2026-10-21.md). The 1 GiB requirement and
512 MiB failure below describe the previous PyTorch stack.

## Result and scope

The repaired Docker image builds and runs the complete application on Linux. Real MiniLM intent
classification/retrieval, Vosk transcription, FFmpeg conversion, static assets, safety fallbacks,
and browser typed submission passed. Model inference worked with outbound networking disabled.
The patch is intended for a local commit only; no push, Render deploy, paid plan change, or secret
upload was performed.

This is a build/startup and application integration audit, not a hosted deployment certificate or
a new model-quality evaluation. The newest authenticated Render logs/settings were unavailable.
The previously supplied missing-file and corpus-hash exceptions were investigated alongside a
fresh local Linux build. The public health endpoint still exposed the older degraded application
when checked during the audit.

## Findings and changes

| Finding | Evidence and repair |
|---|---|
| Large dependency downloads could fail a build | The baseline Docker build failed downloading CPU Torch with uv's default 30-second timeout. A subsequent attempt also timed out on numpy. The image now uses a 300-second uv timeout, limited parallel downloads, and a build cache for dependency downloads. |
| Private inference files were required at startup but easy to paste inconsistently | Added a deterministic ZIP/base64 exporter containing the corpus and compact manifest as one ASCII Secret File. Startup checks the hash, record count, unique IDs, and train membership before loading ML. Existing matching two-file installations remain compatible. |
| Runtime model availability depended on Hugging Face cache layout | Build saves the pinned MiniLM model to `artifacts/models/minilm`; runtime loads that explicit directory offline. Model preparation also executes bundled FFmpeg, detecting binary problems during build. |
| Startup embedding batches increased peak RAM | Reduced evidence and seed embedding batches to four and limited CPU numerical-library threads. |
| 512 MiB was insufficient without swap | A real 512 MiB no-swap run was killed with exit 137 and `OOMKilled=true`. The final entrypoint reads cgroup limits and fails before model load with an actionable requirement for at least 1 GiB, recommending 2 GiB. The final 512 MiB negative test exited 1 with `OOMKilled=false`. |
| An HTML page did not prove inference was ready | Added a Docker health check against `/api/ready`. The integration verifier requires all three readiness flags and Docker health to be healthy, and exercises real HTTP requests. |
| Code edits unnecessarily invalidated expensive dependency/model layers | Moved dependency installation and model preparation before copying application source. A final build reused those layers successfully. |

The fixes preserve evidence and safety validation. Generated evidence, model weights, recordings,
build logs, and caches are excluded from the commit. No raw dataset is bundled in the image.

## Linux verification

The complete image was built with Docker Desktop's Linux engine using the repository Dockerfile.
The tested service had two CPU limits, a Render-style `PORT=10000`, read-only inference bundle
mount, no external network, and **no swap**. Each successful run made seven typed requests,
transcribed the synthetic WAV fixture through the actual HTTP endpoint, and submitted the
transcript for a response.

| Container memory limit | Result | Time until ready | Measured peak memory |
|---|---|---:|---:|
| 2 GiB | Passed; Docker healthy | 25.48 seconds | 813,359,104 bytes (775.7 MiB) |
| 1 GiB | Passed; Docker healthy | 24.90 seconds | 788,869,120 bytes (752.3 MiB) |
| 512 MiB, before preflight | Failed: out-of-memory kill, exit 137 | — | Limit exhausted |
| 512 MiB, final entrypoint | Expected rejection with clear memory message, exit 1; no OOM kill | — | Models were not loaded |

The exact final source image,
`sha256:8963df977b40e64b21cf63ec7695bf757ed63222c429618d5f40d3b72abf8836`,
also passed the complete 2 GiB check after the explicit-path precedence regression tests were
added: ready in 35.97 seconds, peak 795,201,536 bytes (758.4 MiB), Docker healthy, and the
expected voice transcript. The table above retains the earlier measurements from this audit.

These are short single-worker smoke tests. Startup times, concurrent-request memory, and Render's
CPU allocation can differ. The minimum tested size is 1 GiB; 2 GiB provides operational headroom.
An earlier 512 MiB test with Docker's default swap could run, which is why the release verification
explicitly disables swap rather than reporting that misleading result as a supported configuration.

The 4.019-second synthetic WAV was recognized as **“where is my package my delivery is late”**.
Transcription took approximately 3.4 seconds in these constrained Linux tests. This proves the
conversion/recognizer/response pipeline, not human-speech accuracy or physical microphone access.

Browser verification at `http://127.0.0.1:17860/` showed **Ready**, an enabled recording control,
and a working Send question action. “Where is my delayed package?” produced a historical support
response with the visible intent **Delivery tracking**. No response-details panel was added.
The preview used the legacy Secret File pair, also verifying compatibility. A real microphone
recording was not captured while the user was away.

## Release checks

- Python: **121 tests passed**. One existing Starlette/AnyIO deprecation warning remains.
- Frontend: **7 tests passed** and the production build succeeded.
- Ruff: passed across `src`, `scripts`, and `tests`.
- Mypy: passed across **37 source files**.
- Full Linux Docker build: passed.
- Offline, no-swap real-model integration at both 1 GiB and 2 GiB: passed.
- Low-memory rejection, source/evidence integrity, bundle corruption and train-only checks: passed.

Ignored local evidence includes `artifacts/render-release-build.log`,
`artifacts/render-release-check-2g.json`, `artifacts/render-final-check-1g.json`,
`artifacts/render-commit-check-2g.json`, `artifacts/render-final-low-memory.json`, and
`artifacts/render-browser-proof.png`.
The 1/2 GiB reports identify the exact tested image and retain its Docker health-check state.

## Prepared deployment handoff

The private file `artifacts/render/support_agent_bundle.b64` contains **641 verified train
cases** and their matching compact manifest, totaling **88,513 ASCII bytes**. It must be supplied
as a Render Secret File named `support_agent_bundle.b64`; it is deliberately absent from Git.

Follow [the updated deployment instructions](../deployment/render/README.md): publish the local
commit when authorized, supply the bundle, use at least 1 GiB RAM (2 GiB recommended), remove
stale path/command overrides, and set the health-check path to `/api/ready`. Finish by verifying
both typed and microphone input at the public HTTPS URL. The remote service is not changed by
this local audit.
