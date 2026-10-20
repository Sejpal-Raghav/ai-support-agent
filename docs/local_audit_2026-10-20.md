# Local requirements audit — 20 October 2026

Scope: local source, tests, existing data/model artifacts, frontend build, and localhost inference.
Baseline: `docs/voice_assignment_requirements.md`. No additional requirements chat was attached;
the user confirmed this local baseline and authorized autonomous local work. GitHub access,
publishing, commits, and deployment were not performed.

**Verdict: the local demo works, but the assignment's evidence gates are not all satisfied.**
The remaining gaps require reviewed human labels, real held-out recordings, and measured model
results. A pretrained MiniLM similarity demo must not be reported as a validated fine-tuned
BERTweet/DeBERTa system.

## Requirement coverage

| Requirement | Local finding after fixes | Status |
|---|---|---|
| Accept voice input | MediaRecorder UI, cleanup and permission tests; WAV upload API exercised | Implemented; physical microphone checks in Chrome/Edge remain manual |
| Speech to text | Cached Vosk and real FFmpeg conversion successfully recognized a synthetic fixture | Integration verified; 50-clip WER gate pending |
| Display recognized speech | Frontend test holds answer generation pending and verifies visible transcript | Verified in component integration test |
| Deep learning | Cached MiniLM serves the demo; BERTweet/DeBERTa challenger code exists | Formal fine-tuning and macro/per-intent F1 gates pending |
| Appropriate response | Same-intent retrieval, reply relevance, credential/billing fallback, historical quotation, traceable evidence | Engineering checks pass; human response-quality evaluation pending |
| Display response and metadata | Response, action, intent, similarity/confidence, reasons, evidence | Verified in frontend/API tests |
| Public access | Explicitly excluded from this task | Deferred |
| Source submission | Source remains local; tracked data inventory contains fixtures/guidelines, not raw corpus/models | Remote submission explicitly deferred |
| Brief report | `docs/voice_project_report.md` describes measured behavior and missing evidence | Local report complete; final results and public URLs pending |
| Reproduce results | Frozen-result commands verify content hashes; voice evaluator supports cached predictions | Tooling implemented; final frozen evaluation bundle absent |

## Findings fixed

1. **Unsafe requests could retrieve ordinary replies.** Credential/private-data and duplicate or
   unauthorized billing requests now return an explicit safe fallback before neural retrieval.
2. **Historical replies could imply account actions, expose contact details, or retain names.**
   Added action-claim/contact/name filters and further sanitization. Replies are explicitly
   attributed to a historical case, with the demo's account-access boundary stated.
3. **An intent label alone did not establish reply relevance.** Both corpus building and runtime
   now require reply cues matching the selected intent. This is a conservative heuristic, not
   proof that every remaining suggestion is appropriate.
4. **The demo corpus had no traceable train partition.** Rebuilt it using connected conversations
   (both reference directions), chronological 70/15/15 partitions, train-only selection, exclusion
   of normalized exact matches to held-out requests, source thread IDs, and a hash manifest.
   Runtime verifies the corpus hash and every case's train membership before serving; the cached
   MiniLM revision is pinned explicitly.
   Semantic near-duplicate review remains pending. This demo partition is separate from the
   full pipeline's 12,000-thread cohort; their evaluation claims must not be mixed.
5. **Whitespace-only API requests reached inference.** They now receive validation errors.
6. **Audio was read without a bounded read and synchronous ASR blocked the async route.** Reads
   are capped at 10 MB plus one byte, uploads close in all cases, and ASR runs in a worker thread.
   FFmpeg decoding is bounded to 31 seconds so the existing duration validator rejects long clips.
   Multipart parsing may spool before route validation; this is not a total request-body limit.
7. **Microphone tracks leaked when setup failed or permission arrived after unmount.** Added
   lifecycle guards, immediate resource ownership, and regression tests. Already-aborted HTTP
   requests now propagate cancellation correctly. The header also shows startup failure clearly.
8. **The end-to-end test was `assert True`.** Replaced it with audio-to-transcript-to-grounded
   API integration, using explicit model doubles and real WAV handling.
9. **Voice WER tooling was absent.** Added real-audio and cached evaluation, word-edit scoring,
   noise breakdowns, and explicit sample-size/speaker/noise/WER gates. Failed recognition counts
   as deletions. No participant clips or evaluation scores were invented.
10. **Frozen reproduction could skip hashes, follow escaping paths, or overwrite colliding
    basenames.** Reproduction always verifies hashes; directory confinement and preflight
    collision checks now reject those cases.
11. **Transformer reproducibility/acceptance had gaps.** Seed initialization now precedes model
    construction; inference uses training normalization and recorded maximum length. Validation
    packet membership and selection/calibration request separation are checked. The challenger
    records per-intent F1 and quality-gate status, and runtime refuses unapproved model results.
    An unsupported confidence operating point now fails explicitly instead of returning a cutoff
    that could still accept confidence 1. Zero-margin model selection is also fixed.
12. **Thread reconstruction rescanned every edge for every component; cyclic components shared
    an ID.** Edges are grouped once and rootless components hash their member IDs. Near-duplicate
    auditing now uses bounded blocks instead of a full quadratic similarity allocation. Cosine
    scores are clipped to their mathematical range before entering strict response schemas.

## Verification evidence

- Python: **93 tests passed**, including new safety, API, voice metrics, reproduction, thread,
  and integration regressions. One upstream Starlette/AnyIO deprecation warning remains.
- Frontend: **6 tests passed**, including transcript-before-answer and microphone resource cleanup.
- Production frontend build: passed.
- Ruff and strict mypy: run against `src`, `scripts`, and `tests` / configured Python package.
- Actual localhost health: HTTP 200 with ASR, classifier, and retrieval ready using cached models.
- Actual speech smoke: HTTP 200, 4,019 ms synthetic clip, transcript
  `where is my package my delivery is late`. Synthetic smoke is not held-out WER evidence.
- Local generated evidence: `artifacts/local_smoke.json`, `artifacts/local_voice_smoke.json`,
  `artifacts/demo_cases.manifest.json`. These remain ignored.
- Remote dependency audits, clean internet downloads, GPU training, Docker builds, physical
  microphone access, cross-browser checks, and public deployment were not verified in this scope.

## Remaining local evidence gates

1. Complete the full cohort/cluster pipeline and human review; `configs/taxonomy.yaml` is still
   uninitialized. AmazonHelp and six heuristic demo intents do not substitute for that review.
2. Complete 100 selection, 100 calibration, **200 policy**, and 200 untouched golden labels;
   perform consistency review and independent reply ratings. The former status document said
   100 policy labels, conflicting with the actual configuration and annotation guidelines.
3. Train and compare BERTweet and DeBERTa using validation data only. Meet macro-F1 >= 0.80 and
   every supported intent F1 >= 0.65, then calibrate the validated operating point.
4. Collect 50 consenting-speaker held-out recordings (five speakers; 25 quiet, 15 moderate,
   10 noisy), freeze references before evaluation, and demonstrate corpus WER <= 0.30.
5. Measure response appropriateness/safety, coverage, task success and latency on the final
   held-out inputs. Freeze actual per-example records and regenerate the final report tables.

The remaining evidence cannot be replaced by guessed labels, synthetic speakers, or fabricated
metrics. See `docs/voice_evaluation.md` for the prepared recording/evaluation workflow.
