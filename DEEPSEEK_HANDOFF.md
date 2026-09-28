# Complete Project Handoff for DeepSeek

**Snapshot date:** 2026-09-16  
**Project:** Hiver SDE Intern take-home — auditable AI customer-support agent  
**Repository path:** `C:\Users\Raghav Sejpal\Documents\ChatGPT\chatbot`  
**Git remote:** `https://github.com/Sejpal-Raghav/ai-support-agent.git`  
**Current commit:** `6becce7 Initial implementation of AI support agent`  
**Python:** exactly 3.11  
**Package manager:** `uv`  
**Current test report from the user:** `46 passed in 16.34s`

This document is the current source of truth for continuing the project. Read it fully before
changing code. It deliberately distinguishes code that exists from experiments and evidence that
do not yet exist. Do not fabricate labels, model results, judge agreement, failure examples,
confidence intervals, selected brands, or headline numbers.

---

## 1. Assignment, verbatim requirements

The candidate is given real customer-support conversations between customers and brands on
Twitter. Pick **one brand** and build an AI support agent that can:

1. Classify each incoming customer message into a small set of intents derived from the data.
2. Draft a reply grounded in how that brand historically resolved similar issues. Grounding must
   be retrieval over real historical resolutions, not an LLM imitating the brand from general
   knowledge. Every draft must preserve evidence links to the historical cases used.
3. Decide whether the message should be auto-handled or escalated to a human, with a stated reason.
   This is selective prediction: maximize only safe automation, not raw automation volume.

Primary dataset:

- Kaggle `thoughtvector/customer-support-on-twitter`, approximately 3 million tweets, noisy,
  multi-turn, and multi-brand.
- Banking77 is optional for intent work only. We decided **not to use it** because the taxonomy
  should be derived from the selected brand's Twitter messages and Banking77 would add domain
  mismatch and a harder-to-defend evaluation story.

Required deliverables:

1. A runnable repository whose README reproduces headline results in under 15 minutes.
2. A golden evaluation set of 150–250 examples hand-labelled by the candidate, including sampling
   and annotation notes.
3. An evaluation harness with automated metrics and an LLM-as-judge reply rubric, including an
   actual statistic showing agreement between the judge and an independent human rater.
4. A report of at most six pages, or a README section, containing:
   - problem framing and what was intentionally not built;
   - results against at least a trivial baseline and a simple credible baseline;
   - top five failure modes with real examples and causal hypotheses;
   - a mandatory section titled **What is misleading about my headline number?**;
   - what would be done with one more week.
5. A plain 10–15 item decision log explaining non-obvious choices.

Reviewer checks that must be satisfied:

- No conversation/thread leakage across data splits.
- Golden labels are genuinely hand-created and not convenience-sampled from only easy cases.
- Baselines are fair, not deliberately weak.
- Reply evaluation is not circular.
- LLM judge agreement with independent human ratings is quantified with an agreement statistic.
- Headline result includes uncertainty and limitations.
- Escalation threshold is chosen on validation data only, never golden/test data.
- Reported tables reproduce quickly on a subsample.
- Major components are understandable and modifiable during a live interview.

Explicit exclusions:

- No real account actions.
- No private customer data beyond the public dataset.
- No posting replies to Twitter.
- No autonomous resolution of requests needing authentication.
- No attempt to cover every rare intent.
- Do not learn resolution behavior exclusively from unresolved or ambiguous conversations.

Required measurement areas:

- Intent accuracy/macro-F1 and per-intent F1.
- Reply relevance, factual grounding, usefulness, tone, and safety.
- Recall on cases requiring escalation.
- Precision/error rate among auto-handled cases.
- Coverage, meaning the fraction eligible for automation.
- Calibration: whether confidence predicts correctness.
- Human–LLM judge agreement as an actual agreement statistic.

Target headline shape:

> At a fixed safety target, the agent can auto-handle X% of eligible messages, with Y% acceptable
> replies and Z% recall on cases requiring escalation.

No X, Y, or Z currently exists. These must come from the frozen golden-set evaluation.

---

## 2. Current repository and data state

### Completed setup

- Git repository exists and is connected to the GitHub remote above.
- Dependencies and versions are pinned in `pyproject.toml` and `uv.lock`.
- Dataset download support was added using
  `kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")`.
- The raw dataset currently exists locally at `data/raw/twcs.csv`.
- The file is approximately 516,508,641 bytes and has the expected columns:
  `tweet_id, author_id, inbound, created_at, text, response_tweet_id,
  in_response_to_tweet_id`.
- Raw data is ignored by Git.
- The user reports all 46 component tests passing in 16.34 seconds.

### Real-data work not yet performed

At this snapshot:

- `artifacts/` contains only `.gitkeep`.
- `data/annotations/` contains only `.gitkeep` and `annotation_guidelines.md`.
- `data/frozen/` contains only `.gitkeep` and `MANIFEST_FORMAT.md`.
- No brand has been selected.
- No reconstructed-thread artifact exists.
- No 12,000-thread cohort or train/validation/test split exists.
- `configs/taxonomy.yaml` is still uninitialized.
- No cluster review has been completed.
- No validation packets or golden packet have been hand-labelled.
- No trained models exist.
- No retrieval index exists.
- No generated or baseline replies exist.
- No LLM judge results or independent human ratings exist.
- No policy threshold has been selected.
- No headline metrics, failure analysis, or frozen reproduction bundle exists.

### Important test limitation

`tests/test_end_to_end.py` is still only:

```python
def test_phase_zero_placeholder() -> None:
    assert True
```

Therefore, 46 passing component tests do **not** prove that the entire real pipeline composes or
that the assignment is complete.

---

## 3. Deliverable readiness

Strict submission readiness is currently **0 of 5**.

| Deliverable | State | Exact blocker |
|---|---|---|
| Runnable pipeline with <15-minute reproduction | Partial | Component CLIs exist, but there is no complete inference/evaluation runner and no frozen results. `scripts/reproduce.py` only verifies and prints an already-created cached result. |
| 150–250 hand-labelled golden set | Missing | Sampling and validation utilities exist, but no completed `golden_test.csv` exists. |
| Evaluation harness and judge agreement | Partial | Metric and judge modules exist, but no orchestration, real judge outputs, independent ratings, or kappa values exist. |
| Report | Partial | Correct section skeleton exists, but selected brand, results, failure examples, agreement, and headline are pending. |
| Decision log | Partial/nearly ready | Fifteen entries exist, but several need amendments based on council review and real experiments. |

---

## 4. Intended system architecture

The intended inference path is:

```text
incoming public customer context
    -> normalize and validate
    -> calibrated intent classifier
    -> train-only historical-case retrieval
    -> evidence-only draft generator
    -> deterministic draft validator
    -> risk-signal construction
    -> validation-selected selective-prediction policy
    -> auto_handle OR escalate(reason list)
    -> auditable prediction record with source case IDs
```

The system drafts only. It never posts, authenticates, takes account actions, or reads live/private
account state.

### Data path

1. Stream `twcs.csv` in chunks and validate source columns.
2. Convert rows into strict `RawMessage` records.
3. Reconstruct complete conversation components as `ReconstructedThread` objects.
4. Compute brand statistics and select one brand by deterministic EDA criteria.
5. Remove exact duplicate threads.
6. Hash-sample a fixed cohort, currently configured as 12,000 threads.
7. Chronologically split whole threads into 70% train, 15% validation, and 15% test.
8. Audit thread-ID leakage and cross-split near duplicates.
9. Derive taxonomy only from train messages.
10. Use validation only for model selection, calibration, retrieval adequacy, and policy tuning.
11. Use test only for the 200-example hand-labelled golden evaluation.

### Intent path

- Embed train customer messages with `sentence-transformers/all-MiniLM-L6-v2`.
- Reduce and cluster with UMAP/HDBSCAN.
- Export cluster examples and terms for manual review.
- Manually name/merge/drop clusters into 6–10 intent classes, including `other`.
- Train three systems:
  - majority intent baseline;
  - word/character TF-IDF balanced logistic-regression baseline;
  - MiniLM embeddings plus balanced logistic regression as the main system.
- Choose logistic-regression C on a dedicated validation subset.
- Temperature-scale main-model confidence on a disjoint validation subset.

### Retrieval path

- Index only train-side customer-to-brand resolution cases.
- A case contains prior context, incoming customer message, brand reply, intent, source IDs,
  resolution proxy, safety status, and a source URI.
- Encode the combined historical case text with the same pinned MiniLM model.
- Intended vector index: FAISS `IndexFlatIP` over normalized embeddings.
- Retrieve candidates, filter to same predicted intent, exclude the current thread, and MMR-rerank.
- Return at most three cases, preserving case IDs, tweet IDs, source thread IDs, similarity scores,
  MMR scores, rank, and index version.

### Drafting path

- Main drafter snapshot: `gpt-4.1-mini-2025-04-14`.
- Temperature: 0.0.
- Maximum output: 180 tokens.
- Prompt version: `configs/prompts/drafter_v1.txt`.
- The prompt prohibits outside knowledge, private-data requests, account-action claims, fabricated
  policy/status/timelines/links, and uncited material claims.
- Output must be JSON with `draft_reply` and `cited_case_ids`.
- If evidence is insufficient, output an empty draft and no citations.
- Deterministic validation currently checks citations plus unsupported URLs/numeric tokens and
  prohibited patterns. Semantic entailment is not yet adequately checked; this is a known gap.

### Selective-prediction path

Risk signals include:

- calibrated maximum intent probability;
- normalized intent entropy;
- top retrieval similarity;
- retrieval adequacy probability;
- retrieval gap;
- multi-turn/unresolved risk;
- severe negative sentiment keyword risk;
- hard-scope categories;
- draft validation result.

Hard-scope categories are mandatory escalation:

- privacy/credential risk;
- authentication required;
- account action required;
- money/billing risk;
- legal/safety risk;
- live status required;
- unsupported language or intent;
- insufficient evidence;
- repeated unresolved case;
- ambiguous request;
- failed draft validation.

The intended policy chooses the safest validation-only operating threshold. The configured target
is a 95% Wilson upper bound of at most 5% auto-handle error, at least 90% escalation recall, and at
least 50 validation auto-handles. If no operating point passes, the correct safe result is **zero
automation coverage**; do not relax the safety target to manufacture a positive headline.

---

## 5. Exact fixed configuration

```yaml
schema_version: "1"
seeds:
  global: 1729
  golden_sampling: 2718
  bootstrap: 314159
cohort:
  threads: 12000
  split_ratios: [0.70, 0.15, 0.15]
validation:
  model_selection_size: 100
  calibration_size: 100
  policy_size: 200
golden:
  size: 200
  edge_size: 100
  non_edge_size: 100
models:
  embedding:
    model_id: "sentence-transformers/all-MiniLM-L6-v2"
    revision: "fa97f6e"
    dimension: 384
    batch_size: 64
    device: "cpu"
  drafter:
    model: "gpt-4.1-mini-2025-04-14"
    temperature: 0.0
    max_output_tokens: 180
  judge:
    model: "gpt-4.1-2025-04-14"
    temperature: 0.0
    max_output_tokens: 350
policy:
  target_error_upper: 0.05
  minimum_escalation_recall: 0.90
  minimum_auto_count: 50
paths:
  raw_csv: "data/raw/twcs.csv"
  artifacts_dir: "artifacts"
  frozen_dir: "data/frozen"
```

Brand-selection defaults:

```yaml
seed: 1729
pilot_messages_per_brand: 5000
top_brands_for_pilot: 8
minimum_direct_reply_pairs: 8000
minimum_threads: 15000
minimum_resolution_proxies: 8000
minimum_inbound_messages: 25000
minimum_multi_turn_rate: 0.12
minimum_english_share: 0.80
minimum_pilot_clusters: 5
maximum_pilot_clusters: 15
minimum_pilot_cluster_coverage: 0.60
maximum_largest_cluster_share: 0.55
score_weights:
  resolution_proxy: 0.35
  inbound_messages: 0.20
  multi_turn_rate: 0.20
  pilot_cluster_coverage: 0.15
  pilot_cluster_concentration: 0.10
```

These are defaults, not sacred values. Any change must be justified in the decision log and must
not use golden/test performance for tuning.

---

## 6. Strict schemas already implemented

The canonical definitions are in `src/support_agent/schemas.py` and reject extra fields.

### `RawMessage`

- `tweet_id: str`
- `author_id: str`
- `inbound: bool`
- `created_at: timezone-aware datetime`
- `text: non-empty str`
- `response_tweet_ids: tuple[str, ...]`
- `in_response_to_tweet_id: str | None`

### `ReconstructedThread`

- Stable thread and brand IDs.
- Root IDs and ordered messages.
- Start/end timestamps.
- Customer/brand turn counts.
- Multi-turn flag, cycle flag, dangling-reference count, reconstruction version.

### `IntentTaxonomy`

- Exactly 6–10 reviewed intent definitions.
- Must be created from `train`.
- Must include `other`.
- Every intent includes description, inclusion/exclusion rules, examples/counterexamples, source
  clusters, and train support.
- Review-file and taxonomy content hashes are stored.

### `GoldenExample`

- Example/thread/tweet/brand identity.
- Context ending exactly at the incoming tweet.
- Intent label and ambiguity flag.
- Escalation requirement and enum reasons.
- Evidence adequacy label where applicable.
- Required and forbidden reply elements.
- Notes, stratum, two-stage inclusion probabilities, inverse-probability weight.
- One of four partitions: model selection, calibration, policy, or golden test.
- Annotator identity, guideline version, and timestamp.

### `ResolutionCase` and `RetrievedCase`

- Historical source IDs and texts.
- Intent and resolution proxy.
- Safety/filter reasons.
- Source URI.
- Similarity, MMR, rank, case/index versions.

### `PredictionRecord`

- Predicted intent and full probability map.
- Calibrated confidence.
- Retrieved evidence cases.
- Draft, citations, unsupported information, validation failures.
- Full risk-signal object.
- `auto_handle` or `escalate` plus reasons.
- Model, embedding, retrieval, prompt, generator, policy, cache, and timestamp provenance.

### Evaluation records

- Gold and predicted intent/action.
- Judge, primary-human, and independent-human scores.
- Composite safe-to-auto-handle and auto-handle-error flags.
- Metric estimate point/CI/method/sample size/weighting.
- Final trust decision is either `passes_prototype_trust_gate` or
  `not_ready_for_auto_handle`.

---

## 7. Annotation design

The primary annotator is the candidate. AI must not silently fill the labels and call them human.

Validation design currently expects 400 rows total:

- 100 `validation_model_selection` examples.
- 100 `validation_calibration` examples.
- 200 `validation_policy` examples.

Golden design:

- 200 test-only examples.
- 100 risk/edge cases and 100 non-edge cases.
- At most one selected decision point per thread.
- Preserve two-stage sampling probabilities and inverse-probability weights.
- Blind 10% self-rerating for intent and escalation consistency.

Fields the annotator must complete:

- `intent_label` and `intent_ambiguous`.
- `requires_escalation` and every applicable escalation reason.
- Required reply elements.
- Forbidden reply elements.
- Evidence adequacy when retrieved cases are supplied.
- Notes, annotator ID, guideline version `v1.0`, and timezone-aware timestamp.

For judge validation, at least 50 items must be independently blind-rated by another human who
does not see system identity or the LLM judge score. Council advice is to treat 50 as a floor and
increase it if feasible. Preserve rater provenance and an adjudication record.

---

## 8. Evaluation design and formulas

### Intent

- Design-weighted macro-F1 over final taxonomy labels using `sklearn.metrics.f1_score` with
  `average="macro"`, inverse-probability `sample_weight`, and `zero_division=0`.
- Report every per-class F1.

### Reply gate

The independent judge scores 1–4 on:

- relevance;
- grounding against retrieved evidence;
- usefulness;
- tone;
- safety.

Acceptance requires relevance, grounding, usefulness, and tone all at least 3; safety exactly 4;
and no critical violation. Code, not the LLM, derives the final acceptance flag.

Main judge snapshot: `gpt-4.1-2025-04-14`, separate from the drafter snapshot and prompt. Model
separation reduces but does not eliminate circularity; human agreement is mandatory.

### Escalation and automation

- **Coverage:** weighted fraction whose predicted action is `auto_handle`.
- **Escalation recall:** among gold escalation cases, weighted fraction predicted `escalate`.
- **Auto-handle error:** among automated examples, weighted fraction whose composite outcome is
  unsafe. Composite error must include at least wrong intent, gold escalation required, and an
  unacceptable/unsafe draft—not escalation label alone.
- **Acceptable reply rate:** among automated examples, weighted fraction passing reply acceptance.

### Calibration

Ten equal-width bins:

```text
ECE = sum_b (weight_b / total_weight) * abs(weighted_accuracy_b - weighted_confidence_b)
```

Classifier temperature must be fit on validation calibration labels only. Retrieval adequacy and
safe-probability calibration must also use validation only and must respect sampling weights or
the limitation must be clearly reported.

### Human–judge agreement

- Cohen's kappa for binary accept/reject.
- Quadratic-weighted Cohen's kappa for ordinal overall quality.
- Minimum 50 independently blind paired ratings.

### Uncertainty

- Deterministic, stratified, within-stratum nonparametric bootstrap.
- 2,000 replicates.
- Seed `314159`.
- Report percentile 95% intervals with point estimates.

---

## 9. Required baselines

### Trivial baseline

- Intent: always predict the train-majority class.
- Reply: fixed safe acknowledgement with no unsupported resolution promise.
- Escalation: always escalate.

This intentionally yields zero automation coverage, 100% escalation recall, and no automated
reply-acceptance denominator. Report N/A where mathematically undefined rather than pretending it
is zero.

### Simple credible baseline

- Intent: word and character TF-IDF plus balanced logistic regression.
- Reply: verbatim nearest-neighbor safe historical brand reply; no generation.
- Escalation: deterministic hard-risk regex rules plus one validation-selected confidence or
  similarity threshold.

All systems must use the same golden examples, sampling weights, reply judge, uncertainty
procedure, and safety definitions.

---

## 10. All 15 decisions and current council disposition

The original decision and the required amendment are both preserved below. Do not erase dissent.

1. **EDA-based brand selection rather than preselecting a brand.**  
   Original reason: avoid choosing a familiar brand whose data cannot support the system.  
   **Disposition: retain but revise.** Produce an auditable candidate ranking, every eligibility
   failure, selected score, and deterministic fallback. Plausible sanity-check candidates include
   AmazonHelp, AppleSupport, Uber_Support, and SpotifyCares, but none may be assumed before EDA.

2. **Fixed 12,000-thread cohort.**  
   Original reason: enough examples for held-out analysis while remaining locally tractable.  
   **Disposition: revise.** Keep 12,000 as the current default, but justify it with measured
   runtime, memory, class support, and learning-curve or stability evidence. Do not call it
   inherently sufficient before measurement.

3. **Whole-thread chronological 70/15/15 split.**  
   Original reason: prevent message/context leakage and reduce future-data leakage.  
   **Disposition: retain with stronger audits.** Add end-time boundary checks, user/identity
   leakage analysis where possible, exact duplicates, and cross-split near-duplicate audit. Make
   clear whether ordering uses thread start or end time and why.

4. **Train-only clustering plus human review into 6–10 intents.**  
   Original reason: data-derived taxonomy with `other` for rare/noisy cases.  
   **Disposition: retain.** Freeze reviewed mappings, cluster evidence, hashes, rules, examples,
   and counterexamples before validation/test evaluation.

5. **MiniLM embeddings plus balanced logistic regression as main classifier.**  
   Original reason: explainable, CPU-friendly, full confidence distribution.  
   **Disposition: provisional.** Retain only if real-label validation beats fair baselines and has
   acceptable calibration. Do not claim superiority from fixture tests.

6. **Majority and word/character TF-IDF logistic baselines.**  
   Original reason: measure imbalance and provide a credible non-semantic comparison.  
   **Disposition: retain.** They still must be executed end-to-end through reply and policy
   evaluation, not only trained as intent classifiers.

7. **Disjoint validation budgets: 100 selection, 100 calibration, 200 policy.**  
   Original reason: avoid contaminating model choice, confidence calibration, and threshold
   selection.  
   **Disposition: revise.** Explicitly audit that IDs are disjoint, account for stratified sample
   weights, state the statistical budget, and avoid silently reusing labels for multiple tuning
   decisions. Freeze choices before golden evaluation.

8. **Only safe resolved train cases enter retrieval.**  
   Original reason: unresolved/generic/risky examples should not teach automatic resolution.  
   **Disposition: reject the current proxy implementation until audited.** Current logic may treat
   an actionable verb in a brand reply as a resolved case without customer confirmation and may
   exploit future outcomes unavailable at inference. Hand-audit proxy precision and define what
   outcome evidence is allowed. Fail closed until sufficiently precise.

9. **Retrieve up to three same-intent cases using FAISS `IndexFlatIP` and MMR.**  
   Original reason: relevance plus diversity with evidence provenance.  
   **Disposition: revise.** The current code writes a FAISS artifact but runtime retrieval was
   observed using NumPy dot product rather than querying the persisted FAISS index. Either make
   serving actually load/query FAISS or remove the claim. Report retrieval adequacy and ablations.
   Same-intent filtering creates a classifier-to-retriever error cascade; measure it.

10. **Require evidence citations in every non-empty draft.**  
    Original reason: forbid general-knowledge brand imitation.  
    **Disposition: revise.** Citation presence proves provenance, not entailment. Add claim-level
    evidence checking or human evidence-adequacy labels, and surface usable source links. The
    current `tweet://...` style source URI may not be reviewer-accessible.

11. **Separate drafter and judge models/prompts.**  
    Original reason: reduce circular self-evaluation.  
    **Disposition: revise.** Freeze model snapshots and prompt hashes, blind system identity,
    validate against independent humans, and report judge failure modes. Separation alone does
    not establish judge validity.

12. **At least 50 independently blind human judge ratings.**  
    Original reason: report agreement beyond raw percentage.  
    **Disposition: retain as a floor, not a target ceiling.** Preserve rater identity, blinding,
    adjudication, confidence intervals, and any prevalence effects. Increase sample size if time
    permits.

13. **Hard-scope and failed-draft checks force escalation.**  
    Original reason: confidence must never override privacy, money, legal/safety, account-action,
    or unsupported-draft risk.  
    **Disposition: retain.** Audit regex false negatives and false positives on real annotations.

14. **Select automation threshold on validation only.**  
    Original reason: maximize coverage only subject to a Wilson 95% upper error bound <=5%,
    escalation recall >=90%, and minimum automatic count.  
    **Disposition: revise.** Tune against **composite auto-handle failure**, not merely
    `requires_escalation`. Support a legitimate no-feasible-threshold outcome with zero coverage.
    Do not relax the 5% bound to get a positive result. Ensure enough validation examples and
    honor sampling weights.

15. **Freeze content-addressed final artifacts for fast reproduction.**  
    Original reason: reviewer reproduction should verify hashes and avoid network/API work.  
    **Disposition: retain the design but reject it as completed.** Freeze raw-data fingerprint,
    reviewed labels, taxonomy, prompts, model/index versions, policy, judge outputs, baselines,
    tables, examples, environment, and result files. The README must reproduce tables, not merely
    print a hand-written cached JSON.

---

## 11. Council review outcome

A structured five-advisor council reviewed the code and decisions. The report is
`council-report-20260909-205228Z-qdaef9321.html`; the full transcript is
`council-transcript-20260909-205228Z-qdaef9321.md`.

Unanimous conclusion:

- The repository is **not submission-ready**.
- Do not merely fill in data/results and submit.
- Preferred strategy: targeted repair and integration, guarded by a 48-hour end-to-end
  composition gate.
- If one real classifier → retrieval → drafting → policy → baseline → judge → metrics vertical
  slice cannot work inside the gate, simplify/rebuild the incompatible path instead of preserving
  architecture for sunk-cost reasons.

Biases explicitly flagged:

- IKEA effect from having built many modules.
- Anchoring on “46 tests pass.”
- Sunk-cost bias toward the current architecture.
- Planning fallacy around annotations, API calls, and final integration.

Dissent that must remain visible:

1. For a submission gate, start with 200 real end-to-end labels rather than treating only a
   30-example pilot as sufficient. A 30-example slice is acceptable only as a diagnostic gate.
2. The 5% Wilson safety target may honestly produce zero automation coverage.
3. Resolution-proxy construction may use future outcome information unavailable at inference.
4. Validation split-ID disjointness is insufficient without auditing reuse and sampling weights.
5. Frozen infrastructure is implemented, but empirical reproducibility is not complete.

---

## 12. Known technical gaps and defects

These issues are known and should be handled before claiming completion:

1. No final inference/evaluation orchestration command exists.
2. `tests/test_end_to_end.py` is a placeholder.
3. `scripts/reproduce.py` accepts a config but discards it, verifies optional hashes, and prints an
   existing `headline_metrics.json`; it does not recompute headline tables.
4. README refers to a final evaluation runner that does not exist.
5. Report still contains `<Selected Brand>` and pending results/failures/agreement.
6. The resolution proxy can treat an actionable brand reply as resolved without confirmed outcome.
7. Persisted FAISS is not clearly used by runtime retrieval; NumPy similarity was observed.
8. Source URI is not necessarily a reviewer-usable public link.
9. Draft validation checks structured surface signals but not semantic claim entailment.
10. Policy safe-probability calibration currently uses escalation labels; it must align with
    composite auto-handle error.
11. Policy threshold tuning needs a safe zero-coverage fallback rather than failing or relaxing
    constraints.
12. Sampling weights are used in final metrics but need explicit treatment in calibration and
    threshold selection.
13. Same-intent retrieval propagates classifier mistakes.
14. Near-duplicate audit may be expensive/quadratic and is report-oriented rather than a hard
    exclusion policy.
15. Brand selection over all observed history and thread-start chronological ordering require a
    clearer temporal-leakage justification.
16. Human checkpoints, blinding, adjudication, and immutable annotation provenance need stronger
    operational enforcement.
17. Dataset redistribution/privacy/license policy and reviewer-accessible evidence links need to
    be described explicitly.

---

## 13. Implemented files and responsibilities

### Entry points

- `scripts/download_dataset.py`: download and place `twcs.csv` safely.
- `scripts/select_brand.py`: ingest, reconstruct, compute EDA, select brand, write provenance.
- `scripts/build_cohort.py`: retain selected brand, deduplicate, sample, split, audit leakage.
- `scripts/derive_taxonomy.py`: train-only embeddings/clustering, review export, taxonomy finalize.
- `scripts/generate_provisional_labels.py`: generate provisional split labels after taxonomy.
- `scripts/export_annotations.py`: export validation or golden human-label packets.
- `scripts/validate_annotations.py`: enforce annotation consistency.
- `scripts/train_models.py`: train main and baseline intent systems and temperature scaler.
- `scripts/build_retrieval_index.py`: build train-only historical cases and retrieval artifacts.
- `scripts/freeze_results.py`: freeze final result files and SHA-256 manifest.
- `scripts/reproduce.py`: verify and display frozen headline result; incomplete as full reproduction.

### Core modules

- `src/support_agent/config.py`: strict YAML configuration models.
- `src/support_agent/schemas.py`: strict data contracts.
- `src/support_agent/data/download.py`: Kaggle download discovery, validation, safe placement.
- `src/support_agent/data/ingest.py`: chunked CSV ingestion and Parquet output.
- `src/support_agent/data/threads.py`: graph-based thread reconstruction and audits.
- `src/support_agent/data/brand_selection.py`: EDA metrics, resolution proxy, eligibility/ranking.
- `src/support_agent/data/split.py`: cohort selection, chronological thread split, leakage audit.
- `src/support_agent/data/deduplicate.py`: exact removal and cross-split near-duplicate audit.
- `src/support_agent/taxonomy/cluster.py`: embedding, UMAP/HDBSCAN, terms and review packet.
- `src/support_agent/taxonomy/build.py`: validate reviews and construct immutable taxonomy.
- `src/support_agent/intent/train.py`: majority, TF-IDF, and embedding logistic systems.
- `src/support_agent/intent/calibrate.py`: temperature scaling and softmax.
- `src/support_agent/intent/predict.py`: intent prediction and logits.
- `src/support_agent/retrieval/cases.py`: historical case extraction/filtering.
- `src/support_agent/retrieval/index.py`: vector artifacts, MMR, retrieval.
- `src/support_agent/drafting/drafter.py`: evidence prompt, cached LLM draft generation.
- `src/support_agent/drafting/validate.py`: deterministic draft guardrails.
- `src/support_agent/policy/signals.py`: hard and soft risk signals.
- `src/support_agent/policy/selective.py`: isotonic models, Wilson-bound threshold, action decision.
- `src/support_agent/evaluation/annotations.py`: sampling/export/load/audit human annotations.
- `src/support_agent/evaluation/judge.py`: independent judge prompt/input/cache/normalization.
- `src/support_agent/evaluation/metrics.py`: weighted metrics, ECE, kappa, bootstrap intervals.
- `src/support_agent/provenance.py`: input/output artifact manifests.
- `src/support_agent/reproducibility.py`: immutable freezing and hash verification.
- `src/support_agent/hashing.py`: canonical and file SHA-256 utilities.

### Documentation

- `README.md`: setup and staged pipeline instructions; incomplete final-run section.
- `report.md`: required report skeleton with pending empirical fields.
- `decision_log.md`: original 15 decisions.
- `data/annotations/annotation_guidelines.md`: human annotation rules.
- `data/frozen/MANIFEST_FORMAT.md`: frozen result contract.
- `CITATIONS.md`: dataset citation and licensing note.

---

## 14. Dependency pins

Core:

- numpy 1.26.4
- pandas 2.2.3
- scipy 1.14.1
- scikit-learn 1.6.1
- pydantic 2.10.6
- pydantic-settings 2.7.1
- PyYAML 6.0.2
- typer 0.15.1
- click 8.1.8
- matplotlib 3.10.0
- seaborn 0.13.2
- joblib 1.4.2

Data extra:

- pyarrow 19.0.0
- duckdb 1.2.0
- kaggle 1.6.17
- kagglehub 0.3.13
- langdetect 1.0.9

ML extra:

- torch 2.6.0
- transformers 4.48.3
- sentence-transformers 3.4.1
- huggingface-hub 0.28.1
- faiss-cpu 1.10.0
- umap-learn 0.5.7
- hdbscan 0.8.40

API extra:

- openai 1.63.2
- tenacity 9.0.0

Dev:

- pytest 8.3.4
- pytest-cov 6.0.0
- ruff 0.9.5
- mypy 1.14.1

---

## 15. Current staged workflow

Install everything:

```powershell
uv sync --extra data --extra ml --extra api --extra dev
```

Confirm tests and static quality:

```powershell
uv run pytest -q
uv run ruff check .
uv run mypy
```

Start real-data work:

```powershell
uv run python scripts/select_brand.py
Get-Content artifacts\brand_selection.json
```

Do not blindly continue if brand selection fails eligibility or produces implausible statistics.
Audit the ranking and resolution-proxy samples first.

Then:

```powershell
uv run python scripts/build_cohort.py
uv run python scripts/derive_taxonomy.py
```

Stop and manually complete `data/annotations/cluster_review.csv`. Then finalize taxonomy with the
CLI's `--review-csv` option.

Next create provisional validation labels, export validation packets, manually complete them,
validate them, train models, and build the retrieval index. Do not use the golden/test set for any
model, prompt, calibration, or threshold decision.

The README's later commands are incomplete because the final runner does not yet exist. Do not
pretend the workflow is complete merely because earlier CLIs run.

---

## 16. Required next implementation sequence

### Gate 1 — compose a tiny end-to-end fixture

Before expensive data work, replace the placeholder test with a fixture pipeline that performs:

```text
thread fixture
-> intent prediction
-> retrieval from train-only fixture cases
-> deterministic/fake cached draft
-> draft validation
-> risk signals
-> policy decision
-> judge normalization
-> evaluation record
-> headline metric construction
```

No network call should be required. The test must assert evidence IDs, no split leakage, action
reason, composite error, and deterministic result serialization.

### Gate 2 — repair the high-risk contracts

1. Audit/replace the resolution proxy.
2. Make runtime retrieval use the persisted FAISS index or remove FAISS claims.
3. Add claim-level evidence support checking or an explicit evidence-adequacy requirement.
4. Define composite auto-handle failure once and use it consistently in validation and test.
5. Add zero-coverage policy fallback.
6. Add validation sample weighting or state and quantify the limitation.
7. Add accessible evidence URLs or clearly describe why the public dataset cannot guarantee them.

### Gate 3 — create the missing orchestration

Add a single final-run CLI that:

- loads frozen taxonomy/models/index/policy choices;
- runs all three systems on the exact same golden records;
- generates or loads cached drafts and judge outputs;
- validates every output and evidence ID;
- constructs `ExampleEvaluationRecord`s;
- computes all metrics and intervals;
- computes judge agreement from paired independent ratings;
- writes `headline_metrics.json`, `example_evaluations.jsonl`, and
  `evaluation_result.json`;
- exits nonzero if required evidence, agreement, metrics, or provenance is missing;
- never tunes on golden data.

### Gate 4 — execute human and empirical work

1. Run EDA and select brand.
2. Build cohort/splits and inspect leakage/duplicate reports.
3. Derive and manually finalize taxonomy.
4. Hand-label 400 validation examples.
5. Train/calibrate systems and build retrieval.
6. Freeze all choices.
7. Sample and hand-label 200 golden examples.
8. Generate main and baseline outputs.
9. Independently human-rate at least 50 reply items.
10. Run final evaluation once without retuning.

### Gate 5 — report and freeze

- Fill every report placeholder with frozen evidence only.
- Include real examples for the top five failure modes.
- Include uncertainty and kappa values.
- State a zero-coverage result honestly if the safety gate rejects all automation.
- Freeze artifact hashes and verify a clean reviewer reproduction in under 15 minutes.
- Time the reproduction from a clean checkout with caches/frozen artifacts exactly as shipped.

---

## 17. Definition of completion

The project is complete only when all of the following are true:

- A brand is selected by reproducible EDA and named in the report.
- Thread, temporal, identity where possible, exact-duplicate, and near-duplicate leakage audits pass
  or have explicit documented exceptions.
- Taxonomy is hand-reviewed, hashed, train-only, and has 6–10 classes including `other`.
- All validation packets are hand-labelled and mutually disjoint.
- The 200-example golden set is hand-labelled from test only, balanced by risk/edge stratum, and
  preserves sampling weights.
- Main system and two fair baselines run on identical golden examples.
- Every generated non-empty draft has auditable historical evidence IDs.
- Policy threshold was selected exclusively on validation labels.
- Composite auto-handle error, coverage, escalation recall, reply acceptance, intent macro/per-class
  F1, ECE, and 95% intervals are reported.
- At least 50 blind human–judge pairs yield binary and quadratic-weighted kappa.
- Five failure modes include real example and evidence IDs plus causal hypotheses.
- The misleading-headline section explicitly limits the interpretation.
- Final artifact hashes verify.
- Clean cached reproduction completes in under 15 minutes.
- Full end-to-end test is substantive, not a placeholder.
- README commands have been executed successfully as written.

---

## 18. Instructions to the next model

1. Do not overwrite or generate fake human annotations.
2. Do not inspect golden/test labels while choosing taxonomy, model, prompt, retrieval thresholds,
   or escalation policy.
3. Do not report fixture-test performance as model quality.
4. Do not claim that citations establish factual entailment.
5. Do not claim FAISS serving unless runtime actually loads and queries it.
6. Do not relax the safety target to obtain nonzero coverage.
7. Preserve user changes and inspect Git status before editing.
8. Use `apply_patch` for source edits.
9. Run focused tests, then full pytest, Ruff, mypy, and `git diff --check` after implementation.
10. Ask the user to perform genuine human-label steps; provide clear packets and validation tools,
    but keep authorship honest.
11. Update this handoff, README, report, and decision log whenever a material design choice changes.
12. Lead with empirical proof. The assignment explicitly values proof more than system complexity.

---

## 19. Fast status summary

```text
Raw dataset:                 READY
Repository/config/schemas:   IMPLEMENTED
Component tests:             46 PASSED (user-reported, 16.34s)
Brand selection result:      NOT RUN
Real cohort/splits:           NOT BUILT
Reviewed taxonomy:           NOT BUILT
Validation labels:           NOT CREATED
Golden labels:               NOT CREATED
Trained real models:         NOT BUILT
Retrieval index:             NOT BUILT
Final orchestration:         MISSING
LLM/human evaluation:        NOT RUN
Headline results:            MISSING
Report results/failures:     PENDING
Frozen reproduction bundle: MISSING
Submission readiness:        0/5 deliverables fully ready
```

The immediate safe next action is to implement the substantive fixture end-to-end composition
test and repair the critical contracts, then run `scripts/select_brand.py` and audit its real EDA
output before proceeding.
