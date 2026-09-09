# LLM Council Transcript

- Timestamp: 2026-09-09T20:52:28Z
- Mode: Standard (auto-selected — default)
- Question SHA-1 prefix: `daef9321`
- Codex Decision Science: not requested
- Journal lookup: no related run found; Bash helper unavailable on this Windows host
- Workspace scan: no `CLAUDE.md` or `memory/` project context found

## Framed question

DECISION: Is the current Hiver SDE Intern take-home repository genuinely submission-ready, and
which implementation-plan decisions must be retained, revised, or rejected?

CONTEXT: The repository implements phases 0–8 for brand EDA, thread reconstruction, train-only
taxonomy, intent models, retrieval, grounded drafting, selective prediction, evaluation, and
cached reproduction. It reports 42 passing tests, Ruff clean, and mypy clean. However, the real
Kaggle data, selected brand, hand-reviewed taxonomy, 600 human annotations, actual model runs,
judge-human agreement, headline metrics, failure examples, final evaluation runner, and frozen
artifacts are absent. Review every one of the 15 decisions in `decision_log.md` and corresponding
code, not merely the documents.

STAKES: Hiver will scrutinize leakage, fair baselines, non-circular reply evaluation, human–judge
agreement, validation-only thresholds, credible uncertainty, under-15-minute reproduction, and
live explainability.

OPTIONS: (A) submit after filling data/results only; (B) revise architecture/code first, then
execute; (C) substantially simplify/rebuild around a smaller end-to-end proof.

PRIOR: No related council run found.

## Bias audit

- IKEA effect: phases and test volume may inflate confidence in an architecture never run on real
  data. Judge readiness from end-to-end evidence independent of implementation effort.
- Anchoring: “42 tests/Ruff/mypy” emphasizes hygiene over missing outcome evidence.
- Sunk cost: retain components only when they strengthen the smallest credible submission.
- Planning fallacy: data filling will expose rework; budget contingency for architecture changes.

## Anonymization map

- A = First Principles
- B = Outsider
- C = Expansionist
- D = Executor
- E = Red Team

The permutation was generated from the first four SHA digits (`daef`). Persona-signature headings
were structurally sanitized for peer review; confidence blocks were preserved.

## Advisor responses

### Red Team

The repo is an architectural scaffold, not submission-ready. Decisions 1–4 are defensible but
unproven. Cluster pseudo-label training risks measuring conformity to the taxonomy generator.
`retrieval/cases.py` treats actionable instructions as resolved without confirmation;
`retrieve_cases` ignores the saved FAISS index; citations do not establish semantic entailment;
the policy weights and isotonic fit are unvalidated; and `reproduce.py` prints frozen JSON instead
of reproducing the pipeline or tables. The root cause is optimizing contracts and synthetic tests
over end-to-end evidence. Recommendation: retain 1, 3, 4, 6, 10, 12, 13; revise 2, 5, 7–9, 11,
14–15; do not submit until real artifacts exist.

Confidence: high. Unknowns: real distribution, API availability, runtime.

### First Principles

Three reframings: filling data assumes components compose, but the end-to-end test is `assert True`
and reproduction is print-only; proof-first integration repair retains useful modular work while
repairing validation reuse, proxies, FAISS, entailment, judge independence and freezing; a smaller
TF-IDF/nearest-neighbor/rules rebuild is a fallback if integration fails. Strongest: B. Runner-up:
C if one dry run cannot produce auditable outputs within the time budget.

Confidence: high. Breakers: a successful clean fixture run and completed human evidence.

### Expansionist

Option X: proof-spine sprint—add evaluator, executable baselines, audit JSONL and a real E2E test;
run a 30-item pilot. Option Y: redesign the statistical budget because zero errors among 50
auto-handles still yields about a 7.1% Wilson upper bound. Option Z: add a 30-item adversarial
evidence challenge. Dominant: X + Z with Y's correction.

Confidence: high. Unknowns: brand, class balance, case yield, budget and raters.

### Outsider

Clinical-triage lens: auto-handle is a negative test releasing a case without human review.
Component tests do not establish clinical validity. Audit end-time/identity/near-duplicate leakage;
reject actionable-verb resolution until audited; test intent-agnostic and oracle-intent retrieval;
precompute policy feasibility; and conduct prospective validation with every false auto-handle
inspected. Choose B; C only if a real slice fails.

Confidence: high. Unknowns: actual distribution and validation outcomes.

### Executor

OODA: Orient → Act, status incomplete. There is no inference orchestrator, policy-calibration CLI,
baseline runner, human-rating import, or result generator. A gate-first vertical slice followed by
B has the best reach/impact/confidence per effort. Decision 14 must tune the composite auto-handle
failure—including reply unacceptability—and allow zero coverage instead of raising.

Confidence: high (0.96). Remaining work estimate: 2–4 engineering days plus human/API work.

## Peer reviews

1. Strongest A; it covers all 15 decisions. Weakness: it preserves untested choices too readily.
   Collective miss: explicit immutable human checkpoints. `CONSENSUS STRENGTH: 5`
2. Strongest D; it isolates the sequencing error and composite failure definition. Weakness:
   claims require file/line evidence and a cold-run transcript. Collective miss: detailed blind
   human protocol and cached-vs-fresh timing distinction. `CONSENSUS STRENGTH: 4`
3. Strongest B recommendation. Weakness: calling `started_at` automatically leaky overstates the
   proof; the real issue is missing audits. Collective miss: tweet redistribution/privacy policy
   and full provenance. `CONSENSUS STRENGTH: 5`
4. Strongest C; it exposes the Wilson feasibility problem. Weakness: 30 examples are diagnostic,
   not an evaluation. Collective miss: `tweet://` is not a reviewer-usable source link and case
   citations are not claim entailment. `CONSENSUS STRENGTH: 4`
5. Strongest E; it identifies code-level claim failures. Weakness: frozen JSON can be fabricated
   without a provenance-linked evaluator. Collective miss: annotation provenance.
   `CONSENSUS STRENGTH: 5`

Average consensus strength: **4.6/5**. Forced debate triggered.

## Debate round

### Prosecutor

The consensus may still overvalue sunk cost: the only E2E test is `assert True`, so integration
risk is unbounded. Choose C—a minimal one-CLI rebuild around one brand and 200 immutable labels;
reuse functions only after they satisfy that contract. This bounds scope and makes every retained
line serve measured proof.

### Defender

Concedes no system proof. A full rebuild is not yet justified because 41 substantive component
tests and roughly ten CLIs are functional assets. Time-box a tiny-fixture composition gate; if
adapters compose, repair, and if contracts conflict, rebuild only the incompatible path. Remaining
risks are schema drift, weak proxies, zero coverage and absent human evidence.

## Chairman-Consensus verdict

Council confidence: high (5/5 high, 0/5 medium, 0/5 low)

Dominant assumption: existing modules are reusable, but no complete inference-to-evaluation path
has been demonstrated. Breakers: absent real data/evidence; failure of the time-boxed vertical
slice.

The council selects B behind a 48-hour composition gate. Retain decisions 3, 4, 6 and 13; revise
1, 2, 5, 7, 9–12, 14 and 15; reject decision 8's current actionable-verb proxy. Add human
provenance, privacy controls, claim-level support checks, a real evaluator, all baselines and a
reproduction command that recomputes tables. Use C if the vertical slice fails quickly.

## Chairman-Dissent verdict

Also selects B, but demands a one-working-day gate and distinguishes confirmed defects from
unproved choices. It preserves the possibility that a valid 5% Wilson gate yields zero coverage,
warns that resolution selection may use future outcomes, questions validation reuse/weighting,
and treats Decision 15 as not completed rather than merely needing revision.

## Final verdict

Choose B with proof before scale. Do not submit. First create one executable vertical slice that
emits per-system predictions, evidence, drafts, decisions, judge records and metrics. If it cannot
pass substantive assertions within 48 hours, switch to C. Then pilot real data, repair proxy and
statistics, lock annotation protocol, complete 200 gold examples and ≥50 independent ratings, and
freeze provenance-complete outputs.

## Dissent Ledger

- DISSENT PRESERVED: Start with 200 real labels, not only a 30-example pilot—the stronger evidence
  gate is materially more expensive but closer to submission proof.
- DISSENT PRESERVED: A 5% Wilson bound may legitimately yield zero coverage and must not be relaxed.
- DISSENT PRESERVED: Resolution-proxy features may use information unavailable at inference time.
- DISSENT PRESERVED: Validation reuse and sampling weights require explicit audit despite split-ID
  disjointness.
- DISSENT PRESERVED: Decision 15 is infrastructure, not empirically completed reproduction.

