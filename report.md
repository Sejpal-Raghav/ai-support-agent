# AI Support Agent for `<Selected Brand>`

This report is intentionally empty until Phase 8. Its required section order is
fixed by the approved implementation plan.
# Report — Hiver Support Agent

## 1. Problem framing and scope

The agent supports one brand selected by documented EDA, not a pre-chosen brand. “Good” means
automating only public, low-risk messages with a historically supported resolution pattern; it
does not mean maximizing reply volume. The prototype does not authenticate customers, access
private data, take account actions, post to Twitter, infer live status, or attempt unsupported
rare intents.

## 2. Data, split, and golden-set design

The fixed 12,000-thread cohort is hash-sampled after EDA selection, then split chronologically by
full reconstructed thread: 70% train, 15% validation, 15% test. The 200-example golden set is
drawn only from test: target 100 edge/risk-flagged and 100 non-edge examples, with documented
inverse-probability weights. The primary annotator labels intent, ambiguity, escalation need and
reason, required/forbidden reply elements, and later retrieval adequacy. A blind 10% re-rate
checks annotation consistency.

## 3. System

The main classifier is MiniLM embeddings plus balanced logistic regression, selected on 100
dedicated validation labels and temperature-scaled using a disjoint 100-label calibration set.
Retrieval uses FAISS `IndexFlatIP` over only safe, resolved train cases and returns at most three
same-intent, different-thread cases after MMR reranking. The reply drafter is evidence-only and
must cite retrieved case IDs. A deterministic validator and selective-prediction policy fail
closed on unsupported claims, hard-risk signals, or uncertainty.

## 4. Baselines and results

Populate this section only after the frozen run is created. Report all numbers with stratified
bootstrap 95% intervals, not just point estimates.

| System | Intent macro-F1 | Auto coverage | Acceptable replies | Escalation recall | Auto error |
|---|---:|---:|---:|---:|---:|
| Majority + fixed acknowledgement + always escalate | pending frozen run | 0% | N/A | 100% | 0% |
| TF-IDF + nearest historical reply + keyword/confidence rule | pending frozen run | pending | pending | pending | pending |
| MiniLM + retrieval-grounded drafter + selective policy | pending frozen run | pending | pending | pending | pending |

Headline template: “At a fixed safety target, the agent can auto-handle **X%** of eligible
messages (95% CI **[L, U]**), with **Y%** acceptable replies and **Z%** recall on cases requiring
escalation.”

## 5. Reply evaluation and judge agreement

The judge uses a model/prompt separate from the drafter and is blinded to system identity. It
scores relevance, grounding, usefulness, tone, and safety 1–4; acceptance requires all first four
dimensions ≥3, safety =4, and no critical violation. An independent human blind-rates at least 50
items; report Cohen’s kappa for accept/reject and quadratic-weighted kappa for ordinal overall
quality. Both statistics are pending the human ratings.

## 6. Failure analysis

Populate the five rows with real golden examples after the frozen evaluation. Each row must link
the example ID, evidence case IDs, observed failure, and causal hypothesis.

| Rank | Golden example ID | Failure | Causal hypothesis | Mitigation |
|---:|---|---|---|---|
| 1 | pending | pending | pending | pending |
| 2 | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending |
| 4 | pending | pending | pending | pending |
| 5 | pending | pending | pending | pending |

## 7. What is misleading about my headline number?

Coverage is measured only among the selected brand’s public, English-capable, evidence-supported
decision points; it is not the fraction of all support work that can be automated. The gold set is
stratified and weighted, but 200 examples still yield uncertainty, especially for rare hard-risk
cases. The policy threshold is selected on validation data and can overfit that limited partition.
Finally, a reply-quality judge is a proxy: agreement with independent human ratings must accompany
the headline, and it does not make the system safe for account actions.

## 8. One more week

1. Double the independent human ratings and adjudicate disagreements.
2. Audit all false auto-handles and turn the common causes into explicit policy features.
3. Add a retriever re-ranking ablation and label retrieval adequacy on more calibration cases.
4. Re-run the full pipeline on a second time slice to test temporal robustness.
5. Build a human-review UI that displays cited case links, risk signals, and the draft validator’s
   reasons; it would still not send messages automatically.
