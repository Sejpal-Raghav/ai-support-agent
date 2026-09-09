# Decision Log

1. **Choose the brand by EDA rather than by name.** A candidate must meet the configured volume,
   multi-turn, brand-reply, and resolution-proxy criteria; this prevents selecting a familiar brand
   whose data cannot support the task.
2. **Use a fixed 12,000-thread cohort.** It is large enough for credible holdout analysis while
   keeping cached reproduction and local embedding work practical.
3. **Split complete reconstructed threads chronologically 70/15/15.** A message-level random split
   would leak near-identical conversation context; chronological ordering also reduces future-data
   leakage.
4. **Derive 6–10 taxonomy classes from train-only clustering and human review.** The taxonomy is
   data-derived, while `other` absorbs rare/noisy clusters that are unsafe to automate.
5. **Use MiniLM plus balanced logistic regression as the main intent system.** It is explainable,
   fast on CPU, and provides a full confidence distribution for selective prediction.
6. **Use majority and word/character TF-IDF logistic baselines.** They are meaningful controls:
   one tests label imbalance and one tests a credible non-semantic classifier.
7. **Tune C on 100 model-selection labels and temperature only on a disjoint 100 labels.** This
   keeps model selection and confidence calibration from contaminating threshold selection.
8. **Construct retrieval cases only from safe train customer→brand resolutions.** Unresolved,
   generic, hard-risk, and unsupported-intent cases are retained for audit but excluded from the
   FAISS index.
9. **Retrieve up to three same-intent cases using `IndexFlatIP` plus MMR.** This maintains semantic
   relevance while reducing duplicate historical replies, and every result preserves source IDs.
10. **Require evidence citations in every non-empty drafted reply.** The drafter may not rely on
    general brand knowledge; unsupported citations or missing citations fail closed.
11. **Use a separate judge model/prompt from the drafter.** This reduces circular self-evaluation;
    code—not the judge—computes the acceptance gate from the five rubric dimensions.
12. **Require at least 50 independently blind human ratings for judge agreement.** Cohen’s kappa
    and quadratic-weighted kappa reveal agreement beyond raw match percentage.
13. **Make hard-scope matches and failed draft validation mandatory escalation.** A high model
    confidence must never override privacy, money, legal/safety, account-action, or draft risks.
14. **Choose the automation threshold only on validation policy labels.** Maximize coverage only if
    the 95% Wilson upper bound on auto error is ≤5%, escalation recall is ≥90%, and the minimum
    auto count is met.
15. **Freeze content-addressed final artifacts for reviewer reproduction.** The cached command
    verifies SHA-256 hashes and never downloads data or calls APIs, making headline tables quick
    and auditable.
