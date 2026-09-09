# Frozen result contract

`manifest.json` is written only by `scripts/freeze_results.py`. It inventories immutable copies
of the final cached result files and their SHA-256 hashes. Reviewer reproduction requires these
three files:

- `headline_metrics.json` — the exact numbers used in the report headline and tables.
- `example_evaluations.jsonl` — one auditable record per golden example, including evidence IDs.
- `evaluation_result.json` — complete metric estimates, confidence intervals, agreement, and trust gate.

The cached command verifies hashes before printing results and never downloads the dataset, builds
embeddings, or sends an API request.
