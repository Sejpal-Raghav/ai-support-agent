# Hiver Support Agent

An auditable selective-prediction prototype for one Twitter-support brand. It classifies a
customer turn, retrieves only safe historical train cases, drafts an evidence-cited reply, and
escalates whenever the validation-selected safety policy is not satisfied.

## Reproduce the reported results

The submission must include populated `data/frozen/` artifacts. From a clean checkout, cached
reproduction takes seconds and makes no network or API calls:

```text
uv sync --frozen --extra eval
uv run python scripts/reproduce.py --config configs/pipeline.yaml --mode cached --verify-hashes
```

The command checks every frozen artifact hash before printing `headline_metrics.json`. Required
frozen files are documented in [MANIFEST_FORMAT.md](data/frozen/MANIFEST_FORMAT.md).

## Full pipeline prerequisites

Download Kaggle's `twcs.csv` into the expected local path, then run the remaining commands after
completing the human review packets. The raw dataset and human labels are intentionally not
versioned.

```text
uv sync --extra data --extra ml --extra api --extra dev
uv run python scripts/download_dataset.py
uv run python scripts/select_brand.py
uv run python scripts/build_cohort.py
uv run python scripts/derive_taxonomy.py
# Complete data/annotations/cluster_review.csv, then rerun derive_taxonomy with --review-csv.
uv run python scripts/generate_provisional_labels.py --partition validation
uv run python scripts/export_annotations.py --kind validation
# Complete and validate the three validation packets.
uv run python scripts/train_models.py
uv run python scripts/build_retrieval_index.py
```

`download_dataset.py` calls
`kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")`, validates the source
columns, and copies the downloaded `twcs.csv` to `data/raw/twcs.csv`. Kaggle may prompt you to
authenticate; follow Kaggle's local API-token or browser-authentication instructions. The command
refuses to replace an existing CSV unless you explicitly pass `--overwrite`. If you download the
archive manually instead, extract it and copy `twcs.csv` to `data/raw/twcs.csv`.

The final evaluation runner writes `headline_metrics.json`, `example_evaluations.jsonl`, and
`evaluation_result.json`; freeze exactly those files with `scripts/freeze_results.py` before
submitting. The report does not claim results until this data-dependent process is run.

## Guardrails and scope

- All splits are chronological by reconstructed thread; retrieval indexes train cases only.
- Historical evidence is preserved by case ID, source thread/tweet IDs, and source URI.
- A hard-risk keyword or failed draft validation always escalates.
- The system drafts proposed replies only. It never posts, authenticates, accesses private data,
  or takes account actions.

See [report.md](report.md) for the reviewer report and [decision_log.md](decision_log.md) for
the implementation decisions.
