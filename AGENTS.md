# Repository Guidelines

## Project Structure & Module Organization

Python source lives in `src/support_agent/`. The public API is in `api/`, speech processing in
`voice/`, neural intent code in `intent/`, and the working grounded runtime in `demo_runtime.py`.
The React client is under `frontend/src/`. Reproducible commands live in `scripts/`; tests mirror
the Python modules in `tests/`. Raw data, generated models, and retrieval artifacts belong in
ignored `data/raw/`, `data/interim/`, and `artifacts/` directories.

## Build, Test, and Development Commands

```powershell
uv sync --frozen --extra data --extra ml --extra api --extra dev
uv run --extra api pytest -q
uv run ruff check src scripts tests
cd frontend; npm ci; npm run build
uv run --extra ml --extra api python scripts/run_local.py
```

Build the local corpus with `scripts/ingest_local_demo.py` followed by
`scripts/build_local_demo.py`. Never commit `twcs.csv`, generated evidence, recordings, model
weights, secrets, or caches.

## Coding Style & Naming Conventions

Use Python 3.11, four-space indentation, type annotations, immutable dataclasses/Pydantic models,
and 100-character lines. Ruff enforces `E`, `F`, `I`, `UP`, `B`, and `SIM`. Use `snake_case` for
functions/modules, `PascalCase` for classes, and explicit intent IDs such as `delivery_tracking`.

## Testing Guidelines

Pytest files are named `test_<component>.py`; tests are named `test_<behavior>`. Add a focused unit
test for every safety, schema, or API change. Run the full suite and frontend build before a local
release checkpoint.

## Commit & Pull Request Guidelines

Use Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, or `refactor:`. Stage exact
paths rather than `git add .`. Pull requests must describe behavior, tests, artifact/data impact,
and UI screenshots when the interface changes. Never force-push shared branches.
