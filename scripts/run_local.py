"""Run the complete local browser-speech and grounded-response demo."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    if not Path("artifacts/demo_cases.jsonl").exists():
        raise SystemExit("Run `uv run --extra data python scripts/build_local_demo.py` first.")
    if not Path("frontend/dist/index.html").exists():
        raise SystemExit("Run `npm run build` in frontend first.")
    os.environ["DEMO_CASES_PATH"] = "artifacts/demo_cases.jsonl"
    os.environ["STATIC_DIR"] = "frontend/dist"
    vosk_model = Path("artifacts/models/vosk-model-en-us-0.22-lgraph")
    if vosk_model.is_dir():
        os.environ["VOSK_MODEL_PATH"] = str(vosk_model)
    import uvicorn

    uvicorn.run("support_agent.api.app:app", host="127.0.0.1", port=7860, reload=False)


if __name__ == "__main__":
    main()
