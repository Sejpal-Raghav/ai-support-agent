"""Fail clearly on missing deployment assets, then serve the complete app on Render's port."""

from __future__ import annotations

import os
from pathlib import Path


def configure_environment(root: Path, secret_dir: Path = Path("/etc/secrets")) -> None:
    """Resolve explicit configuration or Render secret files without silent empty services."""
    defaults = {
        "DEMO_CASES_PATH": secret_dir / "demo_cases.jsonl",
        "VOSK_MODEL_PATH": root / "artifacts/models/vosk-model-en-us-0.22-lgraph",
        "STATIC_DIR": root / "frontend/dist",
        "HF_HOME": root / "artifacts/huggingface",
    }
    for name, default in defaults.items():
        os.environ.setdefault(name, str(default))
    cases = Path(os.environ["DEMO_CASES_PATH"])
    required_files = [cases, cases.with_suffix(".manifest.json")]
    required_files.append(Path(os.environ["STATIC_DIR"]) / "index.html")
    missing = [path.name for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Missing deployment files: {', '.join(missing)}. "
            "Add demo_cases.jsonl and demo_cases.manifest.json as Render Secret Files, "
            "and run the frontend/model build before startup."
        )
    if not Path(os.environ["VOSK_MODEL_PATH"]).is_dir():
        raise RuntimeError("Vosk is missing. Run scripts/build_render.sh during the Render build.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    configure_environment(root)
    import uvicorn

    uvicorn.run(
        "support_agent.api.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "10000"))
    )


if __name__ == "__main__":
    main()
