"""Fail clearly on missing deployment assets, then serve the complete app on Render's port."""

from __future__ import annotations

import os
from pathlib import Path

from support_agent.deployment import (
    BUNDLE_NAME,
    container_memory_limit,
    unpack_evidence,
    validate_evidence,
    validate_memory_budget,
)


def configure_environment(root: Path, secret_dir: Path = Path("/etc/secrets")) -> None:
    """Resolve explicit configuration or Render secret files without silent empty services."""
    bundle = Path(os.environ.get("INFERENCE_BUNDLE_PATH", str(secret_dir / BUNDLE_NAME)))
    if "DEMO_CASES_PATH" not in os.environ and bundle.is_file():
        staged = unpack_evidence(bundle, root / "artifacts/deployed_evidence")
        os.environ["DEMO_CASES_PATH"] = str(staged)
    elif (
        "DEMO_CASES_PATH" not in os.environ
        and "INFERENCE_BUNDLE_PATH" in os.environ
        and not bundle.is_file()
    ):
        raise RuntimeError("The configured INFERENCE_BUNDLE_PATH does not exist.")
    defaults = {
        "DEMO_CASES_PATH": secret_dir / "demo_cases.jsonl",
        "VOSK_MODEL_PATH": root / "artifacts/models/vosk-model-en-us-0.22-lgraph",
        "STATIC_DIR": root / "frontend/dist",
        "HF_HOME": root / "artifacts/huggingface",
        "MINILM_MODEL_PATH": root / "artifacts/models/minilm",
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
            f"Add {BUNDLE_NAME} under Render Secret Files (or the matching JSONL/manifest pair), "
            "and run the frontend/model build before startup."
        )
    if not Path(os.environ["VOSK_MODEL_PATH"]).is_dir():
        raise RuntimeError("Vosk is missing. Run scripts/build_render.sh during the Render build.")
    model = Path(os.environ["MINILM_MODEL_PATH"])
    if not (model / "modules.json").is_file() or not (model / "config.json").is_file():
        raise RuntimeError("The prepared MiniLM model is missing. Rebuild the deployment image.")
    validate_evidence(cases)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("Deployment assets verified; starting speech and response services.", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    configure_environment(root)
    validate_memory_budget(container_memory_limit())
    import uvicorn

    uvicorn.run(
        "support_agent.api.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "10000"))
    )


if __name__ == "__main__":
    main()
