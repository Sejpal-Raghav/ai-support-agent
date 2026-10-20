#!/usr/bin/env bash
set -euo pipefail

export PIP_DEFAULT_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60

# CPU wheels avoid installing the GPU runtime on a CPU web service.
python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir '.[api]' sentence-transformers==3.4.1 transformers==4.48.3 huggingface-hub==0.28.1
npm --prefix frontend ci
npm --prefix frontend run build
python scripts/prepare_render_models.py
