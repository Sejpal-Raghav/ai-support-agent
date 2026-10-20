#!/usr/bin/env bash
set -euo pipefail

export PIP_DEFAULT_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60

# Install only the compact inference runtime, without research/training packages.
export EMBEDDING_BACKEND=onnx
python -m pip install --no-cache-dir -r deployment/render/requirements.txt
python -m pip install --no-deps .
npm --prefix frontend ci
npm --prefix frontend run build
python scripts/prepare_render_models.py
