# syntax=docker/dockerfile:1
FROM node:24-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static \
    HF_HOME=/app/artifacts/huggingface \
    MINILM_MODEL_PATH=/app/artifacts/models/minilm \
    VOSK_MODEL_PATH=/app/artifacts/models/vosk-model-small-en-us-0.15 \
    EMBEDDING_BACKEND=onnx \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=2 \
    PIP_DEFAULT_TIMEOUT=120 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ETAG_TIMEOUT=60 \
    PORT=7860
WORKDIR /app
COPY deployment/render/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv==0.8.15 \
    && uv venv --python /usr/local/bin/python .venv \
    && uv pip install --python .venv/bin/python -r requirements.txt
COPY scripts/prepare_render_models.py scripts/download_vosk_model.py ./scripts/
RUN --mount=type=cache,target=/root/.cache/support-agent-models \
    MODEL_DOWNLOAD_CACHE=/root/.cache/support-agent-models \
    .venv/bin/python scripts/prepare_render_models.py
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY --from=frontend-build /app/frontend/dist /app/static
EXPOSE 7860
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s \
    CMD [".venv/bin/python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/api/ready', timeout=4)"]
CMD [".venv/bin/python", "scripts/run_render.py"]
