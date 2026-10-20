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
    PORT=7860
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir uv==0.8.15 \
    && uv sync --frozen --no-dev --extra api \
    && uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python .venv/bin/python sentence-transformers==3.4.1 transformers==4.48.3 huggingface-hub==0.28.1 \
    && .venv/bin/python scripts/prepare_render_models.py
COPY --from=frontend-build /app/frontend/dist /app/static
EXPOSE 7860
CMD [".venv/bin/python", "scripts/run_render.py"]
