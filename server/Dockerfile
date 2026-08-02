FROM python:3.10-slim AS base

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY server/app/ ./app/
COPY models/ /models/

# ── Security: run as non-root ─────────────────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Runtime config ────────────────────────────────────────────────────────────
# L1 FIX — comments CANNOT appear inline inside a multi-line ENV instruction.
# Docker treats '#' as part of the value, corrupting every variable that follows.
# Each variable is now on its own ENV line so comments can be placed above them.

# Model path — reflects actual ResNet-18 architecture (not EfficientNet)
ENV MODEL_PATH=/models/resnet18_quant_int8.onnx

# Restrict CORS in production: set to comma-separated origins or "*" for local dev
ENV ALLOWED_ORIGINS=""

# Metrics API key: set a strong secret in production to protect the /metrics endpoint
ENV METRICS_API_KEY=""

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ── Port ─────────────────────────────────────────────────────────────────────
# PORT is honoured so platforms like Render can inject their own port value.
# Falls back to 8000 when not set (local Docker, K8s, ECS).
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/healthz || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# sh -c is required so ${PORT:-8000} is shell-expanded at container start time
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
