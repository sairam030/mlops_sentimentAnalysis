# ──────────────────────────────────────────────────────────
# Dockerfile for Sentiment Analysis Serving API
# ──────────────────────────────────────────────────────────
# Build:  docker build -t sentiment-api ./serving
# Run:    docker run -p 8000:8000 --env-file .env.staging sentiment-api
# ──────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# ── Install Python deps ──────────────────────────────────
COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt

# ── Copy application code ────────────────────────────────
COPY serving/ /app/serving/

# Own files as non-root user
RUN mkdir -p /tmp/sentiment_model && chown -R appuser:appuser /app /tmp/sentiment_model

USER appuser

# ── Health check ─────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# ── Run ──────────────────────────────────────────────────
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
