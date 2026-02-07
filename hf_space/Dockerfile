# ──────────────────────────────────────────────────────────
# Dockerfile for Hugging Face Spaces (Docker SDK)
# ──────────────────────────────────────────────────────────
# HF Spaces exposes port 7860 by default.
# Model is bundled inside the image (no S3/MLflow needed).
# ──────────────────────────────────────────────────────────

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# ── Install Python deps ──────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────
COPY serving/ /app/serving/

# ── Copy model files (bundled in image) ──────────────────
COPY model/ /app/model/

# Own files
RUN chown -R user:user /app
USER user

# ── Environment for local model loading ──────────────────
ENV LOCAL_MODEL_PATH=/app/model \
    MODEL_STAGE=production \
    HOST=0.0.0.0 \
    PORT=7860 \
    LOG_LEVEL=INFO

EXPOSE 7860

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
