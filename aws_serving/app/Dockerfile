# app/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile for Sentiment Analysis Flask API
# Supports whatever model is registered in MLflow (SVM or DistilBERT)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py .
COPY database.py .
COPY templates/ ./templates/

# ── Copy model artifacts ──────────────────────────────────────────────────────
# Run: python aws_serving/scripts/pull_model.py FIRST
# This downloads the best registered model (SVM .joblib or DistilBERT)
# from MLflow (DagsHub) into artifacts/
# Only copy the best_model directory to minimize image size
COPY artifacts/best_model/ ./artifacts/best_model/

# Add this block to ensure vocab.txt is present for DistilBERT
RUN if [ ! -f ./artifacts/best_model/components/tokenizer/vocab.txt ]; then \
      echo "Downloading vocab.txt for DistilBERT..."; \
      wget -q -O ./artifacts/best_model/components/tokenizer/vocab.txt https://huggingface.co/distilbert-base-uncased/resolve/main/vocab.txt; \
    fi \
    && rm -rf /tmp/* /var/tmp/*

# ── Pre-download sentence-transformer if SVM model needs it ───────────────────
# Checks MLmodel file to detect flavor; only downloads if sklearn (SVM)
# Also installs SVM-specific Python packages conditionally
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN if grep -q "sklearn" artifacts/best_model/MLmodel 2>/dev/null; then \
        echo "SVM model detected — installing sklearn dependencies..."; \
        pip install --no-cache-dir scikit-learn==1.4.2 joblib==1.4.2 numpy==1.26.4 sentence-transformers==3.0.1; \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"; \
    else \
        echo 'Best model is transformers-based — skipping SVM dependencies'; \
    fi \
    && rm -rf /root/.cache/pip /root/.cache/torch /tmp/* /var/tmp/*

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ── Launch with gunicorn ──────────────────────────────────────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]