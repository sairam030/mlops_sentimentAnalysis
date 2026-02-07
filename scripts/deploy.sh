#!/bin/bash
# ================================================================
# deploy.sh — Deploy the sentiment API container on EC2
# ================================================================
# Usage:
#   ./scripts/deploy.sh staging     # Deploy to staging
#   ./scripts/deploy.sh production  # Deploy to production
# ================================================================

set -euo pipefail

STAGE=${1:-staging}
IMAGE_NAME="sentiment-api"
CONTAINER_NAME="sentiment-api-${STAGE}"
ENV_FILE="/opt/sentiment-api/.env.${STAGE}"
PORT=$( [ "$STAGE" = "production" ] && echo "80" || echo "8000" )

echo "========================================="
echo "  Deploying: ${STAGE}"
echo "  Container: ${CONTAINER_NAME}"
echo "  Port:      ${PORT}"
echo "========================================="

# ── Stop existing container ───────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[1/3] Stopping existing container..."
    docker stop "${CONTAINER_NAME}" && docker rm "${CONTAINER_NAME}"
else
    echo "[1/3] No existing container to stop"
fi

# ── Pull latest image ────────────────────────────────────
echo "[2/3] Pulling latest image..."
# Uncomment for ECR:
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
# docker pull <account-id>.dkr.ecr.us-east-1.amazonaws.com/${IMAGE_NAME}:latest
# IMAGE_NAME="<account-id>.dkr.ecr.us-east-1.amazonaws.com/${IMAGE_NAME}:latest"

# For Docker Hub:
# docker pull yourusername/${IMAGE_NAME}:latest

echo "  (Using local image for now)"

# ── Run container ─────────────────────────────────────────
echo "[3/3] Starting container..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --env-file "${ENV_FILE}" \
    -p "${PORT}:8000" \
    --memory=4g \
    --cpus=2 \
    "${IMAGE_NAME}:latest"

echo ""
echo "========================================="
echo "  ✅ Deployed!"
echo "========================================="
echo ""
echo "  Health check:  curl http://localhost:${PORT}/health"
echo "  Predict:       curl -X POST http://localhost:${PORT}/predict \\"
echo "                   -H 'Content-Type: application/json' \\"
echo "                   -d '{\"texts\": [\"Great video!\"]}'"
echo "  Logs:          docker logs -f ${CONTAINER_NAME}"
echo ""
