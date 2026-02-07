#!/bin/bash
# ================================================================
# setup_ec2.sh — Bootstrap an EC2 instance for model serving
# ================================================================
# Run on a fresh Ubuntu 22.04 / Amazon Linux 2023 EC2 instance:
#   chmod +x scripts/setup_ec2.sh
#   sudo ./scripts/setup_ec2.sh
# ================================================================

set -euo pipefail

echo "========================================="
echo "  Setting up EC2 for Sentiment API"
echo "========================================="

# ── 1. Update system ──────────────────────────────────────
echo "[1/5] Updating system packages..."
apt-get update -y && apt-get upgrade -y

# ── 2. Install Docker ────────────────────────────────────
echo "[2/5] Installing Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    # Allow ubuntu user to run docker without sudo
    usermod -aG docker ubuntu
    echo "  ✅ Docker installed"
else
    echo "  ✅ Docker already installed"
fi

# ── 3. Install AWS CLI ───────────────────────────────────
echo "[3/5] Installing AWS CLI..."
if ! command -v aws &>/dev/null; then
    apt-get install -y unzip
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
    echo "  ✅ AWS CLI installed"
else
    echo "  ✅ AWS CLI already installed"
fi

# ── 4. Install Docker Compose ────────────────────────────
echo "[4/5] Installing Docker Compose..."
if ! command -v docker compose &>/dev/null; then
    apt-get install -y docker-compose-plugin
    echo "  ✅ Docker Compose installed"
else
    echo "  ✅ Docker Compose already installed"
fi

# ── 5. Create app directory ──────────────────────────────
echo "[5/5] Creating application directory..."
mkdir -p /opt/sentiment-api
chown ubuntu:ubuntu /opt/sentiment-api

echo ""
echo "========================================="
echo "  ✅ EC2 Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Configure AWS CLI:   aws configure"
echo "  2. Copy .env file:      scp .env.staging ubuntu@<ip>:/opt/sentiment-api/.env"
echo "  3. Login to ECR or Docker Hub"
echo "  4. Pull & run container (see deploy.sh)"
echo ""
