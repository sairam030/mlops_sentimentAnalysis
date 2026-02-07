#!/bin/bash
# ================================================================
# deploy_hf_space.sh — Deploy Sentiment API to Hugging Face Spaces
# ================================================================
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login (with your HF token)
#
# Usage:
#   chmod +x scripts/deploy_hf_space.sh
#   ./scripts/deploy_hf_space.sh
#   ./scripts/deploy_hf_space.sh your-hf-username    # custom username
# ================================================================

set -euo pipefail

HF_USER=${1:-sairam030}
SPACE_NAME="sentiment-api"
SPACE_REPO="${HF_USER}/${SPACE_NAME}"
SPACE_DIR="/tmp/hf_space_${SPACE_NAME}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================="
echo "  Deploying to HF Spaces: ${SPACE_REPO}"
echo "========================================="

# ── 1. Check prerequisites ───────────────────────────────
echo "[1/6] Checking prerequisites..."
if ! command -v git-lfs &>/dev/null; then
    echo "  Installing git-lfs..."
    sudo apt-get install -y git-lfs || (curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && sudo apt-get install -y git-lfs)
    git lfs install
fi

if ! pip show huggingface_hub &>/dev/null; then
    echo "  Installing huggingface_hub..."
    pip install huggingface_hub -q
fi
echo "  ✅ Prerequisites ready"

# ── 2. Create/clone the Space ────────────────────────────
echo "[2/6] Setting up Space repo..."
if [ -d "${SPACE_DIR}" ]; then
    echo "  Cleaning existing directory..."
    rm -rf "${SPACE_DIR}"
fi

# Create space if it doesn't exist, then clone
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo('${SPACE_REPO}', repo_type='space', space_sdk='docker', exist_ok=True)
    print('  ✅ Space created/exists: ${SPACE_REPO}')
except Exception as e:
    print(f'  ⚠️  {e}')
"

git clone "https://huggingface.co/spaces/${SPACE_REPO}" "${SPACE_DIR}" || {
    mkdir -p "${SPACE_DIR}"
    cd "${SPACE_DIR}"
    git init
    git remote add origin "https://huggingface.co/spaces/${SPACE_REPO}"
}

cd "${SPACE_DIR}"
git lfs install

# ── 3. Copy files ────────────────────────────────────────
echo "[3/6] Copying files..."

# Clean old files
find "${SPACE_DIR}" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# Copy HF Space config
cp "${PROJECT_ROOT}/hf_space/Dockerfile" "${SPACE_DIR}/Dockerfile"
cp "${PROJECT_ROOT}/hf_space/README.md" "${SPACE_DIR}/README.md"
cp "${PROJECT_ROOT}/hf_space/requirements.txt" "${SPACE_DIR}/requirements.txt"

# Copy serving code
cp -r "${PROJECT_ROOT}/serving" "${SPACE_DIR}/serving"
# Remove env files with secrets
rm -f "${SPACE_DIR}/serving/.env.staging" "${SPACE_DIR}/serving/.env.production" "${SPACE_DIR}/serving/.env.local"

# Copy model files
echo "  Copying DistilBERT model (~260MB)..."
mkdir -p "${SPACE_DIR}/model"
cp "${PROJECT_ROOT}/models/distilbert_sentiment/config.json" "${SPACE_DIR}/model/"
cp "${PROJECT_ROOT}/models/distilbert_sentiment/model.safetensors" "${SPACE_DIR}/model/"
cp "${PROJECT_ROOT}/models/distilbert_sentiment/tokenizer.json" "${SPACE_DIR}/model/"
cp "${PROJECT_ROOT}/models/distilbert_sentiment/tokenizer_config.json" "${SPACE_DIR}/model/"

# Copy model info if available
if [ -f "${PROJECT_ROOT}/models/distilbert_info.json" ]; then
    cp "${PROJECT_ROOT}/models/distilbert_info.json" "${SPACE_DIR}/model/distilbert_info.json"
fi

echo "  ✅ Files copied"

# ── 4. Track large files with LFS ────────────────────────
echo "[4/6] Setting up Git LFS for large files..."
cd "${SPACE_DIR}"
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.pt"
echo "  ✅ LFS tracking configured"

# ── 5. Show what we're deploying ─────────────────────────
echo "[5/6] Deployment contents:"
echo "  ──────────────────────────"
find "${SPACE_DIR}" -not -path '*/.git/*' -not -name '.git' -type f | while read f; do
    size=$(du -sh "$f" | cut -f1)
    echo "  ${size}  $(echo $f | sed "s|${SPACE_DIR}/||")"
done
echo "  ──────────────────────────"

# ── 6. Commit and push ───────────────────────────────────
echo "[6/6] Committing and pushing..."
cd "${SPACE_DIR}"
git add -A
git commit -m "Deploy sentiment-api: DistilBERT fine-tuned (85.5% accuracy)" || echo "  No changes to commit"

echo ""
echo "========================================="
echo "  Ready to push!"
echo "========================================="
echo ""
echo "  Run this command to push:"
echo ""
echo "    cd ${SPACE_DIR} && git push origin main"
echo ""
echo "  After push, your API will be live at:"
echo "    https://${HF_USER}-${SPACE_NAME}.hf.space"
echo ""
echo "  Swagger docs:"
echo "    https://${HF_USER}-${SPACE_NAME}.hf.space/docs"
echo ""
