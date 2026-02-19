"""
pull_model.py
=============
Pulls the best registered model from MLflow on DagsHub into
aws_serving/app/artifacts/ so the Flask app can use it.

Folder layout this script lives in:
    sentiment_mlops/
    ├── params.yaml                      ← project root (2 levels up from here)
    ├── aws_serving/
    │   ├── app/
    │   │   ├── app.py
    │   │   ├── Dockerfile
    │   │   ├── requirements.txt
    │   │   └── artifacts/               ← downloads go HERE
    │   └── scripts/
    │       └── pull_model.py            ← THIS FILE

Run from anywhere:
    python aws_serving/scripts/pull_model.py
    # or
    cd aws_serving/scripts && python pull_model.py
"""

import os
import sys

# ── Anchor all paths to this script's location ────────────────────────────────
# aws_serving/scripts/pull_model.py
SCRIPTS_DIR   = os.path.dirname(os.path.abspath(__file__))   # aws_serving/scripts/
AWS_SERVING   = os.path.dirname(SCRIPTS_DIR)                  # aws_serving/
PROJECT_ROOT  = os.path.dirname(AWS_SERVING)                  # sentiment_mlops/

# ── Load .env from project root ───────────────────────────────────────────────
# Your .env must have:
#   MLFLOW_TRACKING_USERNAME=sairam030
#   MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import yaml
import mlflow
from mlflow import MlflowClient

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load params.yaml from project root
# ─────────────────────────────────────────────────────────────────────────────
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

if not os.path.exists(PARAMS_PATH):
    print(f"❌  params.yaml not found at: {PARAMS_PATH}")
    sys.exit(1)

with open(PARAMS_PATH) as f:
    params = yaml.safe_load(f)

TRACKING_URI  = params["mlflow"]["tracking_uri"]
# https://dagshub.com/sairam030/mlops_sentimentAnalysis.mlflow

MODEL_NAME    = params["mlflow"]["registered_model_name"]
# sentiment-best-model

# Artifacts land here — Flask app will load from this folder
OUTPUT_DIR    = os.path.join(AWS_SERVING, "app", "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✅  params.yaml  : {PARAMS_PATH}")
print(f"    tracking_uri : {TRACKING_URI}")
print(f"    model_name   : {MODEL_NAME}")
print(f"    output_dir   : {OUTPUT_DIR}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Connect
# ─────────────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Find the latest registered version
# ─────────────────────────────────────────────────────────────────────────────
print(f"[1/3] Looking up '{MODEL_NAME}' in MLflow registry ...")

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
except Exception as e:
    print(f"❌  Registry query failed: {e}")
    print(f"    Make sure MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD")
    print(f"    are set in {os.path.join(PROJECT_ROOT, '.env')}")
    sys.exit(1)

if not versions:
    print(f"❌  No versions found for '{MODEL_NAME}'.")
    print(f"    Run evaluate.py (or dvc repro evaluate) first.")
    sys.exit(1)

latest  = sorted(versions, key=lambda v: int(v.version))[-1]
RUN_ID  = latest.run_id
VERSION = latest.version
print(f"    ✅  version {VERSION}  |  run_id: {RUN_ID}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Download the registered model + run artifacts
# ─────────────────────────────────────────────────────────────────────────────
import shutil

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step A: Download the actual MODEL via model URI ───────────────────────────
# mlflow.sklearn.log_model() stores under "best_model/" artifact path.
# The registered model URI points directly to that — use it.
model_uri = f"models:/{MODEL_NAME}/{VERSION}"
model_dst = os.path.join(OUTPUT_DIR, "best_model")

print(f"\n[2/3] Downloading registered model via URI: {model_uri}")
print(f"      destination: {model_dst}")
print(f"      (this includes model.pkl / MLmodel / conda.yaml etc.)\n")

try:
    local_model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=model_dst,
    )
    print(f"  ✅  Model downloaded to: {local_model_path}")
except Exception as e:
    print(f"  ❌  Model download failed: {e}")
    print(f"\n  This means mlflow.sklearn.log_model() in evaluate.py")
    print(f"  did NOT actually upload the .pkl/.joblib to DagsHub.")
    print(f"  Re-run:  python src/evaluate.py")
    sys.exit(1)

# ── Step B: Also download run-level artifacts (info JSONs, comparison) ────────
print(f"\n[3/3] Downloading remaining run artifacts (info files, comparison) ...")

def list_artifacts_recursive(client, run_id, path=""):
    """Recursively list all artifacts in a run."""
    items = client.list_artifacts(run_id, path)
    files = []
    for item in items:
        if item.is_dir:
            files.extend(list_artifacts_recursive(client, run_id, item.path))
        else:
            files.append(item)
    return files

all_artifacts = list_artifacts_recursive(client, RUN_ID)

print(f"    Found {len(all_artifacts)} run artifact(s):")
for a in all_artifacts:
    print(f"      {a.path:60s} {a.file_size / 1024:>10.1f} KB")

downloaded = 0
failed = 0
fail = []
for artifact in all_artifacts:
    # Skip best_model/ files — already downloaded via model URI above
    if artifact.path.startswith("best_model"):
        print(f"  ⏭️   {artifact.path}  (already downloaded via model URI)")
        continue
    try:
        local = client.download_artifacts(RUN_ID, artifact.path, OUTPUT_DIR)
        size_kb = os.path.getsize(local) / 1024
        print(f"  ✅  {artifact.path}  ({size_kb:.1f} KB)")
        downloaded += 1
    except Exception as e:
        print(f"  ❌  {artifact.path}  — {e}")
        fail.append((artifact.path, str(e)))
        failed += 1

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ✅  downloaded : {downloaded} extra artifact(s)")
print(f"  ❌  failed     : {failed}")
print(f"{'='*60}")

if fail:
    print("\n  Failed files:")
    for path, err in fail:
        print(f"    {path}")
        print(f"      {err}")

print(f"\n  aws_serving/app/artifacts/ now contains:\n")
for root, dirs, files in os.walk(OUTPUT_DIR):
    level  = root.replace(OUTPUT_DIR, "").count(os.sep)
    pad    = "  " * (level + 1)
    print(f"{'  ' * level}  📁 {os.path.basename(root) or 'artifacts'}/")
    for fname in sorted(files):
        fsize = os.path.getsize(os.path.join(root, fname)) / 1024
        print(f"{pad}📄 {fname}  ({fsize:.1f} KB)")

if not fail:
    print(f"\n✅  All done.  Next step:")
    print(f"    cd {os.path.join(AWS_SERVING, 'app')}")
    print(f"    docker build -t sentiment-api:latest .")