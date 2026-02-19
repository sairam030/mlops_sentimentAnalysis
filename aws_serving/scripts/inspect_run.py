"""
inspect_run.py — Show EXACTLY what artifacts are stored in your MLflow run
Run this first to know the correct artifact paths before downloading.

Usage:
    python inspect_run.py
"""

import sys
from dotenv import load_dotenv
load_dotenv()

import mlflow
from mlflow import MlflowClient

# ── CONFIG — edit these to match your setup ───────────────────────────────────
TRACKING_URI = "https://dagshub.com/sairam030/mlops_sentimentAnalysis.mlflow"
MODEL_NAME = "sentiment-best-model"
# ──────────────────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print("=" * 65)
print("  MLflow Run Inspector")
print("=" * 65)

# ── Step 1: Find all registered versions ─────────────────────────────────────
print(f"\n[1] Registered model: '{MODEL_NAME}'")
try:
    # New API (mlflow >= 2.9) — search by name instead of get_latest_versions
    results = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not results:
        print("    ❌ No versions found. Did evaluate.py finish successfully?")
        sys.exit(1)

    for v in results:
        print(f"    Version {v.version} | Run ID: {v.run_id} | Status: {v.current_stage}")

    # Use the latest version
    latest = sorted(results, key=lambda v: int(v.version))[-1]
    RUN_ID = latest.run_id
    print(f"\n    → Using latest: Version {latest.version}, Run ID: {RUN_ID}")

except Exception as e:
    # Fallback: use the hardcoded Run ID from your error message
    RUN_ID = "f3b62b7260ec4e33bd27340a1f268984"
    print(f"    ⚠ Could not query registry ({e})")
    print(f"    → Falling back to Run ID from error: {RUN_ID}")

# ── Step 2: List ALL artifacts in the run recursively ────────────────────────
print(f"\n[2] All artifacts in run {RUN_ID}:")
print("-" * 65)


def list_artifacts_recursive(client, run_id, path="", indent=0):
    """Walk the artifact tree and print every file."""
    try:
        artifacts = client.list_artifacts(run_id, path)
    except Exception as e:
        print(f"{'  ' * indent}❌ Error listing '{path}': {e}")
        return

    if not artifacts:
        print(f"{'  ' * indent}(empty folder: '{path}')")
        return

    for artifact in artifacts:
        prefix = "  " * indent
        if artifact.is_dir:
            print(f"{prefix}📁 {artifact.path}/")
            list_artifacts_recursive(client, run_id, artifact.path, indent + 1)
        else:
            size_kb = (artifact.file_size or 0) / 1024
            print(f"{prefix}📄 {artifact.path}  ({size_kb:.1f} KB)")


list_artifacts_recursive(client, RUN_ID)

# ── Step 3: Show run params & metrics ────────────────────────────────────────
print(f"\n[3] Run params & metrics:")
print("-" * 65)
run = client.get_run(RUN_ID)
for k, v in run.data.params.items():
    print(f"    param  {k} = {v}")
for k, v in run.data.metrics.items():
    print(f"    metric {k} = {v:.4f}")

print(f"\n[4] Artifact root URI:")
print(f"    {run.info.artifact_uri}")

print("\n" + "=" * 65)
print("  Copy the exact artifact paths from [2] above")
print("  and use them in pull_model.py")
print("=" * 65)
