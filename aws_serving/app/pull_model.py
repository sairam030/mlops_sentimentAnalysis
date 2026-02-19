"""
pull_model.py — Download the best registered model from MLflow (DagsHub)
into the local artifacts/ directory for Docker packaging.

Usage:
    python pull_model.py
"""

import os
import shutil

from dotenv import load_dotenv

load_dotenv()

import mlflow
import yaml


def load_params(path="../../params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    mlflow_cfg = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    registered_model_name = mlflow_cfg.get("registered_model_name", "sentiment-best-model")
    model_stage = "latest"  # or "Production" if you use stage transitions

    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")

    # Clean old artifacts
    if os.path.exists(artifacts_dir):
        shutil.rmtree(artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Download the registered model (includes .joblib/.pkl or transformers files)
    print(f"[pull_model] Downloading '{registered_model_name}' from MLflow...")
    model_uri = f"models:/{registered_model_name}/{model_stage}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=artifacts_dir)

    print(f"[pull_model] ✅ Model downloaded to: {local_path}")
    print(f"[pull_model] Contents:")
    for root, dirs, files in os.walk(artifacts_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {os.path.relpath(fpath, artifacts_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
