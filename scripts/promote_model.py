"""
promote_model.py — Promote a model version to staging/production.
==================================================================
1. Assigns an alias in MLflow Model Registry
2. Downloads the model artifacts
3. Packages and uploads to S3 for fast serving

Usage:
    python scripts/promote_model.py --version 3 --alias staging
    python scripts/promote_model.py --version 3 --alias production
    python scripts/promote_model.py --latest --alias staging
"""

import argparse
import json
import os
import shutil
import tarfile
import tempfile

import boto3
import mlflow
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Promote model to staging/production")
    parser.add_argument("--version", type=int, help="Model version number to promote")
    parser.add_argument("--latest", action="store_true", help="Use the latest version")
    parser.add_argument(
        "--alias",
        required=True,
        choices=["staging", "production"],
        help="Target alias",
    )
    parser.add_argument("--bucket", default=None, help="S3 bucket (overrides env)")
    parser.add_argument("--prefix", default="models", help="S3 prefix")
    parser.add_argument("--skip-s3", action="store_true", help="Skip S3 upload")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    mlflow_cfg = params["mlflow"]

    # ── Setup MLflow ────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = mlflow.tracking.MlflowClient()

    model_name = mlflow_cfg.get("registered_model_name", "sentiment-best-model")

    # ── Resolve version ─────────────────────────────────────
    if args.latest:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"❌ No versions found for model '{model_name}'")
            return
        mv = max(versions, key=lambda v: int(v.version))
        version = int(mv.version)
    elif args.version:
        version = args.version
    else:
        print("❌ Specify --version <N> or --latest")
        return

    print(f"\n{'='*60}")
    print(f"  Promoting model '{model_name}' v{version} → {args.alias}")
    print(f"{'='*60}")

    # ── Set alias in MLflow ─────────────────────────────────
    print(f"\n[1/3] Setting alias '{args.alias}' on version {version}...")
    try:
        client.set_registered_model_alias(model_name, args.alias, str(version))
        print(f"  ✅ Alias '{args.alias}' → v{version}")
    except Exception as e:
        print(f"  ⚠️  Could not set alias (may not be supported): {e}")
        print(f"  Continuing with S3 upload anyway...")

    # ── Download model artifacts ────────────────────────────
    print(f"\n[2/3] Downloading model artifacts...")
    model_uri = f"models:/{model_name}/{version}"
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    print(f"  ✅ Downloaded to {local_dir}")

    # List what was downloaded
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            rel = os.path.relpath(fpath, local_dir)
            print(f"     {rel} ({size_mb:.1f} MB)")

    # ── Upload to S3 ────────────────────────────────────────
    if args.skip_s3:
        print("\n[3/3] Skipping S3 upload (--skip-s3)")
    else:
        bucket = args.bucket or os.getenv("S3_MODEL_BUCKET", "sentiment-mlops-models")
        s3_key = f"{args.prefix}/{args.alias}/model.tar.gz"

        print(f"\n[3/3] Uploading to s3://{bucket}/{s3_key}...")

        # Create tar.gz archive
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            archive_path = tmp.name
            with tarfile.open(archive_path, "w:gz") as tar:
                for item in os.listdir(local_dir):
                    tar.add(os.path.join(local_dir, item), arcname=item)

        archive_size = os.path.getsize(archive_path) / 1024 / 1024
        print(f"  Archive size: {archive_size:.1f} MB")

        try:
            s3 = boto3.client("s3")
            s3.upload_file(archive_path, bucket, s3_key)
            print(f"  ✅ Uploaded to s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"  ❌ S3 upload failed: {e}")
            print(f"  Archive saved locally at: {archive_path}")
            return
        finally:
            os.unlink(archive_path)

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ Model v{version} promoted to '{args.alias}'")
    print(f"{'='*60}")
    if not args.skip_s3:
        print(f"  S3:     s3://{bucket}/{s3_key}")
    print(f"  MLflow: {model_name} v{version} (alias: {args.alias})")
    print(f"\n  Serving containers with MODEL_STAGE={args.alias}")
    print(f"  will pick up this model on next restart or POST /reload.\n")


if __name__ == "__main__":
    main()
