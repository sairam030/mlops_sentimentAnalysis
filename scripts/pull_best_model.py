"""
pull_best_model.py — Download best model from MLflow Model Registry for CI/CD
===============================================================================
This script is used in the GitHub Actions pipeline to pull the production
model from MLflow and prepare artifacts for Docker image build.

Usage:
    python scripts/pull_best_model.py --alias production --output aws_serving/app/artifacts
    python scripts/pull_best_model.py --alias staging --output /tmp/models
    python scripts/pull_best_model.py --version 5 --output ./artifacts
"""

import argparse
import json
import os
import shutil

import mlflow
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_params(path="params.yaml"):
    """Load parameters from params.yaml"""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Download best model from MLflow for CI/CD deployment"
    )
    parser.add_argument(
        "--alias",
        choices=["production", "staging"],
        help="Model alias to download (production or staging)",
    )
    parser.add_argument(
        "--version", type=int, help="Specific model version number to download"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for model artifacts (e.g., aws_serving/app/artifacts)",
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    if not args.alias and not args.version:
        print("❌ Specify either --alias or --version")
        return

    # Load configuration
    params = load_params(args.params)
    mlflow_cfg = params["mlflow"]

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = mlflow.tracking.MlflowClient()

    model_name = mlflow_cfg.get("registered_model_name", "sentiment-best-model")

    print(f"\n{'=' * 70}")
    print(f"  📦 Downloading Model from MLflow Model Registry")
    print(f"{'=' * 70}")
    print(f"  Model name: {model_name}")
    print(f"  Tracking URI: {mlflow_cfg['tracking_uri']}")

    # Resolve model version
    if args.alias:
        print(f"  Alias: {args.alias}")
        try:
            # Try to get version by alias
            model_version = client.get_model_version_by_alias(model_name, args.alias)
            version = int(model_version.version)
            print(f"  ✅ Resolved alias '{args.alias}' → version {version}")
        except Exception as e:
            print(f"  ⚠️  Could not resolve alias (feature may not be available): {e}")
            print(f"  💡 Using latest version instead...")
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"❌ No versions found for model '{model_name}'")
                return
            mv = max(versions, key=lambda v: int(v.version))
            version = int(mv.version)
    else:
        version = args.version
        print(f"  Version: {version}")

    print(f"{'=' * 70}\n")

    # Download model artifacts
    print(f"[1/4] Downloading model version {version}...")
    model_uri = f"models:/{model_name}/{version}"

    try:
        # Download all artifacts for this model version
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        print(f"  ✅ Downloaded to: {local_dir}\n")

        # Show what was downloaded
        print(f"[2/4] Examining downloaded artifacts...")
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                fpath = os.path.join(root, f)
                size_mb = os.path.getsize(fpath) / 1024 / 1024
                rel = os.path.relpath(fpath, local_dir)
                print(f"     📄 {rel} ({size_mb:.2f} MB)")

        # Copy to output directory
        print(f"\n[3/4] Copying artifacts to {args.output}...")
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

        # Copy best_model directory
        best_model_src = os.path.join(local_dir, "best_model")
        best_model_dst = os.path.join(output_dir, "best_model")

        if os.path.exists(best_model_dst):
            print(f"  🗑️  Removing existing {best_model_dst}...")
            shutil.rmtree(best_model_dst)

        if os.path.exists(best_model_src):
            shutil.copytree(best_model_src, best_model_dst)
            print(f"  ✅ Copied best_model/ to {best_model_dst}")
        elif os.path.exists(os.path.join(local_dir, "MLmodel")):
            # MLflow puts model files at root level (no best_model/ wrapper)
            print(f"  📦 MLmodel found at root — copying entire download as best_model/")
            shutil.copytree(local_dir, best_model_dst)
            print(f"  ✅ Copied model files to {best_model_dst}")
        else:
            print(f"  ⚠️  No model files found in {local_dir}")
            print(f"  📂 Contents: {os.listdir(local_dir)}")

        # Copy comparison artifacts (model_infos/)
        model_infos_src = os.path.join(local_dir, "model_infos")
        if os.path.exists(model_infos_src):
            for info_file in os.listdir(model_infos_src):
                src = os.path.join(model_infos_src, info_file)
                dst = os.path.join(output_dir, info_file)
                shutil.copy2(src, dst)
                print(f"  ✅ Copied {info_file}")

        # Copy params.yaml and model_comparison.csv if present
        for artifact in ["params.yaml", "model_comparison.csv"]:
            src = os.path.join(local_dir, artifact)
            if os.path.exists(src):
                dst = os.path.join(output_dir, artifact)
                shutil.copy2(src, dst)
                print(f"  ✅ Copied {artifact}")

        # Create metadata file
        print(f"\n[4/4] Creating deployment metadata...")
        metadata = {
            "model_name": model_name,
            "model_version": version,
            "alias": args.alias if args.alias else "none",
            "tracking_uri": mlflow_cfg["tracking_uri"],
            "downloaded_from": model_uri,
        }

        metadata_path = os.path.join(output_dir, "deployment_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, indent=2, fp=f)
        print(f"  ✅ Saved deployment_metadata.json")

        # Final summary
        print(f"\n{'=' * 70}")
        print(f"  ✅ MODEL DOWNLOAD COMPLETE!")
        print(f"{'=' * 70}")
        print(f"  📂 Output directory: {output_dir}")
        print(f"  🔢 Model version: {version}")
        print(f"  📦 Artifacts ready for Docker build")
        print(f"{'=' * 70}\n")

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
