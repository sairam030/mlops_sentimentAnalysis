"""
evaluate.py — Evaluate all models, compare, and register the best in MLflow
=============================================================================
Scans all model info JSONs → compares accuracy → registers the winning model
in the MLflow Model Registry.

Usage:
    python src/evaluate.py                         # Evaluate all & register best
    python src/evaluate.py --model svm_baseline    # Evaluate a single model
    python src/evaluate.py --compare               # Print comparison table only
"""

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()  # load .env credentials (MLFLOW_TRACKING_*)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.transformers
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_test_data(processed_dir):
    X_test = np.load(os.path.join(processed_dir, "test_vectors.npy"))
    y_test = np.load(os.path.join(processed_dir, "test_labels.npy"))
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return X_test, y_test, metadata


# ─────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────
def evaluate_svm(model_path, scaler_path, X_test, y_test, metadata, output_dir):
    """Evaluate a single SVM model and save confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    label_map = metadata["label_map"]
    inv_label_map = {int(v): k for k, v in label_map.items()}
    target_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
    }

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\n{'='*50}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"{'='*50}")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    _save_confusion_matrix(cm, target_names, output_dir)

    return metrics


def _save_confusion_matrix(cm, target_names, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved to {cm_path}")


# ─────────────────────────────────────────────
# Compare all models & register the best
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "SVM_NoSMOTE": {
        "info_file": "svm_baseline_info.json",
        "model_file": "svm_baseline_no_smote.joblib",
        "scaler_file": "svm_baseline_scaler.joblib",
        "type": "sklearn",
    },
    "SVM_SMOTE": {
        "info_file": "svm_model_info.json",
        "model_file": "svm_sentiment.joblib",
        "scaler_file": "svm_scaler.joblib",
        "type": "sklearn",
    },
    "DistilBERT": {
        "info_file": "distilbert_info.json",
        "model_dir": "distilbert_sentiment/",
        "type": "transformers",
    },
}


def compare_and_register(models_dir, mlflow_cfg):
    """
    1. Load all model info JSONs.
    2. Print comparison table.
    3. Find the best model by accuracy.
    4. Register the best model in MLflow Model Registry.
    """
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    registered_model_name = mlflow_cfg.get("registered_model_name", "sentiment-best-model")

    # --- Collect all model results ---
    results = []
    for model_name, cfg in MODEL_REGISTRY.items():
        info_path = os.path.join(models_dir, cfg["info_file"])
        if not os.path.exists(info_path):
            print(f"[evaluate] Skipping {model_name} — {info_path} not found")
            continue

        with open(info_path) as f:
            info = json.load(f)

        results.append({
            "name": model_name,
            "accuracy": info["accuracy"],
            "f1_macro": info.get("f1_macro", 0.0),
            "train_samples": info.get("train_samples", "N/A"),
            "info": info,
            "cfg": cfg,
        })

    if not results:
        print("[evaluate] No model info files found. Train models first.")
        return

    # --- Print comparison table ---
    df = pd.DataFrame([{
        "Model": r["name"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "F1 Macro": f"{r['f1_macro']:.4f}",
        "Train Samples": r["train_samples"],
    } for r in results])

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON")
    print("=" * 65)
    print(df.to_string(index=False))

    # --- Find best model ---
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n{'='*65}")
    print(f"  🏆 BEST MODEL: {best['name']}")
    print(f"     Accuracy:  {best['accuracy']:.4f} ({best['accuracy']*100:.1f}%)")
    print(f"     F1 Macro:  {best['f1_macro']:.4f}")
    print(f"{'='*65}")

    # --- Register best model in MLflow ---
    print(f"\n[evaluate] Registering '{best['name']}' in MLflow as '{registered_model_name}'...")

    with mlflow.start_run(run_name=f"BestModel_{best['name']}"):
        # Log all metrics
        mlflow.log_param("best_model", best["name"])
        mlflow.log_metric("accuracy", best["accuracy"])
        mlflow.log_metric("f1_macro", best["f1_macro"])

        # Log all info files for comparison record
        for r in results:
            info_path = os.path.join(models_dir, r["cfg"]["info_file"])
            mlflow.log_artifact(info_path, artifact_path="model_infos")

        # Log & register the actual best model
        cfg = best["cfg"]
        if cfg["type"] == "sklearn":
            model = joblib.load(os.path.join(models_dir, cfg["model_file"]))
            mlflow.sklearn.log_model(
                model,
                artifact_path="best_model",
                registered_model_name=registered_model_name,
            )
            # Also log scaler as artifact
            scaler_path = os.path.join(models_dir, cfg["scaler_file"])
            mlflow.log_artifact(scaler_path, artifact_path="best_model")

        elif cfg["type"] == "transformers":
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
                pipeline,
            )

            model_dir = os.path.join(models_dir, cfg["model_dir"])
            print(f"[evaluate] Loading DistilBERT from {model_dir}...")
            hf_model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            hf_tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
            hf_pipeline = pipeline(
                "text-classification", model=hf_model, tokenizer=hf_tokenizer
            )

            print(f"[evaluate] Logging DistilBERT to MLflow Model Registry...")
            mlflow.transformers.log_model(
                transformers_model=hf_pipeline,
                artifact_path="best_model",
                registered_model_name=registered_model_name,
                pip_requirements=[
                    f"transformers=={__import__('transformers').__version__}",
                    f"torch=={__import__('torch').__version__}",
                    "tokenizers",
                ],
            )

        else:
            print(f"[evaluate] Unknown model type: {cfg['type']}. Skipping registration.")

        # Log comparison table
        comparison_path = os.path.join(models_dir, "model_comparison.csv")
        df.to_csv(comparison_path, index=False)
        mlflow.log_artifact(comparison_path)
        mlflow.log_artifact("params.yaml")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[mlflow] ✅ Model registered as '{registered_model_name}'")
        print(f"[mlflow] Run ID: {run_id}")
        print(f"[mlflow] View at: mlflow ui  (then open http://localhost:5000)")

    print("[evaluate] Done.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate & register best model")
    parser.add_argument(
        "--model",
        choices=["svm_baseline", "svm_smote"],
        default=None,
        help="Evaluate a single SVM model",
    )
    parser.add_argument("--compare", action="store_true",
                        help="Compare all models (no registration)")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    processed_dir = params["paths"]["processed_dir"]
    models_dir = params["paths"]["models_dir"]
    mlflow_cfg = params["mlflow"]

    # --- Single model evaluation ---
    if args.model:
        X_test, y_test, metadata = load_test_data(processed_dir)
        if args.model == "svm_baseline":
            evaluate_svm(
                os.path.join(models_dir, "svm_baseline_no_smote.joblib"),
                os.path.join(models_dir, "svm_baseline_scaler.joblib"),
                X_test, y_test, metadata,
                os.path.join(models_dir, "eval_svm_baseline"),
            )
        elif args.model == "svm_smote":
            evaluate_svm(
                os.path.join(models_dir, "svm_sentiment.joblib"),
                os.path.join(models_dir, "svm_scaler.joblib"),
                X_test, y_test, metadata,
                os.path.join(models_dir, "eval_svm_smote"),
            )
        return

    # --- Compare only (no registration) ---
    if args.compare:
        results = []
        for model_name, cfg in MODEL_REGISTRY.items():
            info_path = os.path.join(models_dir, cfg["info_file"])
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                results.append({
                    "Model": model_name,
                    "Accuracy": f"{info['accuracy']:.4f}",
                    "F1 Macro": f"{info.get('f1_macro', 'N/A')}",
                })
        if results:
            print(pd.DataFrame(results).to_string(index=False))
        return

    # --- Default: Compare all & register best ---
    compare_and_register(models_dir, mlflow_cfg)


if __name__ == "__main__":
    main()