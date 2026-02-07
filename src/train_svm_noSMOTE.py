"""
train_svm_noSMOTE.py — Approach 0: SVM Baseline (No SMOTE) with MLflow
=======================================================================
Loads preprocessed features → scales → trains SVM (RBF) → evaluates → logs to MLflow.

Usage:
    python src/train_svm_noSMOTE.py
    python src/train_svm_noSMOTE.py --params params.yaml
"""

import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_dir):
    """Load preprocessed train/test vectors and labels."""
    X_train = np.load(os.path.join(processed_dir, "train_vectors.npy"))
    y_train = np.load(os.path.join(processed_dir, "train_labels.npy"))
    X_test = np.load(os.path.join(processed_dir, "test_vectors.npy"))
    y_test = np.load(os.path.join(processed_dir, "test_labels.npy"))

    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return X_train, y_train, X_test, y_test, metadata


def train_and_evaluate(X_train, y_train, X_test, y_test, metadata, params):
    """Train SVM, evaluate, and log everything to MLflow."""

    svm_params = params["train"]["svm"]
    models_dir = params["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    label_map = metadata["label_map"]
    inv_label_map = {int(v): k for k, v in label_map.items()}
    target_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train SVM ---
    print("[train] Training SVM (No SMOTE)...")
    model = SVC(
        kernel=svm_params["kernel"],
        C=svm_params["C"],
        gamma=svm_params["gamma"],
        probability=True,
        random_state=svm_params["random_state"],
    )
    model.fit(X_train_scaled, y_train)
    print("[train] Training complete.")

    # --- Evaluate ---
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY (No SMOTE): {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*50}")
    print(report)

    # --- Save artifacts locally ---
    model_path = os.path.join(models_dir, "svm_baseline_no_smote.joblib")
    scaler_path = os.path.join(models_dir, "svm_baseline_scaler.joblib")
    info_path = os.path.join(models_dir, "svm_baseline_info.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    model_info = {
        "model": "SVM (RBF) — No SMOTE Baseline",
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "features": metadata["features"],
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "smote": False,
        "svm_params": svm_params,
        "label_map": label_map,
    }
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    return model, scaler, model_info, model_path, scaler_path, info_path


def main():
    parser = argparse.ArgumentParser(description="Train SVM Baseline (No SMOTE)")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    processed_dir = params["paths"]["processed_dir"]
    mlflow_cfg = params["mlflow"]

    # --- Load data ---
    X_train, y_train, X_test, y_test, metadata = load_data(processed_dir)
    print(f"[train] Train: {X_train.shape}, Test: {X_test.shape}")

    # --- MLflow tracking ---
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name="SVM_NoSMOTE_Baseline"):
        # Log params
        svm_params = params["train"]["svm"]
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", svm_params["kernel"])
        mlflow.log_param("C", svm_params["C"])
        mlflow.log_param("gamma", svm_params["gamma"])
        mlflow.log_param("smote", False)
        mlflow.log_param("feature_type", metadata["features"])
        mlflow.log_param("feature_dim", metadata["feature_dim"])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])

        # Train & evaluate
        model, scaler, info, model_path, scaler_path, info_path = train_and_evaluate(
            X_train, y_train, X_test, y_test, metadata, params
        )

        # Log metrics
        mlflow.log_metric("accuracy", info["accuracy"])
        mlflow.log_metric("f1_macro", info["f1_macro"])
        mlflow.log_metric("f1_weighted", info["f1_weighted"])
        mlflow.log_metric("precision_macro", info["precision_macro"])
        mlflow.log_metric("recall_macro", info["recall_macro"])

        # Log model & artifacts
        mlflow.sklearn.log_model(model, "svm_model")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(info_path)
        mlflow.log_artifact("params.yaml")

        print(f"\n[mlflow] Run logged: {mlflow.active_run().info.run_id}")
        print(f"[mlflow] Experiment: {mlflow_cfg['experiment_name']}")

    print("[train] Done.")


if __name__ == "__main__":
    main()