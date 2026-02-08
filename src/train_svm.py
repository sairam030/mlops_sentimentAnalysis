"""
train_svm.py — Approach 1: SVM + SMOTE with MLflow
====================================================
Loads preprocessed features (with SMOTE already applied in pretrain.py --smote)
→ scales → trains SVM (RBF) → evaluates → logs to MLflow.

Usage:
    python src/pretrain.py --smote            # first: preprocess with SMOTE
    python src/train_svm.py                   # then: train
"""

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()  # load .env credentials (MLFLOW_TRACKING_*)

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
    # SMOTE train data is in the smote/ subfolder
    smote_dir = os.path.join(processed_dir, "smote")
    X_train = np.load(os.path.join(smote_dir, "train_vectors.npy"))
    y_train = np.load(os.path.join(smote_dir, "train_labels.npy"))
    # Test data is always in the base processed/ folder
    X_test = np.load(os.path.join(processed_dir, "test_vectors.npy"))
    y_test = np.load(os.path.join(processed_dir, "test_labels.npy"))

    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return X_train, y_train, X_test, y_test, metadata


def train_and_evaluate(X_train, y_train, X_test, y_test, metadata, params):
    svm_params = params["train"]["svm"]
    models_dir = params["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    label_map = metadata["label_map"]
    inv_label_map = {int(v): k for k, v in label_map.items()}
    target_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    print("[train] Training SVM (with SMOTE-balanced data)...")
    model = SVC(
        kernel=svm_params["kernel"],
        C=svm_params["C"],
        gamma=svm_params["gamma"],
        probability=True,
        random_state=svm_params["random_state"],
    )
    model.fit(X_train_scaled, y_train)
    print("[train] Training complete.")

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\n{'=' * 50}")
    print(f"  TEST ACCURACY (With SMOTE): {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"{'=' * 50}")
    print(report)

    # Save artifacts
    model_path = os.path.join(models_dir, "svm_sentiment.joblib")
    scaler_path = os.path.join(models_dir, "svm_scaler.joblib")
    info_path = os.path.join(models_dir, "svm_model_info.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    model_info = {
        "model": "SVM (RBF) + SMOTE",
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "features": metadata["features"],
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "smote": True,
        "svm_params": svm_params,
        "label_map": label_map,
    }
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    return model, scaler, model_info, model_path, scaler_path, info_path


def main():
    parser = argparse.ArgumentParser(description="Train SVM + SMOTE")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    processed_dir = params["paths"]["processed_dir"]
    mlflow_cfg = params["mlflow"]

    X_train, y_train, X_test, y_test, metadata = load_data(processed_dir)
    print(f"[train] Train: {X_train.shape}, Test: {X_test.shape}")

    # MLflow
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name="SVM_WithSMOTE"):
        svm_params = params["train"]["svm"]
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", svm_params["kernel"])
        mlflow.log_param("C", svm_params["C"])
        mlflow.log_param("gamma", svm_params["gamma"])
        mlflow.log_param("smote", True)
        mlflow.log_param("feature_type", metadata["features"])
        mlflow.log_param("feature_dim", metadata["feature_dim"])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])

        model, scaler, info, model_path, scaler_path, info_path = train_and_evaluate(
            X_train, y_train, X_test, y_test, metadata, params
        )

        mlflow.log_metric("accuracy", info["accuracy"])
        mlflow.log_metric("f1_macro", info["f1_macro"])
        mlflow.log_metric("f1_weighted", info["f1_weighted"])
        mlflow.log_metric("precision_macro", info["precision_macro"])
        mlflow.log_metric("recall_macro", info["recall_macro"])

        mlflow.sklearn.log_model(model, "svm_model")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(info_path)
        mlflow.log_artifact("params.yaml")

        print(f"\n[mlflow] Run logged: {mlflow.active_run().info.run_id}")

    print("[train] Done.")


if __name__ == "__main__":
    main()
