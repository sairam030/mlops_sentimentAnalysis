"""
train_bert.py — Approach 2: Fine-tune DistilBERT with MLflow
==============================================================
Loads raw CSV → tokenizes → fine-tunes DistilBERT → evaluates → logs to MLflow.

Usage:
    python src/train_bert.py
    python src/train_bert.py --params params.yaml
"""

import argparse
import json
import os
import re

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Light cleaning — DistilBERT handles punctuation well."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare(raw_path, test_size, val_ratio, random_state):
    """Load raw CSV, clean, encode labels, split into train/val/test."""
    print("[bert] Loading raw data...")
    df = pd.read_csv(raw_path)
    df["Comment"] = df["Comment"].fillna("").astype(str).apply(clean_text)
    df = df[df["Sentiment"].isin(["positive", "negative", "neutral"])]

    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    df["label"] = df["Sentiment"].map(label_map)

    # 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio, random_state=random_state, stratify=temp_df["label"]
    )

    print(f"[bert] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df, label_map, inv_label_map


def tokenize_datasets(train_df, val_df, test_df, tokenizer, max_length):
    """Convert DataFrames to tokenized HuggingFace Datasets."""
    def make_ds(dataframe):
        ds = Dataset.from_pandas(dataframe[["Comment", "label"]].reset_index(drop=True))
        ds = ds.rename_column("Comment", "text")
        return ds

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = make_ds(train_df).map(tokenize_fn, batched=True, batch_size=256)
    val_ds = make_ds(val_df).map(tokenize_fn, batched=True, batch_size=256)
    test_ds = make_ds(test_df).map(tokenize_fn, batched=True, batch_size=256)

    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    raw_path = params["paths"]["raw_data"]
    models_dir = params["paths"]["models_dir"]
    bert_params = params["train"]["bert"]
    mlflow_cfg = params["mlflow"]

    model_name = bert_params["base_model"]
    epochs = bert_params["epochs"]
    lr = bert_params["learning_rate"]
    batch_size = bert_params["batch_size"]
    max_length = bert_params["max_length"]
    weight_decay = bert_params["weight_decay"]
    warmup_ratio = bert_params["warmup_ratio"]

    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]
    output_dir = os.path.join(models_dir, "distilbert_sentiment")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bert] Device: {device}")

    # --- Data ---
    train_df, val_df, test_df, label_map, inv_label_map = load_and_prepare(
        raw_path, test_size=test_size, val_ratio=0.5, random_state=random_state
    )

    # --- Tokenize ---
    print(f"[bert] Tokenizing with {model_name}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_ds, val_ds, test_ds = tokenize_datasets(train_df, val_df, test_df, tokenizer, max_length)

    # --- Model ---
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=inv_label_map, label2id=label_map
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=random_state,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # --- MLflow ---
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name="DistilBERT_FineTuned"):
        mlflow.log_param("model_type", "DistilBERT")
        mlflow.log_param("base_model", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("train_samples", len(train_ds))
        mlflow.log_param("val_samples", len(val_ds))
        mlflow.log_param("test_samples", len(test_ds))

        # --- Train ---
        print(f"[bert] Fine-tuning {model_name} for {epochs} epochs...")
        train_result = trainer.train()
        print(f"[bert] Training complete in {train_result.metrics['train_runtime']:.0f}s")

        mlflow.log_metric("train_loss", train_result.metrics["train_loss"])

        # --- Evaluate on test set ---
        print("[bert] Evaluating on test set...")
        test_results = trainer.predict(test_ds)
        y_pred = np.argmax(test_results.predictions, axis=-1)
        y_true = test_results.label_ids

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        precision_macro = precision_score(y_true, y_pred, average="macro")
        recall_macro = recall_score(y_true, y_pred, average="macro")

        print(f"\n{'='*50}")
        print(f"  TEST ACCURACY (DistilBERT): {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"{'='*50}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("recall_macro", recall_macro)

        # --- Save model ---
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        model_info = {
            "model": "DistilBERT fine-tuned (3-class)",
            "base_model": model_name,
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "num_labels": 3,
            "label_map": label_map,
            "max_length": max_length,
            "epochs": epochs,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
        }
        info_path = os.path.join(output_dir, "model_info.json")
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        # Log artifacts
        mlflow.log_artifact(info_path)
        mlflow.log_artifact("params.yaml")

        # Log the HF model directory
        mlflow.log_artifacts(output_dir, artifact_path="distilbert_model")

        print(f"\n[mlflow] Run logged: {mlflow.active_run().info.run_id}")

    print("[bert] Done.")


if __name__ == "__main__":
    main()