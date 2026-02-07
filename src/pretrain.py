"""
pretrain.py — Preprocessing & Feature Engineering Pipeline
==========================================================
Reads raw CSV → cleans text → generates embeddings + VADER features
→ splits data → saves train/test artifacts to data/processed/

Usage:
    python src/pretrain.py                     # uses params.yaml defaults
    python src/pretrain.py --smote             # also apply SMOTE balancing
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(raw_path: str) -> pd.DataFrame:
    """Load raw CSV, clean text, encode labels."""
    print(f"[pretrain] Loading data from {raw_path}")
    df = pd.read_csv(raw_path)
    df["Comment"] = df["Comment"].fillna("").astype(str)
    df["raw_comment"] = df["Comment"]  # Keep raw for VADER
    df["Comment"] = df["Comment"].apply(clean_text)
    df = df[df["Sentiment"].isin(["positive", "negative", "neutral"])]

    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    df["label"] = df["Sentiment"].map(label_map)

    print(f"[pretrain] Samples: {len(df)}")
    print(f"[pretrain] Class distribution:\n{df['Sentiment'].value_counts().to_string()}")
    return df, label_map


def generate_features(df: pd.DataFrame, embedding_model: str) -> np.ndarray:
    """Generate 388-d feature vectors: embeddings (384) + VADER (4)."""
    print(f"[pretrain] Generating sentence embeddings ({embedding_model})...")
    st_model = SentenceTransformer(embedding_model)
    embeddings = st_model.encode(df["Comment"].tolist(), show_progress_bar=True)

    print("[pretrain] Extracting VADER sentiment features...")
    analyzer = SentimentIntensityAnalyzer()
    vader_features = []
    for text in df["raw_comment"]:
        s = analyzer.polarity_scores(text)
        vader_features.append([s["compound"], s["pos"], s["neg"], s["neu"]])
    vader_features = np.array(vader_features)

    scaler = StandardScaler()
    vader_scaled = scaler.fit_transform(vader_features)
    combined = np.hstack([embeddings, vader_scaled])

    print(f"[pretrain] Feature matrix: {combined.shape}")
    return combined


def split_and_save(X, y, label_map, output_dir, test_size, random_state, use_smote=False, smote_seed=42):
    """Split data, optionally apply SMOTE, save artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[pretrain] Train: {X_train.shape}, Test: {X_test.shape}")

    if use_smote:
        from imblearn.over_sampling import SMOTE
        print("[pretrain] Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=smote_seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[pretrain] After SMOTE — Train: {X_train.shape}")

    # Determine save directory — SMOTE goes to a subfolder
    if use_smote:
        save_dir = os.path.join(output_dir, "smote")
    else:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Save train data (to smote/ subfolder if SMOTE, else processed/)
    np.save(os.path.join(save_dir, "train_vectors.npy"), X_train)
    np.save(os.path.join(save_dir, "train_labels.npy"), y_train)

    # Save test data ONLY for non-SMOTE run (test set is always the same)
    if not use_smote:
        np.save(os.path.join(output_dir, "test_vectors.npy"), X_test)
        np.save(os.path.join(output_dir, "test_labels.npy"), y_test)

    metadata = {
        "label_map": label_map,
        "embedding_model": "all-MiniLM-L6-v2",
        "feature_dim": int(X_train.shape[1]),
        "features": "sentence_embeddings(384) + vader_scaled(4)",
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "smote_applied": use_smote,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[pretrain] Saved artifacts to {save_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess & feature engineering")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE balancing")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    raw_path = params["paths"]["raw_data"]
    output_dir = params["paths"]["processed_dir"]
    embedding_model = params["preprocess"]["embedding_model"]
    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]
    smote_seed = params["train"]["smote"]["random_state"]

    df, label_map = load_and_clean(raw_path)
    X = generate_features(df, embedding_model)
    y = df["label"].values

    metadata = split_and_save(
        X, y, label_map, output_dir,
        test_size=test_size,
        random_state=random_state,
        use_smote=args.smote,
        smote_seed=smote_seed,
    )
    print("[pretrain] Done.")
    return metadata


if __name__ == "__main__":
    main()
