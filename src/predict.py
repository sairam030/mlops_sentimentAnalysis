"""
predict.py — Inference: Predict sentiment for new comments
============================================================
Usage:
    python src/predict.py "This video is amazing!"
    python src/predict.py "Terrible content" "Great tutorial" "10 min long"
    python src/predict.py --model svm_smote "Love it!"
"""

import argparse
import json
import os
import re

import joblib
import numpy as np
import yaml
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


class SentimentPredictor:
    def __init__(self, model_path, scaler_path, metadata_path, embedding_model):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.st_model = SentenceTransformer(embedding_model)
        self.analyzer = SentimentIntensityAnalyzer()

        with open(metadata_path) as f:
            metadata = json.load(f)
        self.label_map = metadata["label_map"]
        self.inv_label_map = {int(v): k for k, v in self.label_map.items()}

    def predict(self, texts):
        """Predict sentiment for a list of raw text strings."""
        # Embeddings (on cleaned text)
        cleaned = [clean_text(t) for t in texts]
        embeddings = self.st_model.encode(cleaned)

        # VADER (on raw text — needs punctuation, caps, etc.)
        vader_features = []
        for t in texts:
            s = self.analyzer.polarity_scores(t)
            vader_features.append([s["compound"], s["pos"], s["neg"], s["neu"]])
        vader_features = np.array(vader_features)

        # Combine & scale
        features = np.hstack([embeddings, vader_features])
        features_scaled = self.scaler.transform(features)

        # Predict
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)

        results = []
        for i, text in enumerate(texts):
            pred_label = self.inv_label_map[predictions[i]]
            confidence = probabilities[i].max()
            results.append(
                {
                    "text": text,
                    "prediction": pred_label,
                    "confidence": f"{confidence:.1%}",
                }
            )
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment")
    parser.add_argument("texts", nargs="+", help="Text(s) to classify")
    parser.add_argument(
        "--model",
        choices=["svm_baseline", "svm_smote"],
        default="svm_baseline",
        help="Which model to use",
    )
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    models_dir = params["paths"]["models_dir"]
    processed_dir = params["paths"]["processed_dir"]
    embedding_model = params["preprocess"]["embedding_model"]

    metadata_path = os.path.join(processed_dir, "metadata.json")

    if args.model == "svm_baseline":
        model_path = os.path.join(models_dir, "svm_baseline_no_smote.joblib")
        scaler_path = os.path.join(models_dir, "svm_baseline_scaler.joblib")
    elif args.model == "svm_smote":
        model_path = os.path.join(models_dir, "svm_sentiment.joblib")
        scaler_path = os.path.join(models_dir, "svm_scaler.joblib")

    predictor = SentimentPredictor(model_path, scaler_path, metadata_path, embedding_model)
    results = predictor.predict(args.texts)

    print(f"\n{'=' * 60}")
    print(f"  PREDICTIONS ({args.model})")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  [{r['confidence']:>5s}] {r['prediction']:>8s}  │  {r['text'][:70]}")
    print()


if __name__ == "__main__":
    main()
