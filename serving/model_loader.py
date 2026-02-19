"""
model_loader.py — Load the best model from S3 (primary) or MLflow (fallback).
===============================================================================
Supports both DistilBERT (transformers) and SVM (sklearn) models.

Load priority:
  1. S3 bucket  (fast, reliable — set by promote_model.py)
  2. MLflow Model Registry  (direct pull by alias — fallback)
"""

import json
import os
import shutil
import tarfile
from abc import ABC, abstractmethod
from typing import Any


from serving.config import settings
from serving.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Abstract predictor interface
# ─────────────────────────────────────────────
class BasePredictor(ABC):
    """Common interface for all model types."""

    @abstractmethod
    def predict(self, texts: list[str]) -> list[dict]:
        """Return list of {text, prediction, confidence}."""
        ...

    @abstractmethod
    def model_info(self) -> dict:
        """Return metadata about the loaded model."""
        ...


# ─────────────────────────────────────────────
# DistilBERT predictor
# ─────────────────────────────────────────────
class DistilBERTPredictor(BasePredictor):
    def __init__(self, model_dir: str, info: dict):
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            pipeline,
        )

        logger.info(f"Loading DistilBERT from {model_dir}")
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self._pipeline = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, top_k=None
        )
        self._info = info
        self._label_map = info.get("label_map", {})
        logger.info("DistilBERT loaded successfully")

    def predict(self, texts: list[str]) -> list[dict]:
        results: list[dict[str, Any]] = []
        outputs: list[list[dict[str, Any]]] = self._pipeline(texts, batch_size=32)  # type: ignore[assignment]
        for text, preds in zip(texts, outputs):
            # preds is a list of {label, score} sorted by score desc
            best: dict[str, Any] = max(preds, key=lambda x: x["score"])  # pyright: ignore[reportArgumentType]
            results.append({
                "text": text,
                "prediction": best["label"],
                "confidence": round(float(best["score"]), 4),
                "all_scores": {p["label"]: round(float(p["score"]), 4) for p in preds},  # pyright: ignore[reportArgumentType]
            })
        return results

    def model_info(self) -> dict:
        return {
            "model_type": "DistilBERT",
            "base_model": self._info.get("base_model", "distilbert-base-uncased"),
            "accuracy": self._info.get("accuracy"),
            "f1_macro": self._info.get("f1_macro"),
            "num_labels": self._info.get("num_labels", 3),
        }


# ─────────────────────────────────────────────
# SVM predictor
# ─────────────────────────────────────────────
class SVMPredictor(BasePredictor):
    def __init__(self, model_dir: str, info: dict):
        import joblib
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._np = np

        logger.info(f"Loading SVM from {model_dir}")
        self._model = joblib.load(os.path.join(model_dir, "model.joblib"))
        self._scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._analyzer = SentimentIntensityAnalyzer()
        self._info = info
        self._label_map = info.get("label_map", {})
        self._inv_label_map = {int(v): k for k, v in self._label_map.items()}
        logger.info("SVM loaded successfully")

    def predict(self, texts: list[str]) -> list[dict]:
        import re

        cleaned = [re.sub(r"\s+", " ", re.sub(r"http\S+|www\.\S+", "", t.lower())).strip() for t in texts]
        embeddings = self._st_model.encode(cleaned)

        vader_feats = []
        for t in texts:
            s = self._analyzer.polarity_scores(t)
            vader_feats.append([s["compound"], s["pos"], s["neg"], s["neu"]])
        vader_feats = self._np.array(vader_feats)

        features = self._np.hstack([embeddings, vader_feats])
        features_scaled = self._scaler.transform(features)

        predictions = self._model.predict(features_scaled)
        probabilities = self._model.predict_proba(features_scaled)

        results = []
        for i, text in enumerate(texts):
            pred_label = self._inv_label_map[predictions[i]]
            confidence = float(probabilities[i].max())
            all_scores = {
                self._inv_label_map[j]: round(float(probabilities[i][j]), 4)
                for j in range(len(probabilities[i]))
            }
            results.append({
                "text": text,
                "prediction": pred_label,
                "confidence": round(confidence, 4),
                "all_scores": all_scores,
            })
        return results

    def model_info(self) -> dict:
        return {
            "model_type": "SVM",
            "accuracy": self._info.get("accuracy"),
            "f1_macro": self._info.get("f1_macro"),
            "smote": self._info.get("smote", False),
        }


# ─────────────────────────────────────────────
# Model loader: S3 → MLflow fallback
# ─────────────────────────────────────────────
class ModelLoader:
    """Loads model from S3 (primary) or MLflow (fallback)."""

    def __init__(self):
        self._predictor: BasePredictor | None = None
        self._model_version: str | None = None
        self._source: str | None = None

    @property
    def predictor(self) -> BasePredictor:
        if self._predictor is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._predictor

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def status(self) -> dict:
        predictor = self._predictor
        return {
            "loaded": self.is_loaded,
            "model_stage": settings.model_stage,
            "model_version": self._model_version,
            "source": self._source,
            "model_info": predictor.model_info() if predictor is not None else None,
        }

    def load(self) -> None:
        """Try local path first, then S3, then MLflow direct."""
        logger.info(f"Loading model for stage='{settings.model_stage}'...")

        # --- Try local path first (dev mode) ---
        if settings.local_model_path:
            local_path = settings.local_model_path
            if os.path.isdir(local_path):
                logger.info(f"Loading from local path: {local_path}")
                self._load_from_dir(local_path)
                self._source = f"local://{local_path}"
                return
            else:
                logger.warning(f"LOCAL_MODEL_PATH={local_path} not found, trying S3...")

        # --- Try S3 ---
        try:
            self._load_from_s3()
            return
        except Exception as e:
            logger.warning(f"S3 load failed ({e}), falling back to MLflow direct...")

        # --- Fallback: MLflow direct ---
        try:
            self._load_from_mlflow()
            return
        except Exception as e:
            logger.error(f"MLflow direct load also failed: {e}")
            raise RuntimeError(
                f"Could not load model for stage '{settings.model_stage}' "
                f"from S3 or MLflow. Check configuration."
            ) from e

    def _load_from_s3(self) -> None:
        """Download model archive from S3 and load locally."""
        import boto3

        s3 = boto3.client("s3")
        s3_key = f"{settings.s3_prefix}/{settings.model_stage}/model.tar.gz"

        logger.info(f"Downloading from s3://{settings.s3_bucket}/{s3_key}")

        local_dir = settings.local_model_dir
        os.makedirs(local_dir, exist_ok=True)
        archive_path = os.path.join(local_dir, "model.tar.gz")

        s3.download_file(settings.s3_bucket, s3_key, archive_path)

        # Extract
        model_dir = os.path.join(local_dir, "model")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(model_dir)

        self._load_from_dir(model_dir)
        self._source = f"s3://{settings.s3_bucket}/{s3_key}"

    def _load_from_mlflow(self) -> None:
        """Pull model directly from MLflow registry by alias."""
        import mlflow

        settings.validate()
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]
        model_name = settings.model_name
        alias = settings.model_stage

        logger.info(f"Pulling model '{model_name}' with alias '{alias}' from MLflow...")

        # Try alias first, fallback to latest version
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            version = mv.version
        except Exception:
            logger.warning(f"No alias '{alias}' found. Using latest version.")
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise RuntimeError(f"No versions found for model '{model_name}'") from None
            latest = max(versions, key=lambda v: int(v.version))
            version = latest.version

        self._model_version = str(version)
        model_uri = f"models:/{model_name}/{version}"

        logger.info(f"Downloading model from {model_uri}")
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)  # pyright: ignore[reportPrivateImportUsage]

        self._load_from_dir(local_dir)
        self._source = f"mlflow://{model_name}/v{version}"

    def _load_from_dir(self, model_dir: str) -> None:
        """Detect model type from files and instantiate the right predictor."""
        info = {}

        # Look for info JSON
        for fname in [
            "distilbert_info.json",
            "svm_model_info.json",
            "svm_baseline_info.json",
            "model_info.json",
        ]:
            candidate = os.path.join(model_dir, fname)
            if os.path.exists(candidate):
                with open(candidate) as f:
                    info = json.load(f)
                break

        # Detect model type
        has_config = os.path.exists(os.path.join(model_dir, "config.json"))
        has_tokenizer = os.path.exists(os.path.join(model_dir, "tokenizer.json"))
        has_joblib = any(f.endswith(".joblib") for f in os.listdir(model_dir))

        if has_config and has_tokenizer:
            logger.info("Detected DistilBERT model")
            self._predictor = DistilBERTPredictor(model_dir, info)
        elif has_joblib:
            logger.info("Detected SVM model")
            self._predictor = SVMPredictor(model_dir, info)
        else:
            # Check subdirectories (MLflow artifacts structure)
            for subdir in os.listdir(model_dir):
                subpath = os.path.join(model_dir, subdir)
                if os.path.isdir(subpath):
                    if os.path.exists(os.path.join(subpath, "config.json")):
                        self._predictor = DistilBERTPredictor(subpath, info)
                        return
            raise RuntimeError(
                f"Cannot detect model type in {model_dir}. "
                f"Files: {os.listdir(model_dir)}"
            )

        logger.info(f"Model loaded: {self._predictor.model_info()}")


# Singleton
model_loader = ModelLoader()
