"""
conftest.py — Shared fixtures for the test suite.
"""

import json
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure tests don't leak env vars or touch real MLflow/S3."""
    monkeypatch.setenv("LOCAL_MODEL_PATH", "")
    monkeypatch.setenv("MODEL_STAGE", "staging")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "test-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "test-pass")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")


@pytest.fixture
def fake_distilbert_dir():
    """Create a temp dir that looks like a DistilBERT model (no real weights)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal files to pass detection
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump({"model_type": "distilbert", "num_labels": 3}, f)
        with open(os.path.join(tmpdir, "tokenizer.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(tmpdir, "tokenizer_config.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(tmpdir, "distilbert_info.json"), "w") as f:
            json.dump({
                "model_type": "distilbert",
                "base_model": "distilbert-base-uncased",
                "accuracy": 0.855,
                "f1_macro": 0.808,
                "num_labels": 3,
                "label_map": {"negative": 0, "neutral": 1, "positive": 2},
            }, f)
        yield tmpdir
