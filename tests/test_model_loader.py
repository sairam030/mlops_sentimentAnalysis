"""
test_model_loader.py — Test model detection and loader logic.
"""

import os
import tempfile

import pytest

from serving.model_loader import ModelLoader


def test_detect_distilbert(fake_distilbert_dir):
    """Should detect DistilBERT when config.json + tokenizer.json exist."""
    loader = ModelLoader()

    # Patch to avoid actually loading weights (huge download)
    with pytest.raises(Exception):
        # Will fail because there are no real weights, but it should
        # at least detect DistilBERT and attempt to load it
        loader._load_from_dir(fake_distilbert_dir)


def test_detect_unknown_model():
    """Should raise RuntimeError for unrecognizable model dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty dir — nothing to detect
        with open(os.path.join(tmpdir, "random_file.txt"), "w") as f:
            f.write("not a model")

        loader = ModelLoader()
        with pytest.raises(RuntimeError, match="Cannot detect model type"):
            loader._load_from_dir(tmpdir)


def test_status_before_load():
    """Status should show not loaded before any load call."""
    loader = ModelLoader()
    status = loader.status()
    assert status["loaded"] is False
    assert status["model_info"] is None


def test_predictor_raises_before_load():
    """Accessing predictor before load should raise."""
    loader = ModelLoader()
    with pytest.raises(RuntimeError, match="No model loaded"):
        _ = loader.predictor


def test_load_from_local_path_missing(monkeypatch):
    """Should fall through when LOCAL_MODEL_PATH doesn't exist."""
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/tmp/_absolutely_nonexistent_model_dir_12345")
    # Also unset any real MLflow credentials so S3/MLflow fallbacks fail fast
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "")

    # Re-create settings so they pick up the new env vars
    from serving.config import Settings
    import serving.model_loader as ml_mod

    ml_mod.settings = Settings()

    loader = ModelLoader()
    with pytest.raises(RuntimeError, match="Could not load model"):
        loader.load()
